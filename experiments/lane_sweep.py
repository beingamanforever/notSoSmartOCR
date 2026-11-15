"""Sweep independent Falcon OCR lanes on one GPU: throughput, latency, memory and identity.

Each lane is its own experiments/serve_falcon_layout.py process on its own loopback port
and reads one page at a time, as the service does today (bs=1 per document). Lanes pull
whole documents from one shared queue. For each lane count N the sweep adds lanes up to N
(lanes already running stay warm), has every lane read the first document once untimed,
then runs --repeats timed passes. Each pass queues the documents cyclically, at least
--docs-per-lane per lane, so N lanes have enough work to overlap.

Recorded per N: per-page round-trip latency p50/p90, pages/hour (median over passes),
peak VRAM per lane and for the GPU (nvidia-smi), mean GPU and CPU utilisation, minimum
available RAM and peak resident memory of one lane's process group. Per-lane VRAM is
empty when nvidia-smi cannot match memory to lane PIDs (a container's PID namespace, some
MPS setups); peak GPU memory over N is then the estimate.

Outputs must not depend on N. Every /read reply, without its "timings" field, is hashed
and compared with the first timed N=1 pass on the same host. A page whose reply already
changes between N=1 passes (sampled retries) is reported as unstable at N=1, not as a
lane difference.

Usage on a dedicated GPU host, never one that serves the live demo:

    experiments/lane_sweep.sh                              # N = 1,2,4,6,8, public demo pages
    experiments/lane_sweep.sh --mps --lanes 1,2,4,8        # lanes under NVIDIA MPS
    experiments/lane_sweep.sh --docs /data/public_pages --repeats 5
    python3 experiments/lane_sweep.py --dry-run            # stub lanes: checks the harness, no GPU

The launcher runs lanes with $FALCON_PYTHON and the production lane flags in $FALCON_ARGS;
falcon-perception must be importable by that interpreter (add a checkout to PYTHONPATH).
Page preparation mirrors process_document and FalconLayoutReader.read: PDFs render at the
pipeline DPI with their text-layer cells, and scans carry docTR words (--word-device; the
harness interpreter then needs docTR, or pass --word-device none). On a Deep Learning AMI,
unset LD_LIBRARY_PATH first: its cuDNN is older than the one bundled with torch.

Results go to artifacts/benchmarks/lane_sweep/<timestamp>/ (gitignored) and are rewritten
after every N: summary.json, summary.csv, summary.md, pages.csv, lane-<i>.log, baseline/
(normalized N=1 replies; two runs, for example with and without --mps, compare with
`diff -r`) and diffs/ (every reply that differs from its baseline). The exit status is 1
when a page that is stable at N=1 differs at a larger N, a request fails, or the sweep
stops early.

Safety: lane ports start at --base-port and 8193 (the live demo) is refused. A GPU that
already runs compute processes is refused unless --allow-shared-gpu. Another lane starts
only when available RAM and free VRAM cover one more lane's measured peak plus
--min-free-gib, and the sweep stops if available RAM falls below --min-free-gib. Lanes and
the MPS daemon are stopped by signalling the process group this script started for each
of them (its ID is that process's PID), never by name.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import queue
import random
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import cycle, islice
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "experiments" / "serve_falcon_layout.py"
LIVE_DEMO_PORT = 8193
# demo.MAX_PAGE_PIXELS; importing the demo module pulls in the whole web app.
MAX_PAGE_PIXELS = 32_000_000
PUBLIC_PAGES = ("academic_paper", "code_document", "financial_table", "formula_scan",
                "handwriting_notes", "scanned_form")
DEFAULT_DOCS = [REPO / "artifacts" / "demo" / f"{name}.png" for name in PUBLIC_PAGES]
SUMMARY_FIELDS = (
    "lanes", "pages_per_hour", "scaling", "p50_seconds", "p90_seconds", "requests",
    "failed_requests", "differing_pages", "peak_vram_mib_per_lane", "peak_gpu_memory_mib",
    "mean_gpu_util_pct", "mean_cpu_pct", "min_available_ram_gib", "peak_lane_rss_gib",
)
PAGE_FIELDS = ("lanes", "pass", "lane", "doc", "page", "seconds", "layout_seconds",
               "decode_seconds", "retry_seconds", "error", "sha256", "identical")


@dataclass(frozen=True)
class Page:
    doc: str
    number: int
    payload: Path


@dataclass(frozen=True)
class Lane:
    index: int
    port: int
    process: subprocess.Popen


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--stub-lane"]:
        serve_stub(int(argv[1]), drift="--drift" in argv)
        return 0
    # Everything after "--" goes to every lane, so typos in either part fail loudly.
    split = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:split])
    server_args = argv[split + 1 :]
    lane_counts = sorted({1, *(int(n) for n in args.lanes.split(","))})
    if lane_counts[0] < 1:
        raise SystemExit("--lanes must be positive")
    out = args.out or REPO / "artifacts/benchmarks/lane_sweep" / time.strftime("%Y%m%d-%H%M%S")
    ports = range(args.base_port, args.base_port + lane_counts[-1])
    if LIVE_DEMO_PORT in ports:
        raise SystemExit(f"lane ports {ports.start}-{ports.stop - 1} include the live demo port")
    for port in ports:
        with socket.socket() as probe:
            # As the lane servers bind: only a listening socket, not TIME_WAIT, blocks a port.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))  # OSError: the port is taken
    if not args.dry_run:
        if not gpu_rows("name", "gpu"):
            raise SystemExit("nvidia-smi found no GPU; use --dry-run to check the harness")
        busy = gpu_rows("pid,process_name", "compute-apps")
        if busy and not args.allow_shared_gpu:
            raise SystemExit(f"GPU already runs {busy}; use a dedicated host or --allow-shared-gpu")

    out.mkdir(parents=True)
    pages = prepare_pages(collect_docs(args.docs), out / "pages",
                          "none" if args.dry_run else args.word_device)
    docs = list(pages)
    meta = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_revision(),
        "gpu": gpu_rows("name,driver_version,memory.total", "gpu"),
        "cpus": os.cpu_count(),
        "ram_gib": meminfo_gib("MemTotal"),
        "mps": args.mps,
        "dry_run": args.dry_run,
        "server_args": server_args,
        "lane_counts": lane_counts,
        "repeats": args.repeats,
        "docs_per_lane": args.docs_per_lane,
        "documents": {doc: len(pages[doc]) for doc in docs},
        "word_device": "none" if args.dry_run else args.word_device,
    }
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))}
    lanes: list[Lane] = []
    sampler = Sampler(lanes, args.min_free_gib)
    sampler.start()
    mps = None
    rows: list[dict] = []
    records: list[dict] = []
    baseline: dict[str, str] = {}
    unstable: set[str] = set()
    stopped_early = None
    try:
        if args.mps:
            mps, mps_env = start_mps(out)
            env.update(mps_env)
        for n in lane_counts:
            phase_started = time.time()
            while len(lanes) < n:
                if lanes and not args.dry_run and (shortage := sampler.shortage_for_one_more_lane()):
                    stopped_early = f"not enough {shortage} for lane {len(lanes) + 1}"
                    break
                lanes.append(start_lane(len(lanes), args, server_args, env, out))
            if stopped_early:
                break
            run_pass(lanes, [docs[0]] * n, pages, sampler, args.request_timeout)  # warm-up
            timed_started = time.time()
            walls, pass_records = [], []
            for repeat in range(1, args.repeats + 1):
                queued = list(islice(cycle(docs), max(len(docs), args.docs_per_lane * n)))
                done, wall = run_pass(lanes, queued, pages, sampler, args.request_timeout)
                for record in done:
                    record.update(lanes=n, **{"pass": repeat})
                    compare(record, n, baseline, unstable, out)
                walls.append(wall)
                pass_records += done
                if sampler.low_memory.is_set():
                    stopped_early = f"available RAM fell below {args.min_free_gib} GiB at N={n}"
                    break
            records += pass_records
            rows.append(summarize(n, pass_records, walls, sampler, phase_started,
                                  timed_started, rows[0] if rows else None, unstable))
            write_results(out, meta, rows, records, unstable, stopped_early)
            print(json.dumps(rows[-1]), flush=True)
            if stopped_early:
                break
    finally:
        for lane in lanes:
            stop(lane.process)
        if mps:
            stop_mps(*mps)
        sampler.stopped.set()
    write_results(out, meta, rows, records, unstable, stopped_early)
    print(f"results: {out}")
    print((out / "summary.md").read_text())
    failed = any(row["failed_requests"] or row["differing_pages"] for row in rows)
    return 1 if failed or stopped_early else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] [-- serve_falcon_layout.py flags for every lane]")
    parser.add_argument("--lanes", default="1,2,4,6,8", help="lane counts; N=1 always runs")
    parser.add_argument("--docs", nargs="+", type=Path, default=DEFAULT_DOCS,
                        help="documents or folders (default: public pages in artifacts/demo)")
    parser.add_argument("--repeats", type=int, default=4, help="timed passes per lane count")
    parser.add_argument("--docs-per-lane", type=int, default=4,
                        help="minimum documents queued per lane in each pass")
    parser.add_argument("--mps", action="store_true", help="run lanes under NVIDIA MPS")
    parser.add_argument("--python", default=sys.executable, help="interpreter for lanes")
    parser.add_argument("--base-port", type=int, default=8300)
    parser.add_argument("--word-device", default="cpu", help="docTR device for scan words, or none")
    parser.add_argument("--start-timeout", type=float, default=1800, help="seconds per lane start")
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--out", type=Path, help="results folder (default under artifacts/benchmarks)")
    parser.add_argument("--dry-run", action="store_true", help="stub lanes, no GPU or models")
    parser.add_argument("--stub-drift", action="store_true",
                        help="dry run only: lanes after the first answer differently")
    return parser


def collect_docs(paths: list[Path]) -> list[Path]:
    sys.path.insert(0, str(REPO / "src"))
    from ocr_pipeline.pipeline import _source_kind

    files = sorted(p for path in paths for p in ([path] if path.is_file() else path.iterdir()))
    docs = [p for p in files if _source_kind(p) != "unsupported"]
    names = [p.name for p in docs]
    if not docs or len(set(names)) != len(names):
        raise SystemExit(f"need documents with distinct file names, got {names}")
    return docs


def prepare_pages(docs: list[Path], workdir: Path, word_device: str) -> dict[str, list[Page]]:
    """Render each document once and store each page's /read body, as the demo sends it."""
    from ocr_pipeline.pipeline import DEFAULT_PDF_DPI, _prepare_pages, _source_kind
    from ocr_pipeline.text_layer import read_cells

    detect_words = None
    if word_device != "none":
        if importlib.util.find_spec("doctr") is None:
            raise SystemExit("docTR is not importable by this interpreter; pass --word-device none")
        from ocr_pipeline.text_detection import detect_words
    pages = {}
    for doc in docs:
        target = workdir / doc.name
        target.mkdir(parents=True)
        prepared = _prepare_pages(doc, _source_kind(doc), target, DEFAULT_PDF_DPI, "pdftoppm",
                                  max_page_pixels=MAX_PAGE_PIXELS)
        pages[doc.name] = []
        for number, image_path in enumerate(prepared.pages, 1):
            # FalconLayoutReader.read: text-layer cells, else detected words when a detector runs.
            cells = read_cells(image_path)
            extra = (
                {"cells": cells} if cells
                else {"words": [w[:5] for w in detect_words(image_path, device=word_device)]}
                if detect_words else {}
            )
            body = {"image": base64.b64encode(image_path.read_bytes()).decode("ascii"), **extra}
            payload = target / f"page-{number}.json"
            payload.write_text(json.dumps(body, separators=(",", ":")))
            pages[doc.name].append(Page(doc.name, number, payload))
    return pages


def start_lane(index: int, args, server_args: list[str], env: dict, out: Path) -> Lane:
    port = args.base_port + index
    command = (
        [sys.executable, __file__, "--stub-lane", str(port),
         *(["--drift"] if args.stub_drift and index else [])]
        if args.dry_run
        else [args.python, "-u", str(SERVER), "--host", "127.0.0.1", "--port", str(port), *server_args]
    )
    with open(out / f"lane-{index}.log", "ab") as log:
        # Its own session, so stop() can signal exactly this lane and its compile workers.
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    lane = Lane(index, port, process)
    deadline = time.monotonic() + args.start_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"lane {index} exited with {process.returncode}; see lane-{index}.log")
        with contextlib.suppress(OSError, ValueError):
            health = json.loads(http(port, "/health", None, timeout=5))
            if health.get("pid") != process.pid:
                stop(process)
                raise RuntimeError(f"lane {index} answered from PID {health.get('pid')}, "
                                   f"not {process.pid}; --python must start the server directly")
            print(f"lane {index} ready on port {port}, PID {process.pid}", flush=True)
            return lane
        time.sleep(1 if args.dry_run else 5)
    stop(process)
    raise TimeoutError(f"lane {index} was not ready within {args.start_timeout} s")


def run_pass(lanes: list[Lane], queued: list[str], pages: dict[str, list[Page]],
             sampler: Sampler, timeout: float) -> tuple[list[dict], float]:
    """Lanes take whole documents from one queue and read their pages in order."""
    work: queue.SimpleQueue[str] = queue.SimpleQueue()
    for doc in queued:
        work.put(doc)
    records: list[dict] = []

    def drain(lane: Lane) -> None:
        while not sampler.low_memory.is_set():
            try:
                doc = work.get_nowait()
            except queue.Empty:
                return
            for page in pages[doc]:
                body = page.payload.read_bytes()
                started = time.perf_counter()
                try:
                    reply, error = json.loads(http(lane.port, "/read", body, timeout)), None
                except (OSError, ValueError) as failure:  # URLError and HTTPError are OSErrors
                    reply, error = None, repr(failure)
                seconds = time.perf_counter() - started
                timings = (reply or {}).get("timings") or {}
                records.append({
                    "lane": lane.index, "doc": doc, "page": page.number, "seconds": seconds,
                    **{key: timings.get(key) for key in
                       ("layout_seconds", "decode_seconds", "retry_seconds")},
                    "error": error, "reply": reply,
                })

    started = time.perf_counter()
    threads = [threading.Thread(target=drain, args=(lane,)) for lane in lanes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records, time.perf_counter() - started


def compare(record: dict, n: int, baseline: dict[str, str], unstable: set[str], out: Path) -> None:
    """Hash the reply without timings; the first N=1 timed reply of each page is its baseline."""
    reply = record.pop("reply")
    record["sha256"] = record["identical"] = None
    if reply is None:
        return
    reply.pop("timings", None)
    canonical = json.dumps(reply, sort_keys=True, ensure_ascii=False, indent=1)
    record["sha256"] = digest = hashlib.sha256(canonical.encode()).hexdigest()
    key = f"{record['doc']}.p{record['page']}"
    if key not in baseline and n == 1:
        baseline[key] = digest
        write(out / "baseline" / f"{key}.json", canonical)
    if key not in baseline:
        return  # failed in every N=1 pass, which already counts as a failed request
    record["identical"] = digest == baseline[key]
    if record["identical"]:
        return
    if n == 1:
        unstable.add(key)
    write(out / "diffs" / f"n{n}-pass{record['pass']}-lane{record['lane']}-{key}.json", canonical)


def summarize(n: int, records: list[dict], walls: list[float], sampler: Sampler,
              phase_started: float, timed_started: float, one_lane: dict | None,
              unstable: set[str]) -> dict:
    """N=1 runs first, so a page that differs here is unstable at N=1 or a lane difference."""
    ok = [r for r in records if r["error"] is None]
    per_pass = [sum(r["pass"] == p for r in ok) / wall * 3600 for p, wall in enumerate(walls, 1)]
    seconds = sorted(r["seconds"] for r in ok)
    peaks, timed = sampler.window(phase_started), sampler.window(timed_started)
    lane_vram = [max(filter(None, (s["vram_mib"].get(pid) for s in peaks)), default=None)
                 for pid in sampler.pids()]
    pages_per_hour = statistics.median(per_pass) if per_pass else None
    return {
        "lanes": n,
        "pages_per_hour": pages_per_hour,
        "pages_per_hour_by_pass": per_pass,
        "scaling": pages_per_hour / one_lane["pages_per_hour"]
        if one_lane and pages_per_hour and one_lane["pages_per_hour"] else 1.0 if n == 1 else None,
        "p50_seconds": percentile(seconds, 5),
        "p90_seconds": percentile(seconds, 9),
        "requests": len(records),
        "failed_requests": len(records) - len(ok),
        "differing_pages": sorted({f"{r['doc']}.p{r['page']}" for r in records
                                   if r["identical"] is False} - unstable),
        "peak_vram_mib_per_lane": max(filter(None, lane_vram), default=None),
        "vram_mib_by_lane": lane_vram,
        "peak_gpu_memory_mib": peak(s["gpu_used_mib"] for s in peaks),
        "mean_gpu_util_pct": mean(s["gpu_util_pct"] for s in timed),
        "mean_cpu_pct": mean(s["cpu_pct"] for s in timed),
        "min_available_ram_gib": min(filter(None, (s["available_gib"] for s in peaks)), default=None),
        "peak_lane_rss_gib": peak(max(s["rss_gib"].values(), default=None) for s in peaks),
    }


def percentile(values: list[float], decile: int) -> float | None:
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=10, method="inclusive")[decile - 1]


def peak(values) -> float | None:
    return max(filter(None, values), default=None)


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def write_results(out: Path, meta: dict, rows: list[dict], records: list[dict],
                  unstable: set[str], stopped_early: str | None) -> None:
    summary = {**meta, "stopped_early": stopped_early, "unstable_at_one_lane": sorted(unstable),
               "rows": rows}
    write(out / "summary.json", json.dumps(summary, indent=1))
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "differing_pages": len(row["differing_pages"])} for row in rows)
    with open(out / "pages.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, PAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    cell = lambda value: "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
    lines = [
        (f"Lane sweep, {'MPS' if meta['mps'] else 'no MPS'}{', dry run' if meta['dry_run'] else ''}, "
         f"git {meta['git']}, GPU {', '.join(row[0] for row in meta['gpu']) or '-'}, "
         f"{meta['repeats']} timed passes per N."),
        "",
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "|" + "---|" * len(SUMMARY_FIELDS),
        *("| " + " | ".join(cell(len(row[f]) if f == "differing_pages" else row[f])
                            for f in SUMMARY_FIELDS) + " |" for row in rows),
        "",
        f"Pages unstable across N=1 passes (not counted as lane differences): {sorted(unstable) or 'none'}.",
    ]
    if stopped_early:
        lines.append(f"Stopped early: {stopped_early}.")
    write(out / "summary.md", "\n".join(lines) + "\n")


class Sampler(threading.Thread):
    """Samples GPU, CPU and RAM once a second and flags low RAM so lanes stop taking work."""

    def __init__(self, lanes: list[Lane], min_free_gib: float) -> None:
        super().__init__(daemon=True)
        self.lanes, self.min_free_gib = lanes, min_free_gib
        self.samples: list[dict] = []
        self.low_memory, self.stopped = threading.Event(), threading.Event()

    def run(self) -> None:
        before = cpu_ticks()
        while not self.stopped.wait(1):
            now = cpu_ticks()
            gpu = gpu_rows("memory.used,memory.total,utilization.gpu", "gpu")
            sample = {
                "time": time.time(),
                "gpu_used_mib": sum(number(row[0]) or 0 for row in gpu) if gpu else None,
                "gpu_total_mib": sum(number(row[1]) or 0 for row in gpu) if gpu else None,
                "gpu_util_pct": max((number(row[2]) or 0 for row in gpu), default=None),
                "vram_mib": {int(row[0]): number(row[1]) for row in
                             gpu_rows("pid,used_memory", "compute-apps") if row[0].isdigit()},
                "cpu_pct": 100 * (now[0] - before[0]) / (now[1] - before[1])
                if now and before and now[1] > before[1] else None,
                "available_gib": meminfo_gib("MemAvailable"),
                "rss_gib": group_rss_gib(self.pids()),
            }
            before = now
            self.samples.append(sample)
            if sample["available_gib"] is not None and sample["available_gib"] < self.min_free_gib:
                self.low_memory.set()

    def pids(self) -> list[int]:
        return [lane.process.pid for lane in self.lanes]

    def window(self, since: float) -> list[dict]:
        return [s for s in self.samples if s["time"] >= since]

    def shortage_for_one_more_lane(self) -> str | None:
        """The resource one more lane would exhaust, judged by the largest lane so far."""
        if not self.samples:
            return None
        latest = self.samples[-1]
        lane_ram = peak(max(s["rss_gib"].values(), default=None) for s in self.samples) or 0
        # MPS can hide per-process memory; then one lane is the GPU total over the lanes.
        lane_vram = peak(s["vram_mib"].get(pid) for s in self.samples for pid in self.pids()) or (
            peak(s["gpu_used_mib"] for s in self.samples) or 0) / len(self.lanes)
        if latest["available_gib"] is not None and latest["available_gib"] < lane_ram + self.min_free_gib:
            return "RAM"
        if latest["gpu_total_mib"] and latest["gpu_total_mib"] - latest["gpu_used_mib"] < lane_vram + 1024:
            return "VRAM"
        return None


def gpu_rows(fields: str, kind: str) -> list[list[str]]:
    """nvidia-smi --query-<kind> rows for the visible GPUs; [] when there is no GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    command = ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits",
               *(["-i", visible] if visible else [])]
    try:
        text = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[cell.strip() for cell in line.split(",")] for line in text.splitlines() if line.strip()]


def number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]"


def cpu_ticks() -> tuple[int, int] | None:
    try:
        ticks = [int(v) for v in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    except OSError:
        return None
    return sum(ticks) - ticks[3] - ticks[4], sum(ticks)  # busy, total (idle and iowait excluded)


def meminfo_gib(field: str) -> float | None:
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 2**20
    return None


def group_rss_gib(groups: list[int]) -> dict[int, float]:
    """Resident memory per process group: a lane with its torch.compile workers."""
    if not groups or not Path("/proc").is_dir():
        return {}
    rss = dict.fromkeys(groups, 0.0)
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            text = stat.read_text()
            fields = text[text.rindex(")") + 2 :].split()  # state ppid pgrp ... rss is 22nd
            if int(fields[2]) in rss:
                rss[int(fields[2])] += int(fields[21]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    return rss


def start_mps(out: Path) -> tuple[tuple[subprocess.Popen, dict], dict]:
    # A private pipe directory keeps this daemon apart from any system one; short for the socket path.
    env = {"CUDA_MPS_PIPE_DIRECTORY": tempfile.mkdtemp(prefix="mps-"),
           "CUDA_MPS_LOG_DIRECTORY": str(out / "mps")}
    (out / "mps").mkdir()
    with open(out / "mps" / "daemon.log", "ab") as log:
        process = subprocess.Popen(["nvidia-cuda-mps-control", "-f"], env={**os.environ, **env},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if Path(env["CUDA_MPS_PIPE_DIRECTORY"], "control").exists():
            print(f"MPS daemon ready, PID {process.pid}", flush=True)
            return (process, env), env
        time.sleep(1)
    stop(process)
    raise RuntimeError("MPS control daemon did not start; see mps/daemon.log")


def stop_mps(process: subprocess.Popen, env: dict) -> None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True,
                       env={**os.environ, **env}, timeout=30, check=False)
    stop(process)


def stop(process: subprocess.Popen) -> None:
    """Signal the session started for this process (its group ID is its PID), never a name.

    The group is signalled before the leader is reaped, so its ID cannot have been reused.
    """
    for sig, seconds in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(seconds)


def http(port: int, path: str, body: bytes | None, timeout: float) -> bytes:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return reply.read()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def git_revision() -> str:
    run = lambda *cmd: subprocess.run(["git", "-C", str(REPO), *cmd], capture_output=True, text=True, check=False).stdout.strip()
    return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")


def serve_stub(port: int, drift: bool) -> None:
    """Dry-run lane: a reply determined by the page bytes, after a random delay."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._reply({"status": "ready", "pid": os.getpid(), "model": {"id": "stub"}})

        def do_POST(self) -> None:
            page = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seconds = random.uniform(0.02, 0.08)
            time.sleep(seconds)
            text = hashlib.sha256(page["image"].encode()).hexdigest()[:16] + ("!" if drift else "")
            self._reply({"elements": [{"category": "text", "bbox": [0, 0, 1, 1], "text": text}],
                         "model": {"id": "stub"}, "timings": {"decode_seconds": seconds}})

        def _reply(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            return

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
