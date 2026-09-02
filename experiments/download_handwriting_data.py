"""Plan or download explicitly selected public handwriting archives."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import ipaddress
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_MAX_BYTES = 64 * 1024 * 1024
READ_TRAIN_A_BYTES = 21_429_672
SIGNED_QUERY_NAMES = {
    "awsaccesskeyid",
    "googleaccessid",
    "signature",
    "token",
}


@dataclass(frozen=True)
class DownloadItem:
    id: str
    source: str
    url: str
    filename: str
    declared_bytes: int
    groups: Mapping[str, str | None]


CATALOG = {
    item.id: item
    for item in (
        DownloadItem(
            id="read-2017-train-a",
            source="READ 2017",
            url=("https://zenodo.org/api/records/835489/files/Train-A.tbz2/content"),
            filename="Train-A.tbz2",
            declared_bytes=READ_TRAIN_A_BYTES,
            groups={
                "writer": None,
                "document": "PAGE XML and image path inside Train-A",
            },
        ),
        DownloadItem(
            id="nist-sd19-by-write",
            source="NIST SD19",
            url="https://s3.amazonaws.com/nist-srd/SD19/by_write.zip",
            filename="by_write.zip",
            declared_bytes=568_113_446,
            groups={
                "writer": "hsf partition and writer directory",
                "document": "source field file stem",
            },
        ),
        DownloadItem(
            id="nist-sd19-hsf-pages",
            source="NIST SD19",
            url="https://s3.amazonaws.com/nist-srd/SD19/hsf_page.zip",
            filename="hsf_page.zip",
            declared_bytes=348_196_072,
            groups={
                "writer": "hsf partition and fNNNN file stem",
                "document": "full HSF page file",
            },
        ),
        DownloadItem(
            id="nist-sd19-by-field",
            source="NIST SD19",
            url="https://s3.amazonaws.com/nist-srd/SD19/by_field.zip",
            filename="by_field.zip",
            declared_bytes=540_793_335,
            groups={
                "writer": None,
                "document": "field type and source partition",
            },
        ),
        DownloadItem(
            id="nist-sd19-by-class",
            source="NIST SD19",
            url="https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
            filename="by_class.zip",
            declared_bytes=1_031_576_378,
            groups={"writer": None, "document": "class and source partition"},
        ),
        DownloadItem(
            id="nist-sd19-by-merge",
            source="NIST SD19",
            url="https://s3.amazonaws.com/nist-srd/SD19/by_merge.zip",
            filename="by_merge.zip",
            declared_bytes=542_733_164,
            groups={
                "writer": None,
                "document": "merged class and source partition",
            },
        ),
        DownloadItem(
            id="nist-sd6",
            source="NIST SD6",
            url="https://s3.amazonaws.com/nist-srd/SD6/sd06.zip",
            filename="sd06.zip",
            declared_bytes=969_640_744,
            groups={
                "writer": None,
                "document": "submission directory rNNNN and page stem rNNNN_XX",
            },
        ),
    )
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list:
        print(json.dumps([asdict(CATALOG[key]) for key in sorted(CATALOG)], indent=2))
        return 0
    if args.output is None:
        raise SystemExit("output is required unless --list is used")

    items = [CATALOG[item_id] for item_id in args.item]
    if args.selection:
        items.extend(load_selection(args.selection))
    if not items:
        raise SystemExit("select at least one --item or provide --selection")
    validate_plan(items, args.max_bytes)

    plan = {
        "download": args.download,
        "output": str(args.output),
        "max_bytes": args.max_bytes,
        "total_bytes": sum(item.declared_bytes for item in items),
        "items": [asdict(item) for item in items],
    }
    print(json.dumps(plan, indent=2))
    if args.download:
        download(items, args.output, args.max_bytes)
    return 0


def load_selection(path: Path) -> list[DownloadItem]:
    items = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {path}") from error
        items.append(_manual_item(value, path, line_number))
    if not items:
        raise ValueError(f"selection is empty: {path}")
    return items


def validate_plan(items: Sequence[DownloadItem], max_bytes: int) -> None:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    total_bytes = sum(item.declared_bytes for item in items)
    if total_bytes > max_bytes:
        raise ValueError(
            f"selected {total_bytes} bytes exceeds --max-bytes {max_bytes}"
        )
    filenames = [item.filename for item in items]
    if len(filenames) != len(set(filenames)):
        raise ValueError("selected filenames must be unique")


def download(
    items: Sequence[DownloadItem],
    output: Path,
    max_bytes: int,
    *,
    open_url: Callable[[Request, float], Any] | None = None,
) -> list[dict[str, object]]:
    validate_plan(items, max_bytes)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "downloads.jsonl"
    if log_path.exists():
        raise FileExistsError(f"download log already exists: {log_path}")
    for item in items:
        destination = output / item.filename
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")

    records = []
    with log_path.open("x", encoding="utf-8") as log_file:
        for item in items:
            record = _download_one(item, output, max_bytes, open_url)
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
            log_file.flush()
            records.append(record)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--item", action="append", default=[], choices=sorted(CATALOG))
    parser.add_argument(
        "--selection",
        type=Path,
        help="JSONL with explicit source, URL, filename, byte count, and groups",
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def _manual_item(value: object, path: Path, line_number: int) -> DownloadItem:
    if not isinstance(value, dict):
        raise ValueError(
            f"selection row must be an object on line {line_number}: {path}"
        )
    required = {"id", "source", "url", "filename", "declared_bytes", "groups"}
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"missing {', '.join(missing)} on line {line_number}: {path}")
    groups = value["groups"]
    if not isinstance(groups, dict) or not {"writer", "document"} <= groups.keys():
        raise ValueError(
            f"groups must contain writer and document on line {line_number}: {path}"
        )
    for key, group_value in groups.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"invalid group key on line {line_number}: {path}")
        if group_value is not None and not isinstance(group_value, str):
            raise ValueError(f"invalid group value on line {line_number}: {path}")
    item = DownloadItem(
        id=_text(value["id"], "id", path, line_number),
        source=_text(value["source"], "source", path, line_number),
        url=_text(value["url"], "url", path, line_number),
        filename=_text(value["filename"], "filename", path, line_number),
        declared_bytes=value["declared_bytes"],
        groups=dict(groups),
    )
    _validate_item(item, allow_public_nist=False)
    return item


def _validate_item(item: DownloadItem, *, allow_public_nist: bool) -> None:
    if not isinstance(item.declared_bytes, int) or isinstance(
        item.declared_bytes, bool
    ):
        raise ValueError(f"declared_bytes must be an integer for {item.id}")
    if item.declared_bytes < 1:
        raise ValueError(f"declared_bytes must be positive for {item.id}")
    if Path(item.filename).name != item.filename or item.filename in {".", ".."}:
        raise ValueError(f"filename must be a basename for {item.id}")
    _validate_url(item.url, allow_public_nist=allow_public_nist)


def _validate_url(url: str, *, allow_public_nist: bool) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise ValueError("download URLs must be credential-free HTTPS URLs")
    if host == "localhost" or _private_ip(host):
        raise ValueError("download URLs must not target a local or private address")
    is_s3 = (
        host == "s3.amazonaws.com"
        or ".s3." in host
        or host.endswith(".s3.amazonaws.com")
    )
    public_nist = (
        host == "s3.amazonaws.com"
        and parsed.path.startswith("/nist-srd/")
        and not parsed.query
    )
    if is_s3 and not (allow_public_nist and public_nist):
        raise ValueError("manual and redirected private S3 URLs are not allowed")
    query_names = {name.lower() for name, _ in parse_qsl(parsed.query)}
    if any(name.startswith("x-amz-") for name in query_names) or (
        query_names & SIGNED_QUERY_NAMES
    ):
        raise ValueError("signed or tokenized download URLs are not allowed")


def _private_ip(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global


def _download_one(
    item: DownloadItem,
    output: Path,
    max_bytes: int,
    open_url: Callable[[Request, float], Any] | None,
) -> dict[str, object]:
    allow_public_nist = item.id in CATALOG and CATALOG[item.id] == item
    _validate_item(item, allow_public_nist=allow_public_nist)
    request = Request(item.url, headers={"User-Agent": "notSoSmartOCR-public-data/1"})
    response = (
        open_url(request, 60.0)
        if open_url
        else _opener(allow_public_nist).open(request, timeout=60.0)
    )
    destination = output / item.filename
    temporary_path: Path | None = None
    try:
        with response:
            final_url = response.geturl()
            _validate_url(final_url, allow_public_nist=allow_public_nist)
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) != item.declared_bytes:
                raise ValueError(
                    f"Content-Length differs from declared_bytes for {item.id}"
                )
            with tempfile.NamedTemporaryFile(dir=output, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                actual_bytes = _copy_bounded(
                    response, temporary, min(max_bytes, item.declared_bytes)
                )
        if actual_bytes != item.declared_bytes:
            raise ValueError(
                f"received {actual_bytes} bytes for declared {item.declared_bytes}"
            )
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return {
        **asdict(item),
        "actual_bytes": actual_bytes,
        "path": str(destination.resolve()),
    }


def _copy_bounded(source: Any, target: Any, max_bytes: int) -> int:
    total = 0
    for chunk in _chunks(source):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(
                f"download exceeded the declared or allowed {max_bytes} bytes"
            )
        target.write(chunk)
    return total


def _chunks(source: Any) -> Iterator[bytes]:
    while chunk := source.read(1024 * 1024):
        yield chunk


class _SafeRedirect(HTTPRedirectHandler):
    def __init__(self, allow_public_nist: bool) -> None:
        self.allow_public_nist = allow_public_nist
        super().__init__()

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request | None:
        _validate_url(newurl, allow_public_nist=self.allow_public_nist)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener(allow_public_nist: bool) -> Any:
    return build_opener(_SafeRedirect(allow_public_nist))


def _text(value: object, name: str, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {name} on line {line_number}: {path}")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
