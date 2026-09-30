# Serving under load: queue, streaming, caching, and the dashboard

Measured on 30 September 2026 on one NVIDIA A10G (23 GB), eight CPU cores, and 30 GB of RAM without swap.
The load figures are one gentle burst against the live demo, not a production percentile benchmark.

## Why this work was needed

The host froze for about 35 minutes on 30 September.
A second demo and Falcon service ran beside the live stack while a 14-page, 8 MB PDF was being read.
With no swap, the machine thrashed instead of killing a process; the kernel journal shows no OOM kill, only SSH handshake failures and a watchdog timeout.
The instance was restarted, which also erased its NVMe instance storage, including the usage history the dashboard read.

Three separate weaknesses were involved:

| Weakness | Consequence |
| --- | --- |
| Unbounded waiting parses | Every waiting upload held a worker thread, a spooled file, and rendered pages; enough of them starved the thread pool, so even the page stopped loading. |
| No memory or liveness guard | A process near the memory limit froze the whole host rather than failing alone. |
| State on instance storage | Usage records, feedback, and the spending ledger did not survive a restart. |

## A fair queue in front of the GPU

Falcon reads one document at a time; batch composition changes its tokens, so parses are deliberately serial.
Before this change, waiting parses queued on a `threading.Lock`, which wakes waiters in no set order.
Under load, a document could wait behind ones that arrived after it.

`FairQueue` in `src/ocr_pipeline/demo.py` hands out tickets in arrival order and serves them in that order.
The browser learns its place through the progress endpoint and shows "Waiting in line · 2 documents ahead of yours" until reading starts.
The dashboard header shows the same queue live, for example "Reading 1 · 2 waiting".

Two limits keep a flood from exhausting the host:

- `AdmissionMiddleware` counts running and waiting parses on the event loop and returns `503` with `Retry-After: 30` past `--max-active-parses` (32 by default). It runs before the upload is read, so a turned-away request costs nothing.
- The worker thread pool is raised to 128, larger than the queue, so page loads, progress polls, and the dashboard always get a thread.

An early version returned `503` past six parses.
In a burst of ten concurrent uploads, three were turned away, which made the interface feel restrictive; the limit is now a last guard, not the normal way to wait.

| Burst of 10 distinct uploads | Accepted | Turned away | Slowest page load during the burst | Lowest available memory |
| --- | --- | --- | --- | --- |
| Limit of 6, unordered lock | 7 | 3 | 0.006 s | 19,739 MiB |
| Fair queue, limit of 32 | 10, served in order | 0 | 0.086 s | 21,254 MiB |

The ten requests finished between 24 and 83 seconds after submission, because each waited for the ones ahead of it.

## Pages appear as they are read

A long PDF used to show nothing until every page was read.
The browser now sends a random key with the upload and polls `/api/progress/{key}` every 800 ms.
Each finished page is published as soon as `process_document` stages it, and its image is served from `/api/progress/{key}/pages/{n}`.
The finished result replaces the preview, and the page the reader moved to is kept.

The key is generated with `crypto.getRandomValues`.
`crypto.randomUUID` exists only on HTTPS and localhost, and the demo is also served over plain HTTP on the private network; the first release failed there.
Runtime exceptions such as that one are now logged to the console, and people see a plain sentence instead of browser jargon.

## Caching

`BoundedCache` in `src/ocr_pipeline/cache.py` is a least-recently-used cache bounded by the bytes it holds.
Values are pickled and zlib-compressed, entries expire after `--cache-ttl-hours` (24 by default), and `get_or_compute` runs one computation per key, so concurrent requests for the same input wait for its result instead of paying for it twice.
It lives in memory only; nothing reaches disk, and a restart, which is also how new code is deployed, clears it.

| Cache | Key | Default in the demo |
| --- | --- | --- |
| Page cache (`--page-cache-mb`) | SHA-256 of the page image and its sidecar, with the page number | 1024 MB |
| LLM reply cache (`--qwen-reply-cache-mb`) | SHA-256 of the exact request body; only complete replies are kept | 64 MB |

Earlier measurements on the same host: repeated parses took 0.1 to 0.8 s instead of 9 to 16 s, with identical results.
Each usage record carries its own cache hits and misses, exact because parses and refinements run one at a time.

## Keeping every usable LLM answer

Qwen 3.7 Flash rereads only the crops routed to it, in one request per page.
Three kinds of imperfect replies used to cost the whole page:

| Reply | Before | Now |
| --- | --- | --- |
| Cut off at the token limit | Rejected | Crops that closed before the cut are kept |
| One malformed record (Qwen once sent `null` for a control's `group_label`) | Whole reply rejected by the schema | Every record that passes the crop schema on its own is kept; the malformed crop keeps its OCR text |
| Provider error inside a streamed reply (HTTP status still 200) | Parsed as an empty reply | Treated as a provider error and retried like a 5xx |

Before these fixes, two of the six featured examples finished with refinement failed; afterwards all six completed.
A crop the LLM could not improve keeps its OCR text and confidence badge, so the page reports "Complete" rather than a warning.

## Sessions and deferred refinement

The server keeps the 96 most recently used sessions.
During a burst, a session whose LLM refinement was still running could be evicted; its folder was deleted, the background job then failed silently inside its executor, and the browser waited on a 404.
Sessions awaiting refinement are now never evicted, and the job logs any failure and always releases the browser's long poll.

## Usage log and admin dashboard

With `--usage-root`, every parse stores its metrics, routing details, a 1400 px page image, and a 360 px thumbnail, written atomically and pruned past `--usage-retention` records (2000 by default).
Below 2 GB of free disk, a record keeps its metrics but skips the images.
The directory is private to the service user.

`/admin` exists only when the environment variable named by `--admin-token-env` (default `OCR_ADMIN_TOKEN`) is set.
It accepts the token as a cookie from its sign-in form or as a bearer header, compared in constant time.

![Admin dashboard with requests, pages, users, latency, spend, cache use, live queue, and routing reasons](../../artifacts/demo/admin-dashboard.png)

The overview shows requests, pages, users, median time, LLM spend, and the share served from cache, each with a trend line.
The usage chart separates requests read by OCR alone from those that also had regions checked by the LLM.
"Where the time goes" splits the average request into OCR, LLM, and waiting and upload, so the parts add up to the whole.
Activity, logs, and feedback views follow; each request opens a panel with its page, the regions routed to the LLM and why, confidence scores, and the OCR reading each accepted answer replaced.

## Operations

The deployed host runs one supervised stack: the Falcon service, the web app, and a memory guard.

- A supervisor restarts a service when it exits, or when its health URL stops answering for a minute after it had answered once.
- A memory guard kills the largest supervised service when available memory falls below 3 GB, before the host starts thrashing; its supervisor starts it again. Supervised services also get `oom_score_adj` 1000, so the kernel takes them before `sshd` if it ever acts.
- Usage records, feedback, and the spending ledger live on the root disk; only models, caches, and scratch files use instance storage.
- Deploying new web-app code restarts only the web app, which answers again within about a second while Falcon stays warm, then replays the featured examples to refill the caches.
- Benchmarks do not run beside the live stack.

## Remaining limits

The queue is in one process; a second web-app process would need a shared queue.
A client that disconnects while waiting is still read when its turn comes.
Queue position is shown only after the upload is received and its pages are rendered.
