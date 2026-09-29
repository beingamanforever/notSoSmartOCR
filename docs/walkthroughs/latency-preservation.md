# Faster OCR without reducing source evidence

Measured on 29 September 2026 against commit `939da31`, on one NVIDIA A10G (23 GB) with eight CPU cores.
These are small, warm, paired measurements, not a production percentile benchmark or a new accuracy evaluation.

## What changed

Independent page rereads now overlap, with results applied in original page order.
The primary crop request and the owned-form-row request also overlap when both exist, using the same prompts, crops, model, generation limits, and validation.
The demo uses its existing page-worker limit (four by default).
A pending spending reservation waits for an active request to settle before reporting budget exhaustion; failed requests with unknown charges retain their reservation.
The spending cap remains enforced in SQLite.

CPU layout tiles run concurrently, using four workers by default in the Falcon service.
Each tile still uses the same batch-one model invocation, original pixels, precision, thresholds, and suppression rules.
Tile results merge in their original order, and the detector's page lock bounds concurrency across documents.
`--layout-workers 1` restores serial tiles; GPU layout remains serial if explicitly selected.
No model, OCR resolution, retry allowance, or token limit was reduced.
Falcon continues using its existing GPU batching, compiled execution, and CUDA graphs.

## Measurements

The end-to-end measurement includes the process response and the complete Markdown export, from a client on the server.
It excludes laptop-to-server network variation and browser painting.
The public three-page input is a losslessly encoded TIFF containing the repository's academic paper, handwritten notes, and financial table examples.
Baseline/candidate order alternates across the first two trials.
Partial results remain in the timing denominator.

| Scope | Baseline | Candidate | Evidence |
| --- | ---: | ---: | --- |
| Three-page document, parallel rereads only, two trials | 82.93 s mean | 60.27 s mean | 27.3% lower; baseline partial in both runs, candidate success once and partial once |
| Native paper reader, two paired trials | 9.31 s mean | 8.09 s mean | 13.1% lower; entire response identical apart from timings |
| Native table reader, two paired trials | 16.36 s mean | 14.52 s mean | 11.3% lower; entire response identical apart from timings |
| Three-page document, combined changes, one trial | 84.55 s | 52.46 s | 38.0% lower; both partial, with existing handwriting validation failures retained |
| Single approved contract, combined changes, one trial | 18.98 s | 18.16 s | Both successful; local reader 13.69 to 11.86 s, hosted response 5.20 to 6.21 s |

Single-page hosted calls remain a major bottleneck.
A separate baseline handwriting example spent 28.09 of 33.67 seconds in refinement.
Page concurrency cannot shorten a single hosted generation, and provider variability can outweigh local savings.
The initial single-contract reread-only trials did not demonstrate an improvement: baseline mean 19.10 s, candidate 21.05 s including a retry.

## Preservation checks and rejected changes

Four paired native-service comparisons used byte-identical request bodies.
All four produced identical response content, including text, layout boxes, scores, crop order, and generation evidence, after removing the timing object.
An offline concurrency regression compares exact outgoing hosted request bodies and complete canonical results against serial execution, including a failed middle page and deterministic failure ordering.
Separate checks cover concurrent budget settlement and retained unknown charges.
These checks establish preservation for the tested execution changes; they do not claim all OCR content is correct or live hosted responses are deterministic.

Moving Heron to GPU reduced detector time dramatically in a probe, but changed boxes and added a table-sample detection (103 versus 102 proposals).
It was rejected for this preservation-focused release.
Reducing CPU intra-operation threads from four to one or two was slower and changed score values, so that was also rejected.
Model replacement, quantization, reduced resolution, shorter generation, fewer retries, and fewer crops were not used.

## Feedback and confidence

Both feedback buttons save an immutable event containing the original input file, reviewed page PNG, full response JSON, and Markdown at the judged revision.
Optional negative-feedback details create a subsequent event.
Writes publish atomically; stale revisions and failed saves are rejected.
The existing retention limit keeps the most recent 1,000 events.
Synthetic upload/edit/vote tests verify both verdicts, full multipage input preservation, matching Text and Markdown, and retrieval after session cleanup.

The UI uses “Confidence” and “Routed to LLM”; raw provider provenance remains in exported evidence.
The current Qwen3.7 Flash endpoint advertised `logprobs` and `top_logprobs`, but two successful synthetic requests returned `choices[0].logprobs: null`, including one with `top_logprobs: 1`.
No LLM score is fabricated, and requesting unused log probabilities was not added to every document call.

## Verification

The complete local project suite passed: 706 tests, two skipped.
Focused Python lint and JavaScript syntax checks passed.
Browser checks covered both feedback buttons, simplified confidence labels, the favicon, no console errors, and a narrow layout without horizontal overflow.

## Sources and limits

Sources accessed 29 September 2026:

- [Falcon-Perception implementation](https://github.com/tiiuae/Falcon-Perception): continuous batching, background tokenization, compilation, and CUDA graph support already exist in the native engine.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html): equivalent mathematical operations can differ across devices and batched execution; measured equality must not be assumed.
- [OpenRouter latency guidance](https://openrouter.ai/docs/guides/best-practices/latency-and-performance): distinguish time to first token, generation throughput, and provider routing from application overhead.
- [Current model endpoint metadata](https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints): advertised parameters are not proof that a response contains usable log probabilities.

The remaining handwriting, table topology, and margin-note accuracy issues are separate work.
Private source documents and raw clinical responses are excluded from this report and the repository.
