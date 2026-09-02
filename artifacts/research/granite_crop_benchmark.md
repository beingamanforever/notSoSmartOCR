# Granite Docling 258M on the frozen C14 handwriting crops

**Evaluation date:** 2026-09-02
**Decision:** Do not integrate Granite Docling 258M as a crop recognizer or route
additional private full pages through it.

Granite recovered only 2 of 44 crops exactly, reached 320.00% CER, emitted 891
inserted characters against 340 reference characters, and returned an empty
prediction for 16 crops. Nine generations exhausted the 128-token crop budget;
manual review confirmed that all nine had already entered erroneous repetition.
It fails both accuracy paths and the hallucination condition in the predefined
promotion rule.

The requested five-page triage was not run. The crop result was clearly unusable,
which activated the explicit stop rule and avoided sending five more private pages
through a failed challenger.

## Model eligibility and official interface

The candidate passes the project's model-origin and license screen:

- The [official model card](https://huggingface.co/ibm-granite/granite-docling-258M)
  identifies IBM Research as the developer, Apache 2.0 as the license, and the
  architecture as a Google SigLIP2 base patch16-512 vision encoder plus a Granite
  165M language model.
- The [official SigLIP2 model card](https://huggingface.co/google/siglip2-base-patch16-512)
  is published by Google and declares Apache 2.0.
- The [Hugging Face model API](https://huggingface.co/api/models/ibm-granite/granite-docling-258M)
  reported 257,517,120 BF16 parameters and current revision
  `982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe`. The cached run pinned that exact
  revision.
- The model card documents both prompts used by the harness:
  `Convert this page to docling.` and
  `OCR the text in a specific location: <loc_x1><loc_y1><loc_x2><loc_y2>`.
  The bbox tokens use the official
  [Docling Core `DocumentToken.get_location`](https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/tokens.py)
  500 by 500 coordinate grid.

All sources above were accessed on 2026-09-02. Jina MCP was not available in this
session, so the check used direct official pages only. No private image, text, URL,
cookie, or credential was sent to a public service.

## Frozen protocol

- Panel: all 44 annotated C14 handwriting spans from `C14-D001-P001` and
  `C14-D002-P001`. Every crop remains in the denominator, including empty outputs
  and generation failures.
- Input: the retained native tight crop with its existing 4-pixel context padding.
  The 3x crop was used only for human inspection.
- Privacy: inference ran only on the authorized A10G host. Ground truth stayed on
  the local machine; the remote inference JSON contains predictions and field IDs
  but no reference strings.
- Runtime: NVIDIA A10G, BF16, PyTorch 2.7.1+cu126, Transformers 5.10.1, SDPA,
  greedy decoding (`do_sample=False`, `num_beams=1`), seed 0, deterministic
  algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Crop generation cap: 128 new tokens. The longest reference is 24 characters.
  Every capped generation contained incorrect text before the cap, so
  extending those generations could not recover strict exact match and would
  retain their existing insertion errors.
- Extraction: preserve decoded content while removing generated control tokens and
  DocTags, HTML-unescape, then collapse whitespace. No spelling, punctuation, or
  semantic correction is applied.
- Strict exact: raw reference equals extracted prediction. Normalized exact and
  edit metrics use Unicode NFKC, case folding, and collapsed whitespace only.
- CER and WER: micro Levenshtein edits divided by 340 reference characters and 60
  reference words, respectively. Missed and hallucinated rates are deletions and
  insertions divided by reference characters.
- Clinical accounting: 39 spans are treated as critical. The five noncritical
  spans are the explicit empty-marker controls `C14-D001-P001-H009` through
  `C14-D001-P001-H013`.
- Latency: warm per-crop wall latency covers processor work, host-to-device transfer,
  generation, synchronization, decode, and extraction. Image-file open time and the
  one unreported warm-up call are excluded.

## Result

| Metric | Granite crop result |
| --- | ---: |
| Frozen fields | 44 |
| Runtime exceptions | 0 |
| Strict exact | 2 / 44 (4.55%) |
| Normalized exact | 2 / 44 (4.55%) |
| CER | 1,088 / 340 (320.00%) |
| WER | 191 / 60 (318.33%) |
| Character substitutions | 83 |
| Missed characters | 114 / 340 (33.53%) |
| Hallucinated characters | 891 / 340 (262.06%) |
| Empty predictions | 16 / 44 (36.36%) |
| 128-token limit reached | 9 / 44 (20.45%) |
| Critical substitutions | 26 / 39 |
| Critical misses | 11 / 39 |
| Total incorrect critical fields | 37 / 39 |
| Warm latency p50 | 3.370 s/crop |
| Warm latency p95 | 13.854 s/crop |
| Warm latency max | 14.304 s/crop |
| Highest observed process VRAM | 1,656 MiB |

Zero runtime exceptions must not be read as success. The dominant failures were
semantic: empty output, unrelated document boilerplate, label leakage, corrupted
identifiers, and repetitive decoding.

The 1,656 MiB VRAM figure is the highest process-level `nvidia-smi` observation,
not PyTorch peak allocated memory. The process was stopped as soon as the crop arm
was confirmed complete, before its end-of-run allocator summary. NVIDIA accounting
was disabled, so an exact historical allocator peak is unavailable without an
otherwise unnecessary rerun. This limitation is retained rather than relabeling a
snapshot as a measured peak.

## Matched 44-crop comparison

These rows use the same 44 native crops and retained references. WER and critical
counts for the three earlier candidates were recomputed from their retained rows
with the Granite scorer so that the definitions match.

| Candidate | Strict exact | Normalized exact | CER | WER | Missed rate | Hallucinated rate | Critical substitutions / misses | Warm p50 / p95 | VRAM evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence-2 base-ft | 3/44 (6.82%) | 4/44 (9.09%) | 58.82% | 95.00% | 7.06% | 23.24% | 35 / 0 | 0.063 / 0.107 s | 619 MiB peak allocated |
| Florence-2 large-ft | 4/44 (9.09%) | 5/44 (11.36%) | **54.41%** | **91.67%** | 9.41% | **23.53%** | **34 / 0** | 0.129 / 0.214 s | 1,790 MiB peak allocated |
| Ministral-3 3B Base | 2/44 (4.55%) | **9/44 (20.45%)** | 109.41% | 130.00% | **5.59%** | 90.00% | 30 / 0 | 0.154 / 0.968 s | 7,389 MiB peak allocated |
| Granite Docling 258M | 2/44 (4.55%) | 2/44 (4.55%) | 320.00% | 318.33% | 33.53% | 262.06% | 26 / 11 | 3.370 / 13.854 s | 1,656 MiB highest observed process use |

Revision pins for the matched baselines are retained in their local score files:
Florence base `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`, Florence large
`4a12a2b54b7016a48a22037fbd62da90cd566f2a`, and Ministral
`6f9c4b12a95b139af68670a6713616b757923735`.

### Promotion rule

The best matched strict-exact and CER baseline is Florence-2 large-ft.

| Required condition | Passing value | Granite | Outcome |
| --- | ---: | ---: | --- |
| At least 15 percentage-point strict-exact gain | At least 24.09%, which requires 11/44 | 4.55%, a 4.55-point loss | Fail |
| Or at least 25% relative CER reduction | At most 40.81% CER | 320.00%, a 488.11% relative increase | Fail |
| No hallucination increase | At most 23.53% | 262.06% | Fail |
| No critical-substitution increase | At most 34 | 26 | Numeric pass, but with 11 new critical misses |

Granite therefore fails both possible accuracy routes and one mandatory safety
condition. Its lower substitution count is not a safety improvement: empty outputs
moved 11 critical errors into the miss category, and 37 of 39 critical fields were
still wrong.

## Historical specialist context

TrOCR and Nemotron were evaluated earlier on a related 46-field handwriting panel,
not this exact 44-row frozen view. They provide directional context only and are not
used for promotion arithmetic. Their old result schema does not contain matched
insertion or critical-substitution counts.

| Historical candidate and view | Panel | Strict exact | CER | WER | Latency evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| TrOCR base handwritten | 46 | 1/46 (2.17%) | 82.51% | 136.23% | 31.42 ms p50, 33.98 ms p95 at batch 8 |
| Nemotron OCR v2, grayscale 1x | 46 | 7/46 (15.22%) | 67.00% | 81.25% | Four-view stack: 2.538 s p50, 4.592 s p95 |
| Nemotron OCR v2, grayscale 2x | 46 | 9/46 (19.57%) | 74.88% | 87.50% | Same four-view stack |
| Nemotron OCR v2, majority with 1x fallback | 46 | 7/46 (15.22%) | 70.94% | 84.38% | Same four-view stack |

The historical source is
[`experiments/HANDWRITING_SPECIALIST_EVALUATION.md`](../../experiments/HANDWRITING_SPECIALIST_EVALUATION.md).

## Manual review of 20 crops

The scorer deterministically selected the 10 largest character-error cases, then
used seed 0 to sample 10 nonoverlapping cases from the remainder. Review used the
retained 3x visualizations only for readability; inference used native crops. Raw
clinical strings are deliberately omitted from this tracked report.

| Set | Field | Crop assessment | Prediction failure |
| --- | --- | --- | --- |
| Worst | `C14-D001-P001-H008` | Clear short numeric handwriting | Repeated unrelated PDF boilerplate to token limit |
| Worst | `C14-D002-P001-H007` | Clear relationship handwriting | Repeated unrelated PDF boilerplate to token limit |
| Worst | `C14-D002-P001-H005` | Legible name handwriting | Repeated wrong short sequence to token limit |
| Worst | `C14-D001-P001-H016` | Clear decimal handwriting | Repeated copyright boilerplate to token limit |
| Worst | `C14-D002-P001-H013` | Clear date handwriting | Repeated PDF and copyright boilerplate to token limit |
| Worst | `C14-D001-P001-H007` | Clear fraction-like numeric handwriting | Repeated unrelated PDF boilerplate to token limit |
| Worst | `C14-D002-P001-H009` | Clear word with adjacent printed label | Injected printed context and changed the handwritten word |
| Worst | `C14-D001-P001-H001` | Moderately clear date handwriting | Corrupt multilingual repetition to token limit |
| Worst | `C14-D001-P001-H003` | Clear height handwriting | Unrelated PDF boilerplate |
| Worst | `C14-D001-P001-H005` | Clear paired numeric handwriting | Repeated value with a wrong prefix to token limit |
| Random | `C14-D002-P001-H001` | Legible name handwriting | Punctuation and location-token loop to token limit |
| Random | `C14-D002-P001-H003` | Legible facility text with a strike-over | Phonetic corruption and repetition |
| Random | `C14-D001-P001-H006` | Clear decimal handwriting | Empty output |
| Random | `C14-D001-P001-H014` | Clear short numeric handwriting | Exact |
| Random | `C14-D001-P001-H023` | Clear medication handwriting | Empty output |
| Random | `C14-D001-P001-H022` | Clear medication handwriting | Empty output |
| Random | `C14-D001-P001-H019` | Clear medication with neighboring-row spill | Inserted unrelated text and split the medication name |
| Random | `C14-D002-P001-H002` | Clear date handwriting | Changed, expanded, and repeated digits |
| Random | `C14-D001-P001-H015` | Moderately clear date handwriting | Empty output |
| Random | `C14-D002-P001-H006` | Clear identifier with printed-label spill | Copied label text and changed identifier digits |

Nineteen of the 20 reviewed predictions were wrong. The review includes all nine
token-limit cases. The crops were generally readable; limited neighboring label or
row spill appears in a few crops but does not explain empty output or generated PDF
and copyright boilerplate.

## Stopped arms and retained evidence

The official bbox prompt is supported cleanly, and the harness reconstructs the 44
remote bboxes by exact native-crop template matching before applying Docling's
official coordinate conversion. The process transitioned into that arm while the
44-crop completion check was in flight and checkpointed 16 of 44 bbox rows before
termination. The scored JSON labels them `partial_not_scored`; they are excluded
from every table and decision because the arm is incomplete.

No inference was run for the requested five-page triage set:
`C14-D001-P001`, `C14-D002-P001`, `C08-D003-P001`, `C08-D007-P007`, and
`C08-D017-P003`. This is an intentional stop, not missing evidence.

## Exact A10G isolation and cache reuse

The isolated environment was created with this exact command. No package was
installed into it or into the live OCR environment. Its interpreter reports Python
3.12.10 and `include-system-site-packages = true`:

```bash
/opt/pytorch/bin/python3 -m venv --system-site-packages /home/ubuntu/aman/granite-crops-venv
```

The pinned cached model resolved to:

```text
/home/ubuntu/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M/snapshots/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe
```

The snapshot was already present. No Hugging Face download or cache-mutating
command was run.

The retained run was launched from `/home/ubuntu/aman/notSoSmartOCR` with:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages \
CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/ubuntu/aman/granite-crops-venv/bin/python \
  experiments/benchmark_granite_crops.py \
  --inference-only \
  --model ibm-granite/granite-docling-258M \
  --model-revision 982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe \
  --output internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/ministral-c14-crops-source-only/granite_docling_258m_inference.json
```

The explicit `PYTHONPATH` is required on this host because the compatible PyTorch
and Transformers packages live under `/opt/pytorch`. For another candidate, use a
separate venv directory rather than installing into this Granite environment or the
live server environment.

Local scoring kept ground truth off the GPU host:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python \
  experiments/benchmark_granite_crops.py \
  --score-input internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/ministral-c14-crops-source-only/granite_docling_258m_inference.json \
  --modes crop \
  --output internal-clinical-ocr-benchmark/challenging-formats-20260902/runs/ministral-c14-crops-source-only/granite_docling_258m_results.json
```

## Artifacts and verification

- Reusable harness: `experiments/benchmark_granite_crops.py`
- Raw remote inference checkpoint:
  `granite_docling_258m_inference.json` in the gitignored C14 run directory
- Failure-inclusive local score:
  `granite_docling_258m_results.json` in the same gitignored directory
- Remote stdout and warnings: `granite_docling_258m.log` in the same directory
- Durable redacted report: `artifacts/research/granite_crop_benchmark.md`

The harness passed Ruff formatting and lint. The local scoring path completed over
all 44 crop rows, reproduced the retained Florence-base, Florence-large, and
Ministral strict/normalized exact and character-edit totals, and left production
providers, environments, and servers unchanged. The existing focused provider
suite also passed: `6 passed` in `tests/test_granite_docling_provider.py` with
`PYTHONPATH=src`.

## Limitations

- This is a two-page, 44-crop handwriting panel. It is decisive for rejecting this
  crop challenger under the stated rule, not a general evaluation of Granite on
  printed documents.
- The bbox arm is incomplete and intentionally unscored.
- The five-page triage was intentionally skipped after the crop stop condition.
- Exact PyTorch peak allocated and reserved VRAM are unavailable because the failed
  challenger was terminated before its final allocator summary. The observed
  process maximum is reported with its narrower meaning.
- TrOCR and Nemotron comparisons use the older 46-field panel and are not paired
  promotion evidence.
