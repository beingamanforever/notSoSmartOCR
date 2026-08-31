# Public OCR research data

Downloaded datasets live under `data/public/` and are excluded from Git. This directory records source, use boundary, split, and evaluation role. It is not a release or integrity record.

## Acquired first suite

### ClinOCR-Bench v1.0

- Official release: <https://github.com/ClinOCR-Bench/ClinOCR-Bench/releases/tag/v1.0>
- License: MIT, according to the official repository and release.
- Scale: 384 synthetic clinical documents in six 64-document subsets.
- Split used: the official 328-document evaluation split. The 56 template exemplars are not scored in zero-shot evaluation.
- Role: clinical transcription under normal, handwriting-font, degraded, rotated, table, and mixed conditions.
- Primary metric: word error rate, with character error rate and failure coverage reported as additional diagnostics.
- Limitation: synthetic documents and reused templates do not establish real clinical safety or field-level correctness.

Expected local directory: `data/public/clinocr-bench-v1.0/`.

### FUNSD original release

- Official download: <https://guillaumejaume.github.io/FUNSD/dataset.zip>
- Project page: <https://guillaumejaume.github.io/FUNSD/>
- Use boundary: non-commercial research and education, according to the official project page.
- Scale: 199 noisy scanned forms, split into 149 training and 50 testing images.
- Role: word transcription, form entities, and key-value relations.
- Metrics: transcription CER and WER first; entity and relation metrics require a structured extraction adapter and remain separate.
- Limitation: small dataset with known annotation inconsistencies. Results must name the original variant used here.

Expected local directory: `data/public/funsd/`.

## Deferred datasets

- OmniDocBench: acquire only after selecting an exact public release because official page counts and scoring changed across revisions.
- MPDocBench-Parse: second-phase multi-page benchmark with released counts of 420 PDFs and 3,135 pages.
- CheckboxQA: add after its document download script and underlying DocumentCloud terms are reviewed.
- DocLayNet: use for layout only; the core package is about 28 GB.
- PubTables-1M: use a fixed test slice before the roughly 117 GB full corpus.
- IAM: official registration is required. Unofficial mirrors are not used.
- ICDAR 2019 cTDaR: confirm archive rights before acquisition.

No private clinical image is submitted to an external provider without separate PHI approval.
