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

Local directory: `data/public/clinocr-bench-v1.0/ClinOCR-Bench/`.

### FUNSD original release

- Official download: <https://guillaumejaume.github.io/FUNSD/dataset.zip>
- Project page: <https://guillaumejaume.github.io/FUNSD/>
- Use boundary: non-commercial research and education, according to the official project page.
- Scale: 199 noisy scanned forms, split into 149 training and 50 testing images.
- Role: word transcription, form entities, and key-value relations.
- Metrics: transcription CER and WER first; entity and relation metrics require a structured extraction adapter and remain separate.
- Limitation: small dataset with known annotation inconsistencies. Results must name the original variant used here.

Local directory: `data/public/funsd/dataset/`.

### OmniDocBench v1.6

- Official dataset: <https://huggingface.co/datasets/opendatalab/OmniDocBench>
- Official evaluator: <https://github.com/opendatalab/OmniDocBench>
- Data terms: CC BY-NC 4.0 for the downloaded dataset release. Research use does not imply commercial approval.
- Acquired revision: v1.6, with exactly 1,651 page images and 1,651 annotation records.
- Role: version-pinned end-to-end text, table, formula, and reading-order evaluation through the official `end2end` evaluator.
- Important boundary: the evaluator's current main branch has moved beyond the acquired data revision. Results require a matching v1.6 evaluator checkout and must name both revisions.
- Status: data and evaluator source are present. Model predictions and official scores have not been produced yet.

Local data directory: `data/public/omnidocbench-v1.6/`.

### MPDocBench-Parse

- Official release: <https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/MPDocBench>
- Released annotation scale: 420 documents and 3,135 pages in `MPDocBench.json`.
- Role: multi-page hierarchy, reading order, cross-page continuation, tables, figures, and formulas.
- Acquired: ground truth, evaluator, and official model adapters.
- Not acquired: the page-image archive and restricted SlideVQA-derived pages. Underlying source-document terms must be reviewed before model submission or redistribution.
- Status: metadata and evaluator research are ready, but no valid image-level run can be claimed.

Local directory: `data/public/Qwen-Doc/MPDocBench/`.

### CheckboxQA

- Official release: <https://github.com/Snowflake-Labs/CheckboxQA>
- Terms: evaluation-only CC BY-NC, with separate DocumentCloud terms for underlying documents.
- Acquired and validated: 88 of 88 PDFs, 2,048 retained PDF pages, and 579 gold QA annotations.
- Role: downstream understanding of checkable content with the official ANLS* evaluator.
- Limitation: it is not direct checkbox-state or checkbox-to-label detection ground truth. Derived component metrics require new human annotation and must not be presented as official CheckboxQA scores.

Local directory: `data/public/CheckboxQA/`.

## Deferred or access-controlled datasets

- DocLayNet: use for layout only; the core package is about 28 GB.
- PubTables-1M: use a fixed test slice before the roughly 117 GB full corpus.
- IAM: official registration is required. Unofficial mirrors are not used.
- ICDAR 2019 cTDaR: confirm archive rights before acquisition.

No private clinical image is submitted to an external provider without separate PHI approval.
