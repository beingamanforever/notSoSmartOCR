# Not So Smart OCR

Evidence-linked OCR for structured and clinical documents. Research software, not validated for unattended clinical use.

![OCR evidence pipeline](artifacts/architecture/ocr-evidence-pipeline-preview.png)

## Pipeline

`input -> orientation -> positioned OCR -> structure -> bounded specialists -> canonical evidence -> review -> UI and exports`

- Nemotron reads positioned literal text; geometry establishes tables, forms, controls, and reading order.
- TrOCR handwriting and Falcon formula readers receive only eligible source-resolution crops.
- Specialist output stays an alternative until independent evidence or human review accepts it.
- Accepted revisions replace consumed fragments once. Source boxes, raw responses, alternatives, and review state remain attached.
- Text, Markdown, Copy, and Download render the same canonical revision.

## Demo

| | |
| --- | --- |
| ![Landing](artifacts/screenshots/final/01-landing.jpg) | ![Clinical table text](artifacts/screenshots/final/02-clinical-table-text.jpg) |
| ![Clinical evidence](artifacts/screenshots/final/03-clinical-table-evidence.jpg) | ![Academic text](artifacts/screenshots/final/04-academic-text.jpg) |
| ![Academic evidence](artifacts/screenshots/final/05-academic-source-link.jpg) | ![Financial table](artifacts/screenshots/final/06-financial-table.jpg) |
| ![Financial evidence](artifacts/screenshots/final/07-financial-glossary.jpg) | ![Formula review](artifacts/screenshots/final/08-formula-text.jpg) |
| ![Formula evidence](artifacts/screenshots/final/09-formula-evidence.jpg) | ![Scanned form](artifacts/screenshots/final/10-scanned-form-text.jpg) |
| ![Form evidence](artifacts/screenshots/final/11-scanned-form-source-link.jpg) | ![Code text](artifacts/screenshots/final/12-code-text.jpg) |
| ![Code Markdown](artifacts/screenshots/final/13-code-markdown-export.jpg) | |

## Measured limits

| Evaluation | Result |
| --- | ---: |
| Financial diagnostic cells | 187/187 exact |
| Public table structure | 60 tables, 0.977380 cell exact |
| IAM pilot, TrOCR | 3/15 exact, 6.19% CER |
| Notebook sentence, page pipeline | 20.55% CER |
| Formula source crops, Falcon | 1/5 exact |

Execution, confidence, and valid LaTeX are not counted as correct recognition. Page-level handwriting localization and handwritten mathematics remain unvalidated; formula alternatives require review.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Pillow fastapi uvicorn python-multipart
PYTHONPATH=src python -m ocr_pipeline.demo
```
