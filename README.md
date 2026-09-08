# Not So Smart OCR

Evidence-linked OCR for structured and clinical documents. Research software, not validated for unattended clinical use.

## Pipeline

Every stage reads and writes the same positioned evidence record. Specialists are asked only about eligible crops and return alternatives, never replacements, and no stage deletes a reading it did not create.

- Orientation runs first: a PP-LCNet document-orientation classifier checked against Tesseract OSD evidence.
- Two readers are fused into one record: Falcon-Perception layout OCR supplies the text and semantic kinds, Nemotron OCR v2 supplies a pixel box and a recognition confidence for every word. Claimed-but-absent words return as under-read repair children, gated by fuzzy token presence and a recognition-confidence floor; a read that loops is retried at escalating temperature and, unrecovered, renders as unreadable rather than as content.
- Tables are a three-model argument: TATR, reader layout regions, and heron-101 propose crops; a conflicted or trivially small near-page detection is decomposed into the layout detector's panels. TATR and Docling TableFormer both parse each crop, and the grid the page's own words fill best wins. Token matching is ours (centre-in-cell); cells carry per-cell recognition confidence, and Tesseract challengers re-read disagreeing cells.
- Controls are geometric: checkboxes, hand-drawn rings, and slashed null glyphs, each labelled only by the words their own geometry supports.
- A specialist reading (TrOCR handwriting, Falcon formulas) stays an alternative until independent evidence or a human accepts it.
- An accepted revision supersedes what it consumes, once. Source boxes, raw responses, alternatives, and review state stay attached.
- Text, Markdown, Copy, and Download all render the same canonical revision. The Markdown lane passes through a formatting model that is structurally unable to add a number: any output containing a digit sequence the extraction did not produce is discarded and the raw markdown ships.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Pillow fastapi uvicorn python-multipart
PYTHONPATH=src python -m ocr_pipeline.demo
```
