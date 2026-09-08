# From pixels to text: the restored Not So Smart OCR pipeline

This series explains what the demo actually runs. Its active route is **page preparation → orientation → Heron layout → Falcon-OCR → canonical evidence → exports and review**. A recent PP-OCRv5 text-instance experiment was withdrawn because it degraded handwritten mathematics and clinical forms. It is not the serving pipeline described here.

## Start with an image, even when the input is a PDF

The upload boundary accepts supported document and image formats and enforces page and decoded-pixel limits. PDFs are rendered into page images with Poppler; the configured default is 300 DPI. Multi-page inputs retain page identity. Image loading handles EXIF orientation and RGB conversion.

The original page image provides the coordinate system for source inspection. Model resizing and crop preparation must preserve a mapping back to that page. A model-input box and a source-page box are not interchangeable.

More pixels are not always an improvement. They cost memory and processing time, and the model may resize them again. Fine print and faint ink can still disappear during resizing. Higher resolution therefore needs measurement on the actual downstream reader.

## Establish page orientation

DocTR supplies a page-orientation proposal, and Tesseract OSD provides additional orientation evidence. The pipeline distinguishes right-angle page rotation from skew and from perspective distortion within a small region. Conflicting orientation evidence is retained for review rather than presented as certainty.

Coordinate restoration is part of correctness. Text can be read correctly after rotation while the UI points to the wrong part of the original image if that transformation is lost. The pipeline retains the relationship between processing coordinates and source coordinates.

## Detect document regions with Heron

Heron predicts semantic layout elements such as text blocks, headings, tables, pictures, headers, footers, and checkbox states. The Heron-101 model uses an RT-DETRv2 detector with a ResNet-101 backbone.

Each detection provides a box, category, and score. The service selects a primary class per detector query. That does not guarantee that separate detector queries never overlap. This restored route can still produce repeated readings when overlapping text-bearing regions cover the same ink. The user chose to retain this behavior rather than accept the recognition regression of the replacement route.

## Read the selected crops with Falcon-OCR

Falcon-OCR, from TII, accepts image crops and native task prompts. It generates text or task-appropriate markup. Tables and formulas require different representations from ordinary prose, while pictures can be retained directly as source pixels.

Falcon-OCR is a vision-language OCR model. Its native architecture uses early fusion; describing it simply as an independent image encoder attached to a separate language decoder would misrepresent that design. Heron is the separate layout detector in this application.

The service batches crops and returns their native generation metadata alongside the reading. The result preserves model identity, source geometry, raw output, and resolution information. Generation success does not establish that every character is correct. A digit may be missing even when the decoder emits a fluent-looking result.

## Why we rolled back the text-instance experiment

The alternative separated text detection from layout and gave each detected instance one reading. It improved specific duplicate examples. However, it split two-dimensional mathematical expressions into fragments and produced poor readings of faint or narrow regions. Clinical form output also deteriorated.

A targeted comparison showed that complete formula and paragraph crops preserved substantially more context. That diagnosis is useful for future work, but the proposed follow-up routing was not promoted. The serving decision was to restore the earlier route immediately.

The lesson is to evaluate complete documents, including missed content, exact digits, formulas, and reading order. Removing repeated output is not an improvement if it destroys the content being extracted.

Continue with [layout, tables, and visual elements](02-layout-tables-and-crops.md) and [confidence, exports, and review](03-confidence-exports-and-review.md).

Implementation inspected on 9 September 2026: `orientation.py`, `falcon_layout.py`, and `experiments/serve_falcon_layout.py`. Official references: [Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) and [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception).
