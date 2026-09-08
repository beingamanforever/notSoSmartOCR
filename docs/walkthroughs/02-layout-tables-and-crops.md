# Layout, tables, formulas, and visual elements

[Part one](01-from-image-to-text.md) followed an upload through orientation, Heron detection, and Falcon recognition. This post explains what the active route does with different document elements, and what remains experimental.

## Layout categories describe document roles

A paragraph, a heading, and a footer can contain similar text but serve different roles. Heron predicts those categories and their source boxes. The canonical evidence retains the category, the reading, and the provenance instead of flattening everything into one string immediately.

Presentation uses those records to construct readable blocks and preserve spatial relationships. It distinguishes detector sequence from presentation ordering. A detector's output enumeration is not automatically the document's reading order. Multi-column forms and unusual layouts remain challenging.

The bounding boxes shown in the UI come from actual detector geometry. They are not invented by spreading text offsets across a rectangle. Keeping that distinction matters when a reviewer clicks a particular value and expects the overlay to point to real source evidence.

## Tables in the active demo

Heron identifies a table region, and Falcon reads its crop with the corresponding native task. The resulting table markup is parsed into structured cells and spans for display and export. This can preserve relationships that a plain text transcription loses, but the generated structure can still contain missing cells, wrong values, or incorrect spans.

The repository also contains Microsoft Table Transformer and Docling TableFormer adapters. **They are not active in the restored demo.** Similarly, the experiment using Docling's official layout postprocessor has been withdrawn. Keeping an adapter in the repository does not mean the running pipeline executes it.

During the withdrawn experiment, a table integration bug confused overlap between padded text envelopes with a collision in the logical grid. That diagnosis remains useful research evidence. It is not a claim that the restored serving route now uses that experimental repair.

For future table comparisons, exact cell text and row/column relationships need separate evaluation. Lower page-level character error can hide misplaced values. A correct number in the wrong column is still an incorrect extraction.

## Mathematical expressions need their full structure

Formulas have two-dimensional relationships: fractions, superscripts, matrices, and nested expressions. Reading disconnected text fragments can destroy those relationships even when several individual symbols are recognizable.

The restored route sends a complete detected formula crop to Falcon. Generated mathematical markup is displayed with local KaTeX. Successful rendering means the markup is parseable; it does not prove that the equation matches the source or is mathematically correct. Mixed prose and formulas also require careful handling so ordinary language is not accidentally displayed as a sequence of mathematical variables.

The supplied [document-parsing survey](https://arxiv.org/html/2410.21169), read on 9 September 2026, separates text recognition, layout, formula recognition, table structure, and visual-element parsing. It discusses structure-aware formula methods and specialized evaluation. My conclusion from that taxonomy is to evaluate these tasks separately while also checking the final document. The survey does not establish a winning model for our data.

## Images, signatures, and checkboxes

Detected pictures retain their source pixels. The visual inspector offers full-resolution PNG crops for selected regions, including text, tables, images, headers, and footers. Crop downloads use the original source rectangle and exclude colored overlay labels. Browser checks compare downloaded pixels directly with that source area.

The UI supports an explicit signature category, but Heron does not provide a native signature class in this setup. A picture prediction must not be described as a verified signature. Signature recognition would require a suitable detector or reviewed annotation.

Detected selected and unselected controls are displayed as checkbox symbols. Their state remains available through accessibility text and hover information. State confidence is separate from text-recognition confidence. A missed tick remains a real detection error and must not be repaired by guessing.

Charts and chemical diagrams are preserved as visual assets. This release does not claim chart-to-data conversion or chemical graph recognition. Those are additional tasks with their own ground truths and evaluation requirements.

Continue with [confidence, exports, and review](03-confidence-exports-and-review.md).

Implementation inspected on 9 September 2026: `heron_layout.py`, `falcon_layout.py`, `rendering.py`, and `demo.html`. The inactive specialist adapters are in `tables.py` and `tableformer_structure.py`.
