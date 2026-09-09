# From research papers to our pipeline

*What we reused, what we changed, and what the experiments taught us · 9 September 2026*

[All walkthroughs](../../README.md#walkthroughs) · [Falcon internals](05-falcon-technical-report.md) · [Alternative parsing strategies](06-document-parsing-strategies.md)

Our pipeline combines **direct reuse of upstream models and inference code**, **application-specific integration**, and **lessons from research and unsuccessful experiments**. These are different forms of contribution. A paper can explain a failure or suggest the next experiment without its method being implemented here.

The most substantial adopted pieces are Falcon-OCR's element-level recognition, Heron's semantic layout predictions, and established orientation and rendering libraries. Our changes concentrate on the boundaries between them: preserving detections, interpreting query outputs correctly, retaining generation evidence, avoiding redundant reader passes, and keeping source geometry attached to the final document.

The papers read later also provide useful ways to understand those decisions. Architectural resemblance alone does not establish that an earlier implementation was derived from a particular paper. The sections below distinguish documented reuse from that retrospective analysis.

## 1. What is actually inherited from other projects?

| Source | What we took or adapted | Where it appears | Result and limit |
| --- | --- | --- | --- |
| Falcon-Perception / Falcon-OCR | OCR checkpoint, task prompts, image preparation, native paged generation | [Falcon service](../../experiments/serve_falcon_layout.py) | One recognizer reads text, formula, and table crops; difficult content can still be wrong |
| Docling's Heron-101 | Semantic region detector and its category vocabulary | [Heron adapter](../../src/ocr_pipeline/heron_layout.py) | Source boxes and roles drive recognition and inspection; full Docling conversion is not invoked |
| DocTR and Tesseract | Page-orientation prediction, OSD evidence, and selected margin text | [Orientation](../../src/ocr_pipeline/orientation.py) | Rotation decisions have explicit evidence rather than relying on OCR fluency |
| Microsoft TATR | Table detection/structure integration in the earlier positioned-text route | [Table adapter](../../src/ocr_pipeline/tables.py) | Separates table geometry from recognition; inactive in the restored Falcon route |
| Docling TableFormer | An image-only cell-structure adapter | [TableFormer adapter](../../src/ocr_pipeline/tableformer_structure.py) | Keeps geometry separate from token assignment; inactive in the restored route |
| Poppler, Pillow, KaTeX | Rasterization, image handling, and mathematical display | [Page pipeline](../../src/ocr_pipeline/pipeline.py), [demo](../../src/ocr_pipeline/demo.html) | Established tools handle these operations; successful rendering does not prove correct transcription |
| Our application | Canonical evidence, coordinate bookkeeping, review, exports, feedback | [Contracts](../../src/ocr_pipeline/contracts.py), [rendering](../../src/ocr_pipeline/rendering.py), [feedback](../../src/ocr_pipeline/demo.py) | Makes results inspectable and keeps representations connected |

The last row is our application design. It should not be credited to an imported Docling document object or claimed as a novel model architecture.

## 2. Falcon taught us to preserve complete document elements

### The source idea

The Falcon report separates layout detection from recognition. A detected text, table, or formula element is cropped from the source image and transcribed with a task-specific prompt. The same OCR model generates text, LaTeX, or HTML. Its shared Transformer processes visual patches and text with different attention masks. [Falcon technical report, Sections 2 and 6](https://arxiv.org/html/2603.27365).

### What we adopted

We use Falcon's native recognition and serving infrastructure, while substituting Heron for the layout detector described in the report. The local service maps detected categories to recognition tasks and associates each generated sequence with its original page and region.

Complete crops matter for handwriting and mathematics. A numerator, denominator, or handwritten annotation can lose its meaning when it is split into isolated fragments. The local line-oriented experiment deteriorated on the supplied handwriting and form examples, so the complete-region route was restored. That is a qualitative regression observation, not a reported accuracy gain over every alternative recognizer. [Serving decision](01-from-image-to-text.md#7-why-we-restored-region-crops-after-the-line-experiment).

### What changed around the recognizer

The older composition wrapped a positioned-text reader with tiling, wider-band recovery, and a second page-frame read. These mechanisms addressed gaps in that earlier detector. The Falcon branch now bypasses those general reread wrappers rather than automatically repeating a full layout-and-recognition pass. The branch is explicit in [the composition code](../../experiments/serve_gpu_demo.py).

This removes redundant work by construction. It does not eliminate the separate problem of overlapping Heron regions scheduling distinct crops. A recognizer should receive the intended source element once, but proving which regions own which ink is still necessary.

## 3. Heron and DETR taught us to respect prediction identity

### The source idea

DETR treats detection as prediction of an object set, trained through one-to-one matching with labeled objects. RT-DETR and RT-DETRv2 make this family more efficient through changes to feature processing, query initialization, and attention. Heron applies that detector family to document regions. [DETR](https://github.com/facebookresearch/detr), [RT-DETR](https://arxiv.org/abs/2304.08069), [RT-DETRv2](https://arxiv.org/abs/2407.17140).

The **Advanced Layout Analysis Models for Docling** report also emphasizes annotation quality and category coverage. A dataset that labels forms as tables, or omits controls entirely, teaches inconsistent boundaries. Its expanded document categories are part of why Heron is useful beyond finding plain text rectangles. [Heron report](https://arxiv.org/html/2509.11720).

### What we adopted and corrected

We use the Heron checkpoint directly. Its categories distinguish pictures, formulas, tables, headings, headers, footers, controls, and other regions. The adapter retains the highest-scoring category for each query, along with that query's identity and alternative class scores.

That distinction matters because a query/class score matrix is not a list of independent source objects. Expanding multiple class alternatives for the same query can cause the same geometry to be recognized more than once. Interpreting one query as one primary object hypothesis fixes that interface error without a string-deduplication rule. [Implementation](../../src/ocr_pipeline/heron_layout.py).

The improvement is narrow: fewer opportunities to schedule multiple readings from **one query's alternative labels**. Distinct queries can still overlap, and their boxes may remain inaccurate. We did not retrain Heron or demonstrate a general increase in localization accuracy.

## 4. The table papers taught us that structure and text are different evidence

### PubTables-1M and Table Transformer

The PubTables-1M paper addresses consistent structural annotation, including oversegmentation, and trains detection models for tables and their internal roles. TATR predicts geometric objects such as rows, columns, headers, and spanning cells. A separate OCR or PDF-text source supplies the characters. [PubTables-1M paper](https://arxiv.org/abs/2110.00061), [TATR implementation](https://github.com/microsoft/table-transformer).

The lesson for our integration was to treat **which cell a value belongs to** separately from **what characters were recognized**. A perfect word string can still enter the wrong column. A correct grid cannot repair an incorrectly read digit.

### TableFormer and token matching

The TableFormer paper similarly separates table structure and cell localization from text content, including the use of programmatic PDF text. Our adapter uses its image-only structure path with `do_matching=False`, then leaves text assignment to the application. [TableFormer paper](https://arxiv.org/abs/2203.01017), [local adapter](../../src/ocr_pipeline/tableformer_structure.py).

There is a concrete integration reason for this separation. [Docling model issue #188](https://github.com/docling-project/docling-ibm-models/issues/188) reports orphan-cell recovery assigning unrelated content through a nearest-column fallback. Avoiding that matcher prevents our adapter from silently accepting that particular assignment path. It does not prove the application's alternative cell assignment is always correct.

### What remains in the current route

TATR and TableFormer are inactive in the restored serving composition. Falcon now generates table HTML, which our parser turns into logical rows, columns, headers, and spans. The useful lesson survives: preserve source geometry and recognized text separately, and do not invent cell pixel boxes simply because a logical HTML cell exists.

This is an example of learning from an integration even when its model does not remain in the final serving route. [Current table parsing](../../src/ocr_pipeline/falcon_layout.py), [conditional stage construction](../../experiments/serve_gpu_demo.py).

## 5. The document-parsing survey taught us to name the failing stage

The supplied **Document Parsing Unveiled** survey divides parsing into layout, OCR, mathematical expressions, tables, visual elements, and unified VLM approaches. That taxonomy helps distinguish failures that can look identical in the UI. [Survey](https://arxiv.org/abs/2410.21169).

For a missing percent sign, the useful sequence of questions is:

1. Was the sign visible in the decoded page image?
2. Did a detected region include it?
3. Did resizing or cropping preserve enough detail?
4. Did the recognizer emit it?
5. Did parsing and rendering retain it?

These are different repair locations. A Markdown patch cannot recover a symbol absent from the crop. A larger recognizer cannot fix a character that was emitted correctly but lost by serialization.

The survey helps explain our separation of source regions, generation evidence, structure, and presentation. It is not an imported algorithm, and reading it did not by itself change model accuracy.

## 6. Parser-Oriented Structural Refinement names the unresolved handoff problem

The **Parser-Oriented Structural Refinement** paper is especially relevant to repetition and order. It observes that the parser consumes a *retained and serialized set* of regions, not the detector's entire raw hypothesis pool.

Its method uses query features, semantic information, geometry, and image evidence in a learned refinement module. A shared structural state predicts localization, retention, and order together. The retention head learns which candidates should reach the parser instead of relying only on a fixed overlap-suppression rule. [Paper](https://arxiv.org/html/2604.02692).

This taught us to describe the actual objective more precisely: **construct a faithful, correctly ordered set of recognition inputs**, not merely suppress strings after decoding. Selecting regions and ordering them independently can produce an inconsistent handoff even when many individual boxes look reasonable.

**Status here:** the learned refinement module and its training objectives are not implemented. Our query-aware adapter is also not an implementation of this paper: it has no learned joint retention/order head. The paper provides a concrete research direction for the remaining ownership problem; it should not be listed as a shipped duplicate fix.

## 7. Structured Layout Priors suggests another use for a detector

**Structured Layout Priors for Robust Out-of-Distribution Visual Document Understanding** describes a different interface. It runs a layout detector, serializes detections into the parser's native DocTags vocabulary, and includes that prior alongside the **full page image**. This supplies explicit structure while retaining global visual context. [Paper](https://arxiv.org/abs/2605.19866).

The lesson is that detector output need not only be used to cut crops. It can also condition generation. That makes it a distinct option for cases where independent crops lose relationships between columns or repeated fields.

**Status here:** we do not inject DocTags layout priors into Falcon. Falcon's active prompts select crop recognition tasks. The paper's result cannot be reproduced by pasting arbitrary bounding-box text into a prompt: the representation and parser must be compatible. This remains an alternative to evaluate, not an explanation for an already measured gain in our system.

## 8. The Character Error Vector paper sharpens evaluation

**The Character Error Vector** separates parsing-related error, recognition error, and their interaction. It examines why a single sequence-based score is hard to interpret when page segmentation or ordering is wrong, and presents character-distribution and spatial approaches to diagnosis. [Paper](https://arxiv.org/abs/2604.06160).

For our examples, that means distinguishing duplicated or omitted content from incorrect glyphs. A system that returns half the page with excellent spelling is not necessarily better than one that preserves the page with some character substitutions. Likewise, moving an otherwise correct column can drastically affect a string comparison.

**Status here:** this paper informs the evaluation discussion; its CEV/SpACER implementation is not wired into this repository's current serving route or reported measurements. Character-distribution measures also cannot by themselves prove reading order, and we should not fabricate character-level coordinates to use a spatial metric. The useful conclusion is to report coverage, repetition, order, transcription, and structure separately.

## 9. The handwritten-slide paper reinforces a lesson from our regression

**Typeset Replacement of Handwritten Text and Mathematics on Lecture Slides Using Vision-Language Models** uses a detector to distinguish handwritten text and math, a VLM with corresponding prompts to transcribe them, and a rendering stage to place typeset output back onto slides. Its study concerns annotations from a particular lecturer, not arbitrary clinical handwriting. [Paper](https://ieeexplore.ieee.org/abstract/document/11675789/).

This supports two useful distinctions: a handwritten expression needs sufficient spatial context, and successful typesetting is not the same as correct transcription. The latter matters when a formula compiles beautifully after the model changes a symbol.

**Status here:** we have not adopted that detector, VLM, or slide-replacement workflow. Heron also does not have the paper's dedicated handwritten-text/math taxonomy. Our existing complete-crop design and local rendering are related ideas, but the later reading is not evidence that this paper originally caused those implementation choices.

## 10. What the newer model comparisons taught us

The [strategy comparison](06-document-parsing-strategies.md) gives the architectures in detail. Their contribution to this project is presently research guidance:

| Research or implementation | What it taught us | What is not implemented here |
| --- | --- | --- |
| [LightOnOCR-2](https://arxiv.org/html/2601.14251v2) | Data quality, visual resolution, blank-page examples, and explicit localization supervision can matter as much as parameter count | Its distillation, RLVR, model merging, and bbox checkpoints |
| [FireRed-OCR](https://arxiv.org/html/2603.01840v1) | Teach grounded perception and serialization, then evaluate both structure and content; valid syntax alone is insufficient | Its staged fine-tuning and format-constrained GRPO |
| [Unlimited-OCR](https://arxiv.org/html/2606.23050v1) | Separate long-output memory growth from recognition quality; fixed reference attention and recent-output attention serve different roles | R-SWA or its n-gram decoding restrictions |
| [MinerU2.5](https://arxiv.org/abs/2509.22186) | Global layout and fine glyph recognition need different resolution budgets | Its trained coarse-to-fine VLM and backend orchestration |
| [PaddleOCR-VL](https://arxiv.org/abs/2510.14528) and [VL-1.5](https://arxiv.org/abs/2601.21957) | A compact VLM can sit inside a modular parser; physical distortion and spotting require explicit handling | Their active layout/recognition route in this demo |
| [Chandra 2](https://github.com/datalab-to/chandra) | A page-level model can generate content, roles, coordinates, and structure together, making output validation central | Its checkpoint, generated layout format, or image interpretation |
| [Surya 2](https://github.com/datalab-to/surya) | One shared model can expose page and block modes; the orchestration is still a separate design choice | Its shared VLM or recognition interfaces |

These readings broaden the options for future changes. We should not claim our current pipeline improved because of their training methods when we have not run those methods. Similar crop strategies can arise independently, and architectural comparisons are not controlled ablations.

## 11. Our own contributions came from tracing failures through upstream code

Two changes have public, concrete evidence beyond architectural inspiration:

**Preserving predictions and their scores.** Falcon [PR #32](https://github.com/tiiuae/Falcon-Perception/pull/32) changes region preservation and optional generation metadata. Its reported paired checks kept the detections and OCR weights fixed, preserved existing text, retained picture records, and exposed native token evidence. This improves information preservation and inspection, not the learned coordinates or calibrated accuracy.

**Reducing output-transfer overhead.** Falcon [PR #34](https://github.com/tiiuae/Falcon-Perception/pull/34) stacks GPU tensors before transferring them to CPU. The published 4,096-token retrieval median changed from 141.345 ms to 7.350 ms. That is a measurement of output materialization, not a 19-times faster complete OCR request.

These repairs came from inspecting the implementation and measuring the responsible boundary. They do not need a paper to justify them. The same applies to assigning generated sequences by request identity, preserving source crops, separating token scores from detector scores, and keeping feedback with its image and result. [Detailed contribution walkthrough](02-layout-tables-and-crops.md), [generation implementation](../../experiments/serve_falcon_layout.py), [feedback storage](../../src/ocr_pipeline/demo.py).

## 12. What improved, and what still requires evidence?

| Outcome | Evidence we have | Boundary of the claim |
| --- | --- | --- |
| Pictures and generation metadata survive output assembly | Public PR checks and the local assembly path | Preservation, not better recognition weights |
| One query's class alternatives do not become separate crop requests | Current Heron adapter | Does not resolve overlap between different queries |
| General full-page recovery wrappers are bypassed for Falcon | Current composition branch | Less redundant work; no new overall speed ratio claimed |
| GPU output materialization is faster | Reported paired A10G timings | Retrieval operation only |
| Complete-region recognition was restored after regressions | Supplied failures and documented serving decision | Qualitative recovery, not a universal handwriting benchmark |
| Text, boxes, structure, scores, and feedback can be inspected together | Canonical contracts, renderer, and storage code | Traceability does not establish correctness |
| Learned joint retention/order, injected layout priors, new training recipes | Research descriptions only | Not implemented or measured as local improvements |

The pipeline's development is best understood as a combination of borrowed capabilities, careful interface repairs, and lessons from regressions. The next fundamental improvement should address a remaining failure at its actual source and demonstrate better complete output. Reading a relevant paper helps choose that experiment; it is not the experiment itself.

---

Public sources were accessed on 9 September 2026. Source mechanisms, existing implementation behavior, historical local observations, and proposed research directions are distinguished above. No model changes or new benchmark runs were performed for this documentation update.
