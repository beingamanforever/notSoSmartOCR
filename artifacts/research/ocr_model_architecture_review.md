# OCR model architecture review

Date: 2026-09-02

## Decision

Use the ideas from LightOnOCR-2, Infinity Parser 2, and dots.mOCR, but do not use
their released weights in the deployable path. Each model contains a
Chinese-origin foundation model or backbone:

- LightOnOCR-2 uses a Qwen3 decoder.
- Infinity Parser 2 uses Qwen3.5.
- dots.mOCR uses Qwen2.5 and a Chinese-origin vision encoder.

The best compliant adaptation candidate found is
`mistralai/Ministral-3-3B-Base-2512`. It combines a 3.4B Mistral decoder, a
0.4B Pixtral vision encoder, the multimodal projector, and a 2x2 spatial merge
in one Apache-2.0 checkpoint. Mistral reports that BF16 inference fits within
16 GB VRAM, making it a practical A10G baseline and QLoRA target.

The deployment sequence is native-first. Preserve Ministral's aligned Pixtral
encoder, projector, Tekken tokenizer, and decoder for the first supervised
pilot. Test the OCR-tuned LightOn vision tower only as a later research
ablation. This separates gains from the training recipe from gains that depend
on weights jointly optimized with a prohibited Qwen decoder.

This is an architecture decision, not a quality claim. Ministral has not yet
earned a production route on the clinical benchmark.

## Why LightOnOCR is called 1B

LightOnOCR does not use Mistral Small 3.1's 24B language decoder. Its official
configuration combines:

- a compact Qwen3 decoder with 28 layers and hidden size 1,024, corresponding
  to the roughly 0.6B Qwen3 class;
- a Pixtral vision encoder with 24 layers and hidden size 1,024, roughly 0.3B
  to 0.4B parameters; and
- a small two-layer projector with 2x2 patch merging.

The combined total is rounded to the 1B class. The important idea is the
allocation of capacity between a strong native-resolution visual encoder and a
small decoder, not compression of a 24B model.

## Transferable mechanisms

### LightOnOCR-2

Adopt or test:

- aspect-preserving page rendering at 200 DPI with a 1,540 pixel longest edge;
- native-resolution Pixtral visual features;
- 2x2 spatial merging to reduce visual tokens fourfold;
- canonical Markdown, HTML-table, LaTeX, figure placeholder, and blank-page
  targets;
- deterministic normalization before deduplication;
- moderate degradation augmentation with an independently measured schedule;
- assistant-token-only loss;
- deterministic rewards for repetition, EOS, valid math, visible headers and
  footers, and schema compliance;
- checkpoint averaging only when checkpoints have complementary validation
  evidence.

Do not adopt its vocabulary pruning as a lossless optimization. The authors
report an 11.6 percent speed gain, but also report quality loss and weaker
non-Latin behavior.

#### Weight and interface boundary

The transferable artifacts are the page-processing, target-formatting,
augmentation, loss, and evaluation recipes. LightOnOCR, Infinity Parser 2, and
dots.mOCR weights remain excluded from the deployable path.

LightOn and Ministral use the same Pixtral vision dimensions: hidden size
1,024, 24 layers, 16 heads, patch size 14, and 2x2 spatial merging. This permits
a controlled research ablation that transfers LightOn's vision encoder,
vision RMSNorm, and decoder-independent patch merger. It does not permit a
whole-checkpoint conversion:

- LightOn's Qwen decoder hidden size is 1,024, while Ministral's is 3,072.
  LightOn's decoder-facing projector layers therefore have incompatible
  shapes. Retain Ministral's native bridge or train a new bridge.
- LightOn uses the Qwen tokenizer and Qwen image and assistant tokens.
  Ministral uses Tekken with `[IMG]`, `[IMG_BREAK]`, and `[IMG_END]`. Re-tokenize
  every target and derive assistant-loss boundaries from the active template.
  Never copy the notebook's hardcoded Qwen token IDs.
- LightOn bounding boxes are ordinary decimal coordinates normalized to
  0-1,000, not learned coordinate tokens. They mainly locate embedded images
  and do not replace text, table-cell, handwriting, or control geometry.

The first pilot should retain Ministral's complete native multimodal alignment.
Only if that pilot succeeds should an equal-data, equal-seed ablation replace
the vision-side components with LightOn's OCR-tuned versions and briefly warm
up the retained Ministral bridge.

### Infinity Parser 2

The most useful result is the weakness-driven data flywheel. In the authors'
ablation, weakness-targeted pseudo-labeled real documents produced a much
larger gain than adding synthetic scale. The project should therefore mine
measured misses, correct them, keep family-disjoint evaluation pages held out,
and retrain on the resulting failure clusters.

Other useful mechanisms:

- one ordered representation containing element type, literal text, geometry,
  and reading order;
- separate text and structure rewards;
- a frozen-vision first adaptation phase;
- selective high-resolution crops rather than a global 4K page route;
- task specialists for tables, controls, formulas, and charts behind one
  output contract;
- blank-page training and explicit no-content behavior.

The authors report 6.06 seconds per page on one H100 at concurrency 1 and 0.95
seconds per page on two H100s at concurrency 8. The latter is throughput
scaling, not single-page latency and is not transferable to the A10G.

### dots.mOCR

Reuse the data-engine ideas:

- stratify by domain, density, block count, tables, formulas, handwriting,
  controls, orientation, and degradation;
- oversample measured hard regimes;
- validate pseudo-labels with schema rules and render-back comparison;
- use DOM-aligned webpage rendering when licenses permit;
- canonicalize tables, formulas, and vector graphics before training;
- retain general multimodal data to limit catastrophic specialization.

Do not use its model-judge Elo as independent accuracy evidence. Its page
transcription and vector reconstruction also require separate passes, so the
reported system is not a one-pass latency baseline.

## Candidate stack

| Role | Primary candidate | Reason | Status |
| --- | --- | --- | --- |
| Printed text and reading order | NVIDIA Nemotron OCR v2 | Current measured local improvement over the Tesseract floor | Retain |
| Structured page challenger | Ministral 3 3B Base after OCR tuning | Compliant, aligned Pixtral stack, A10G-sized | Pilot |
| Zero-shot multimodal comparison | Phi-4 Multimodal | Microsoft, MIT, compact native vision | Fixed panel only |
| Compact layout parser | IBM Granite Docling 258M | Small, non-Chinese structure model | Fixed panel only |
| Table geometry | Microsoft Table Transformer | Strong measured PubTables geometry | Retain |
| Table text | Existing evidence-preserving multi-reader fusion | Exact gains on the reviewed financial slice | Retain and broaden |
| Clear controls | Deterministic geometry | Fast and locally accurate on clear controls | Retain |
| Handwriting | No adopted specialist yet | Existing PyLaia and TrOCR candidates failed strict clinical review | Research |

Gemma 3 4B is a useful gated comparison but uses the Gemma terms rather than a
permissive license. Molmo 7B is excluded because its decoder is Qwen2-7B.
Mistral Small 3.1 24B remains a research baseline, but its roughly 55 GB BF16
requirement does not fit the 24 GB A10G.

## Released data

### LightOnOCR

The main public mix contains about 16.43M target rows. The bounding-box mix
contains 417,507 training rows and 2,000 validation rows. These releases contain
text targets and metadata, not paired source page images. Their dataset license
is marked `other`, upstream document obligations remain, and the targets are
model generated. The paper reports supervision from Qwen3-VL-235B and GPT-4o,
so the released targets also carry teacher-provenance risk even though the
LightOn model code is Apache-2.0.

Use these releases to study formatting and normalization, not as a
provenance-cleared paired corpus. Maintain a separate clean-room track using
permissively licensed source-derived, human-reviewed, or deterministic
synthetic targets if the no-Chinese-model policy also excludes Qwen-generated
labels.

### Infinity Parser 2

`Infinity-Doc2-5M` includes images, boxes, classes, contents, and reading order,
but is roughly 900 GB and is CC BY-NC-SA 4.0. It can support a separated
research-only reproduction, not the durable commercial training path.

### dots.mOCR

No complete public release of the reported training corpus was found as of the
review date. The code repository is MIT, but model origin and weight eligibility
remain disqualifying for the deployable path.

## A10G adaptation plan

Start with a measured pilot rather than extrapolating the 43M-page LightOn
recipe:

1. Run the stock Ministral checkpoint on a public, frozen OCR slice.
2. Build a family-disjoint pilot from permissive public data and manually
   corrected synthetic clinical failures.
3. Train the native Ministral projector first, then add rank-8 or rank-16
   decoder LoRA while keeping the Pixtral encoder frozen.
4. Start at 700 and 1,024 pixel longest-edge settings with 1,024 to 1,536
   output tokens. Test the 1,540-pixel, 6,144-token LightOn setting only after
   the smaller route demonstrates a quality benefit.
5. Use BF16 compute, 4-bit frozen base weights, microbatch 1, gradient
   accumulation, and gradient checkpointing.
6. Measure a 100-step throughput and peak-memory profile before approving the
   10K-page phase.
7. Open only the final vision blocks if a controlled ablation shows that
   visible glyph recognition, rather than decoding, is the residual bottleneck.
8. Test the LightOn vision-tower transplant only after the native Ministral
   LoRA arm passes. Add deterministic RLVR only after supervised training has a
   stable residual.

Unmeasured planning ranges, uncertain by more than 2x until the throughput
profile:

| Stage | Estimated A10G time |
| --- | ---: |
| Frozen 100 to 500-page evaluation | 1 to 3 GPU-hours |
| Projector-only, 2K to 5K pages | 4 to 12 GPU-hours |
| Ministral 3 LoRA, 10K pages, one epoch | 12 to 36 GPU-hours |
| Ministral 3 LoRA, 50K pages | 60 to 180 GPU-hours |
| Ministral 3 LoRA, 100K pages | 120 to 360 GPU-hours |

These are capacity estimates, not benchmark results. Actual cost is measured
GPU hours multiplied by the applicable A10G price. The provided owned GPU has
no rental charge, excluding power and opportunity cost. Full AdamW tuning is
not expected to fit safely in 24 GB. Mistral Small 3.1 24B remains unsuitable
for this path even if its weights are quantized because long visual sequences,
KV cache, activations, and adapter training state consume the remaining memory.

## Adoption rule

Ministral may move from challenger to a production candidate only when a paired,
failure-inclusive evaluation shows all of the following:

- the primary clinical composite has a paired 95 percent confidence interval
  above zero;
- CER, missed-text rate, and unsupported-text rate improve or remain
  non-inferior;
- no handwriting, table, control, faint-text, tiny-text, or rotation slice drops
  by more than 2 absolute points;
- table topology, control-label association, and reading order do not regress;
- every failure and abstention remains in the denominator;
- every disagreement and at least 50 randomly selected apparent successes are
  manually reviewed;
- A10G p95 latency and cost define a better measured Pareto point; and
- the same frozen evaluation beats the chosen 30B and frontier API baselines
  before any public superiority claim.

## Sources

Accessed 2026-09-02.

- [LightOnOCR-2 paper](https://arxiv.org/abs/2601.14251)
- [LightOnOCR-2 official config](https://huggingface.co/lightonai/LightOnOCR-2-1B/blob/main/config.json)
- [LightOnOCR processor config](https://huggingface.co/lightonai/LightOnOCR-2-1B/blob/main/processor_config.json)
- [LightOnOCR tokenizer config](https://huggingface.co/lightonai/LightOnOCR-2-1B/blob/main/tokenizer_config.json)
- [LightOnOCR main dataset](https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126)
- [LightOnOCR bbox dataset](https://huggingface.co/datasets/lightonai/LightOnOCR-bbox-mix-0126)
- [Transformers LightOnOCR implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lighton_ocr/modeling_lighton_ocr.py)
- [Infinity Parser 2 paper](https://arxiv.org/abs/2607.07836)
- [Infinity Parser 2 repository](https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser2)
- [Infinity-Doc2-5M dataset](https://huggingface.co/datasets/infly/Infinity-Doc2-5M)
- [dots.mOCR paper](https://arxiv.org/abs/2603.13032)
- [dots.mOCR repository](https://github.com/rednote-hilab/dots.mocr)
- [Ministral 3 3B Base model card](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)
- [Ministral 3 configuration](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512/blob/main/config.json)
- [Mistral Small 3.1 model card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- [IBM Granite 3.3 2B model card](https://huggingface.co/ibm-granite/granite-3.3-2b-base)
- [Phi-4 Multimodal model card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [Gemma 3 4B model card](https://huggingface.co/google/gemma-3-4b-it)
- [Molmo 7B model card](https://huggingface.co/allenai/Molmo-7B-D-0924)
