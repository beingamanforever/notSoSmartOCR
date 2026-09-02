# OCR fine-tuning plan and execution status

Date: 2026-09-02

## Execution update

The planned Phi-4 decoder-LoRA canary has now completed on 356 training fields
from 68 families. The family-held development set contains 43 resolved crops
and 4 abstention crops from 6 unseen families. C14 remained fully excluded from
training and development.

The v4 adapter improved family-held resolved exact match from 11/43 to 37/43
and fixed C14 exact match from 32/88 to 50/88 across native and scaled views.
It still made 28 critical substitutions on 78 critical C14 field-view pairs,
and C14 native exact match reached only 24/44 rather than the planned 36/44
adoption threshold. It is therefore retained only as a selective handwriting
crop specialist for review. Localization, page-level recall, and end-to-end
pipeline benefit remain unassessed. Full measured details are in
[Phi-4 handwriting adapter v4](phi4_finetuning_cost.md).

## Decision

Fine-tune `microsoft/Phi-4-multimodal-instruct` first, but only as a
handwriting crop specialist. The frozen 44-crop comparison changed the model
selection: Phi-4 recovered 16 normalized-exact fields, compared with 9 for
Ministral and 5 for Florence-2 Large. Phi-4 still made 23 critical
substitutions, so its frozen checkpoint is not accurate enough for a live
route. Fine-tuning must earn that route on held-out data.

Use a staged experiment:

1. Run exactly two optimizer steps on public or owned synthetic crops to verify
   the infrastructure without private data.
2. Train the checkpoint's existing `vision` decoder LoRA on 120 to 200
   adjudicated, family-disjoint handwriting fields. Keep C14 fully held out.
3. Compare the adapter with the frozen checkpoint on the same dev and C14
   protocols, retaining every failure and critical substitution.
4. Scale to 2,000 fields only if the small canary improves exact recovery with
   no increase in unsupported text.
5. Unfreeze vision components only if a localized error analysis proves that
   the frozen visual representation, rather than decoder copying, is limiting.

Do not full-tune the model on one card. Do not start with RL, DPO, a decoder
swap, 4-bit quantization, or a transplanted vision tower. Each would combine
too many causal changes before the data and target contract have been
validated.

The paragraphs below record the pre-run plan and design rationale. They should
not be read as the measured v4 configuration or as an adoption claim.

## Model architecture

Phi-4 Multimodal has approximately 5.6B parameters and a variable-resolution
vision path. The canary pins the official
`microsoft/Phi-4-multimodal-instruct` repository at revision
`93f923e1a7727d1c4f446756212d9d3e8fcc5d81`; model overrides are rejected.
The released checkpoint already embeds a `vision` LoRA in its language
decoder. The official configuration uses rank 256, alpha 512, and the
`qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj` modules.

The first canary updates only those existing decoder-LoRA tensors. The base
decoder, token embeddings, language-model head, vision encoder, vision
projector, and audio path remain frozen. This preserves the checkpoint's
existing image-language alignment while testing the measured failure: literal
handwriting decoding and copying. The processor uses `dynamic_hd=1` for the
small localized crops, BF16 compute, SDPA attention, a 1,024-token cap, and no
quantization.

### GOT-OCR2.0 research control

GOT-OCR2.0 is a useful compact OCR-specific control, but not a deployable-path
candidate under the no Chinese-origin model rule. The official paper and code
use a Qwen decoder. More precisely, GOT combines an approximately 80M ViTDet-B
encoder, Vary-style final encoder layers, a linear connector, and a Qwen 0.5B
decoder for about 580M parameters. A 1,024 by 1,024 page becomes 256 visual
tokens, and the decoder supports roughly 8K output context.

The reusable ideas are high visual compression, separate encoder pretraining,
mixed page and crop supervision, a curriculum that retains 80 percent of
earlier-stage data, region-prompt OCR, and dynamic multi-crop inference. The
paper reports that multi-crop improves small text, formula, and table results,
but those source benchmarks are not evidence for clinical forms.

The frozen GOT run was rejected: it reached CER 2.220879, WER 3.95935, and 15
of 79 exact handwritten spans on the five-page triage, with 3 of 5 outputs
unusable or token limited. Keep its compression and curriculum ideas as
research inputs, but do not route output to it.

## Canonical training target

Train against the same information model used at runtime. Every page target
must contain ordered regions and preserve literal evidence:

```json
{
  "schema_version": 2,
  "page": {"width": 1000, "height": 1000, "coordinate_space": "normalized_1000"},
  "regions": [
    {
      "kind": "text",
      "bbox": [74, 92, 412, 130],
      "text": "literal source text",
      "reading_order": 1,
      "resolution": "resolved"
    }
  ],
  "tables": [],
  "controls": [],
  "abstentions": []
}
```

The complete target vocabulary includes text, title, header, footer, figure,
table, table cell, formula, handwriting, checkbox, radio button, signature,
and blank page. Tables retain row, column, spans, cell text, and geometry.
Controls retain type, state, label, label association, and geometry. Illegible
content is represented as an abstention, not invented text.

Keep literal OCR evidence separate from normalized Markdown or HTML. Rendering
rules are deterministic and are never learned as substitutes for source text.

This structured target is for a future full-page parser stage. The current
Phi-4 canary does not train page JSON, boxes, reading order, controls, or table
structure. It trains only the adjudicated literal string for each localized
handwriting crop, followed by the checkpoint's assistant terminators.

## Data plan

Use document-family splits so that pages from one template, patient packet,
writer, synthetic generator seed, or source document cannot cross train and
evaluation boundaries.

| Stage | Private real | Owned synthetic | Public | Total |
| --- | ---: | ---: | ---: | ---: |
| Canary | 150 | 250 | 100 | 500 |
| Architecture choice | 700 | 900 | 400 | 2,000 |
| First useful model | 3,000 | 5,000 | 2,000 | 10,000 |
| Scale study | 10,000 | 30,000 | 10,000 | 50,000 |

Target composition at every scale:

| Slice | Share |
| --- | ---: |
| Clean printed pages | 25% |
| Faint, tiny, fax, and degraded text | 20% |
| Tables | 15% |
| Forms and controls | 15% |
| Handwriting | 15% |
| Rotation and reading order | 5% |
| Blank pages and abstention cases | 5% |

The mix is a starting policy. Update it from measured failure clusters, not
aggregate score alone. The first hard-set denominator is C14 with 2 pages and
44 annotated handwritten spans, split 30 and 14. Keep this separate from the
older 44-page private hard set and the older 46-field handwriting panel.

Useful public sources include DocLayNet for layout, PubTables-1M for table
geometry, NIST SD19 for handwriting, and carefully reviewed form data such as
FUNSD. Dataset and upstream document terms still require per-source review
before redistribution or a commercial training route. LightOnOCR target mixes
are useful for studying format and curriculum, but they are not a turnkey
paired, provenance-cleared corpus.

## Private annotation protocol

Keep private pages, crops, labels, and review notes local or on the authorized
GPU host. Do not send them to Jina, web tools, or third-party APIs.

1. Create an initial page target from the original image and a 2x review view.
2. Run a second blinded review with randomized page order and no first-pass
   answer visible.
3. Adjudicate every text, box, reading-order, table, and control disagreement.
4. Require a second human review for every critical clinical field and a
   stratified 10 percent sample of the remainder.
5. Record literal uncertainty with alternatives or abstention. Do not silently
   normalize names, dates, measurements, medication names, or diagnosis codes.
6. Keep held-out document families physically separate from train and tuning
   data.

The annotation UI should show source pixels, region type, box, literal text,
reading order, control state, table coordinates, and reviewer decision in one
screen. Automatic proposals may reduce typing, but only adjudicated labels
enter the gold set.

## Supervised training recipe

- Input: the compiler's family-disjoint `train.jsonl` and `dev.jsonl`; the
  harness rejects any C14 identifier before loading a model. Canary mode
  requires at least two distinct review JSON files. Each must cover the exact
  crop set, explicitly report zero uncertain fields, and agree on accepted
  identities. The separate review-only crop pool is never loaded.
- Loss: assistant-token-only cross entropy. User, image, and padding tokens use
  label `-100`; the assistant EOS remains supervised.
- Packing: disabled. Training examples over 1,024 total tokens are rejected,
  not truncated.
- Precision: BF16 with SDPA and no quantization.
- Vision input: `dynamic_hd=1` because the first experiment uses localized
  handwriting crops rather than full pages.
- Batch: microbatch 1, effective batch 8 through gradient accumulation.
- Memory: non-reentrant gradient checkpointing.
- Optimizer: AdamW, beta1 0.9, beta2 0.95, 3 percent warmup, cosine decay.
- Decoder LoRA learning rate: start at `5e-5`.
- Trainable tensors: only the checkpoint's existing rank-256 `vision` LoRA on
  qkv, output, gate-up, and down projections.
- Late-vision ablation learning rate: between `5e-6` and `1e-5`.
- Epochs: 3 for the 120 to 200-field canary and 2 at 2,000 fields.
- Seeds: one canary seed, then two matched seeds for every promotion decision.

Before a long run, inspect the emitted trainable-name and parameter-count
audit. Accept a saved adapter only if a clean base-model reload produces the
same greedy dev predictions as the in-memory adapter.

## Hardware, time, and cost

A single L4, A10G, or A30 has 24 GB. An A40 has 48 GB; an A30 is not a 48 GB
option. Phi-4's BF16 weight files are about 11.15 GB, and the local frozen crop
run peaked near 11.4 GiB allocated. A BF16 decoder-LoRA crop canary with
`dynamic_hd=1`, microbatch 1, and gradient checkpointing is therefore plausible
on 24 GB, but training peak memory has not been measured. If it does not fit,
move the same BF16 run to the A40 before changing precision or data semantics.

The following ranges are planning estimates, not local measurements or cloud
provider quotes. They apply a conservative 3 to 6 times training-over-forward
allowance to the observed roughly 0.49-second frozen crop median, then add model
load, evaluation, save, and reload time. Dollar ranges assume $0.50 to $1.50 per
GPU-hour solely to expose sensitivity to rental price.

| Stage | Estimated 24 GB time | Illustrative rental cost |
| --- | ---: | ---: |
| Public or synthetic two-step infrastructure run | 5 to 15 min | $0.04 to $0.38 |
| 120 to 200 fields, 3 epochs | 20 to 60 min | $0.17 to $1.50 |
| 2,000 fields, 2 epochs | 3 to 8 h | $1.50 to $12 |

For an owned GPU, marginal rental cost is zero. Replace these ranges with
measured wall time, peak allocated and reserved VRAM, and the actual provider
price after the first run.

## RLVR decision

Do not use RLVR until the 10,000-page supervised checkpoint is stable and at
least 500 family-disjoint pages have deterministic checks. DPO is not the
default for faithful OCR because preference pairs can reward fluent but
unsupported text.

Each rollout receives zero reward if any non-compensable condition fails:

- invalid schema or malformed geometry;
- impossible or out-of-page boxes;
- missing EOS or repetition loop;
- unsupported text;
- changed critical clinical literal;
- missed required abstention; or
- broken control, table-cell, or reading-order relationship.

Among valid outputs, score literal text, box IoU, reading order, table
topology, control state and label association separately. Begin with four
completions per page, one seed, and a 100-page falsification run. Scale only if
the reward ranks known-bad outputs below known-good outputs and cannot be gamed
by omission.

## Evaluation and promotion

Every candidate is compared on the same frozen pages and every failure stays
in the denominator. Report CER, WER, normalized edit distance, missed and
unsupported text, layout F1 and mAP, reading-order edit distance, TEDS and
GriTS, field and relation accuracy, control macro-F1, latency, peak VRAM, cost,
failure rate, and abstention rate.

Promote a checkpoint only when:

- the paired 95 percent confidence interval for the primary clinical composite
  is above zero;
- no critical substitution is introduced;
- no handwriting, faint, tiny, rotation, table, form, or control slice drops
  by more than 2 absolute points;
- schema validity is at least 99.5 percent;
- all disagreements and at least 50 apparent successes receive manual review;
- the latency and cost point improves the measured Pareto frontier; and
- the same protocol beats the selected 30B and frontier API baselines before
  any superiority claim.

### Frozen Phi-4 and GOT experiment

Before training, run both challengers on identical local source bytes. Preserve
each documented native image processor, but record every resize, tile, crop,
token limit, dtype, attention backend, latency, failure, and peak allocated and
reserved GPU-memory value.

- Full-page arm: all 2 C14 pages and all 20 C08 pages.
- Crop arm: the 44 C14 native handwriting crops, with scaled crops reported as
  a separate fixed arm rather than best-of-two selection.
- Prompt: literal transcription in reading order with line breaks preserved,
  no correction, inference, completion, or summarization.
- Decode: greedy, no sampling, declared 4,096-token ceiling, documented stop
  tokens only.
- Failures: OOM, exception, empty output, repetition, and truncation remain in
  the denominator.

The current C14 pipeline baseline is 2 of 2 pages completed, CER 0.2632, WER
0.4123, and 15 of 44 handwritten spans recovered exactly. Frozen Phi-4 reached
16 of 44 normalized-exact crops but made 23 critical substitutions. This does
not justify a live route. It does justify a small decoder-LoRA experiment
because Phi-4 was the strongest frozen model on the matched crop panel. C14
must remain untouched during training and tuning.

Promote an adapted crop specialist only if it improves the family-disjoint dev
set over its own frozen checkpoint, completes every C14 crop, reaches at least
36 of 44 normalized-exact C14 spans, introduces no critical substitution, and
does not increase unsupported text. Paying the 5.6B decode cost per crop is
otherwise unjustified.

## Sources

All sources accessed 2026-09-02.

- [Ministral 3 3B Base model card](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)
- [Ministral 3 raw configuration](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512/blob/main/config.json)
- [Ministral 3 BF16 model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
- [Transformers Mistral 3 documentation](https://huggingface.co/docs/transformers/model_doc/mistral3)
- [Mistral fine-tuning repository](https://github.com/mistralai/mistral-finetune)
- [Phi-4 Multimodal model card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [Phi-4 vision LoRA configuration](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/vision-lora/adapter_config.json)
- [Phi-4 vision fine-tuning example](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_finetune_vision.py)
- [Phi-4 technical report](https://arxiv.org/abs/2503.01743)
- [GOT-OCR2.0 paper](https://arxiv.org/abs/2409.01704)
- [GOT-OCR2.0 official repository](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [GOT-OCR2.0 released implementation](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/GOT-OCR-2.0-master/GOT/model/GOT_ocr_2_0.py)
- [GOT-OCR2.0 model](https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- [Granite Vision 4.1 4B model card](https://huggingface.co/ibm-granite/granite-vision-4.1-4b)
- [Granite Vision repository](https://github.com/ibm-granite/granite-vision-models)
- [Florence-2 large model card](https://huggingface.co/microsoft/Florence-2-large-ft)
- [PARSeq repository](https://github.com/baudm/parseq)
- [LightOnOCR-2 paper](https://arxiv.org/abs/2601.14251)
- [LightOnOCR-2 configuration](https://huggingface.co/lightonai/LightOnOCR-2-1B/blob/main/config.json)
- [LightOnOCR target mix](https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126)
- [LightOnOCR bounding-box mix](https://huggingface.co/datasets/lightonai/LightOnOCR-bbox-mix-0126)
- [olmOCR-2 paper](https://arxiv.org/abs/2510.19817)
- [olmOCR repository](https://github.com/allenai/olmocr)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [TRL supervised fine-tuning documentation](https://huggingface.co/docs/trl/sft_trainer)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [DocLayNet repository](https://github.com/DS4SD/DocLayNet)
- [PubTables-1M repository](https://github.com/microsoft/table-transformer)
- [PubTables-1M paper](https://openaccess.thecvf.com/content/CVPR2022/html/Smock_PubTables-1M_Towards_Comprehensive_Table_Extraction_From_Unstructured_Documents_CVPR_2022_paper.html)
- [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19)
- [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/)
- [IAM handwriting dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [NVIDIA A30 specification](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
- [NVIDIA A40 specification](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/nvidia-a40-datasheet.pdf)
- [Google Cloud GPU pricing](https://cloud.google.com/products/compute/gpus-pricing)
- [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS Price List API](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-the-aws-price-list-bulk-api-fetching-price-list-files-manually.html)
