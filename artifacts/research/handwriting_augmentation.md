# Handwriting augmentation for clinical form crops

Accessed: 2026-09-02

## Decision

Use STRAug as a catalog, not as a preset. The first handwriting fine-tune should use a small online, train-only recipe with mild affine and acquisition changes. Keep half of training draws unchanged, apply at most two transforms, preserve every visible stroke, and split real source families before producing crop views or augmentations.

This recommendation has medium confidence. Scene-text and handwriting studies support conservative augmentation, but none establishes optimal settings for faint handwriting embedded in clinical forms.

## Why the full STRAug recipe is unsafe here

STRAug contributes 36 functions in eight groups: Warp, Geometry, Noise, Blur,
Weather, Camera, Pattern, and Process. It reports aggregate absolute accuracy
gains of 0.89 to 2.10 points across six scene-text recognizers, but its own
group ablation shows why it is a catalog rather than a preset. On RARE, Pattern
has only a 0.01-point aggregate gain and falls on several individual datasets;
Camera falls by 2.56 points on CUTE80, Noise falls by 1.38 points on CUTE80,
and other groups also have dataset-specific negative cells. The combined
RandAugment sweep peaks between two and four groups, not at the maximum number
of transforms.

Its training data and severity choices do not match this task. STRAug uses
synthetic scene-text word crops, while our measured failures contain thin pen
strokes, printed neighbors, rules, checkboxes, and empty fields. Several
released settings are destructive for these crops, including JPEG quality 15
to 25, pixelation factors 0.4 to 0.6, Gaussian noise starting at 0.06, blur up
to 2 pixels, and large rotations.

The closest controlled handwriting study found that small rotation, shift, shear, elastic deformation, and scaling helped stenographic lines. Its combined policy reduced mean CER from 0.3174 to 0.3090 and mean WER from 0.5715 to 0.5603. Large rotation, erosion, dilation, 40 percent masking, and strong Gaussian blur were harmful. The authors connect morphology and blur failures to thin strokes and loss of small-symbol distinctions.

A 2026 TrOCR ablation provides a useful cross-check and an important direction
correction. It does not report that augmentation harmed both held-out sets.
Removing its full augmentation bundle raised mean CER by 0.45 points on the
Cortonese set, a non-significant change, and by 0.75 points on READ-16, a
significant change. Removing both CLAHE and augmentation was also 0.45 points
worse on READ-16 but non-significant. The same recipe therefore had uncertain
value on one corpus and clear value on another. This supports a data-specific
ablation, not a universal transform bundle. The paper's bundle includes
rotation up to 5 degrees, elastic deformation, blur, brightness, contrast,
speckle noise, morphology, and simulated shadows or stains. Those settings are
candidate families to test selectively, not defaults for thin clinical pen
strokes.

## Initial recipe

For each logical field, sample one view and one outcome per epoch:

| Choice | Probability | Range |
| --- | ---: | --- |
| Tight crop | 50% | Exact reviewed box |
| Context crop | 50% | Expand each side by `max(8 px, 0.5 x field height)` |
| Unchanged | 50% | No transform |
| One geometry transform | 25% | From the geometry rows below |
| One acquisition transform | 20% | From the acquisition rows below |
| One geometry plus one acquisition | 5% | Maximum of two transforms |

Recommended candidate ranges:

| Family | Transform | Candidate range |
| --- | --- | --- |
| Geometry | Rotation | `[-1.5, +1.5]` degrees |
| Geometry | Translation | horizontal `+/-1.5%`, vertical `+/-3%` |
| Geometry | Isotropic scale | `[0.90, 1.05]` |
| Geometry | Horizontal shear | `[-3, +3]` degrees |
| Acquisition | Gamma | `[0.90, 1.10]` |
| Acquisition | Smooth illumination gain | `[0.95, 1.05]` |
| Acquisition | JPEG quality | `[75, 95]` |
| Acquisition | Downsample and restore | factor `[0.85, 1.0]` |
| Acquisition | Gaussian noise sigma | `[0.003, 0.012]` on `[0,1]` pixels |

Pad by at least one quarter of the crop height before any geometry transform. Fill from the median outer-border value and retain the transformed canvas. Never center-crop after a transform. Preserve aspect ratio.

Do not use in the first candidate: CutOut, CutMix, MixUp, erasing, masking, flips, 90-degree rotation, thresholding, erosion, dilation, synthetic scribbles, synthetic checkmarks, aggressive blur, weather effects, TPS warps, elastic deformation, or composite degradation pipelines.

## Context and abstention supervision

The measured problem is not only missed handwriting. The current 30-field manual evaluation has a 33.6 percent hallucinated-character rate. Two training cases target that failure directly:

1. Context-padded pairs: use the same handwritten field as a tight crop and as a crop containing nearby printed labels and rules. Both have the same target. This teaches the recognizer not to copy printed neighbors.
2. Abstention examples: use real blank fields, printed-only fields, stray non-character marks, and unreadable handwriting. Do not manufacture blanks by deleting ink from positives.

Suggested batch mix:

| Target state | Share | Output |
| --- | ---: | --- |
| Resolved handwriting | 75% | Literal transcription |
| Blank field | 10% | `<NO_HANDWRITING>` |
| Printed-only field | 10% | `<NO_HANDWRITING>` |
| Stray mark | 5% | `<NO_HANDWRITING>` |

Use `<UNREADABLE>` when reviewers cannot distinguish a text-bearing mark from noise. An empty output is reserved for downstream rendering after the explicit state has been verified.

## Lineage and split controls

- Split by document family before cropping or augmentation.
- Sample family, then document, then logical field so dense pages do not dominate.
- Use at most one field from a page per microbatch and two fields from a family per effective batch.
- Give tight and context views the same field identity and target.
- Generate transformations online. Do not count augmented siblings as independent examples.
- Cap recorded descendants at 8 to 16 views per parent for analysis.
- Validation and test contain only unchanged real images.

The current 15-field dual-reviewed pilot is accurate on manual inspection but comes from one source packet. It validates the labeling flow only. It is not adequate evidence for training or generalization.

## Cheapest valid ablation

Hold optimizer steps, example counts, initialization, seed, and abstention mix constant.

| Arm | Crop policy | Augmentation |
| --- | --- | --- |
| A | Padded only | None |
| B | 50:50 tight/context | None |
| C | 50:50 tight/context | Geometry only |
| D | 50:50 tight/context | Acquisition only |
| E | 50:50 tight/context | Combined schedule |

Screen all arms with one seed on the family-disjoint real dev set. Rerun Arm B and the safest candidate with two additional matched seeds. Touch the locked real test set once after selection.

Report exact match, normalized exact match, CER, character insertions, deletions, substitutions, critical clinical substitutions, false transcription on every abstention subtype, false resolution on unreadable fields, proposal recall, and failures in every denominator. Report paired field-level differences and family-level uncertainty.

Adopt augmentation only if all three matched seeds improve real-only dev normalized exact match by at least 5 points, CER improves in at least two seeds, and neither unsupported text nor critical substitutions increase. On locked test, require at least `max(2 fields, 5% of positives)` additional exact fields with no abstention regression.

## Training order

1. Build 500 to 1,000 independently reviewed real fields across at least 30 held-out families.
2. Fine-tune the existing Phi-4 vision-decoder LoRA with assistant-token-only loss.
3. Establish the no-augmentation and context-pair baselines.
4. Run the augmentation ablation above.
5. Consider OPD only after the frozen crop and page gates pass. Keep reward checks deterministic and reject rollouts with unsupported literals.

## Primary sources

- [STRAug paper](https://arxiv.org/pdf/2108.06949) and [official code](https://github.com/roatienza/straug)
- [Handwriting augmentation study](https://arxiv.org/pdf/2303.02761) and [archived author code](https://doi.org/10.5281/zenodo.7905299)
- [Best Practices for Handwritten Text Recognition](https://arxiv.org/pdf/2404.11339) and [official code](https://github.com/georgeretsi/HTR-best-practices)
- [TrOCR for Medieval HTR ablation](https://arxiv.org/abs/2606.24302) and [official code](https://github.com/LaudareProject/TrOCR-analysis)
- [Union14M paper](https://arxiv.org/pdf/2307.08723) and [official code](https://github.com/Mountchicken/Union14M)

Source-reported findings, local measurements, and proposed settings are separated above. The numeric transform ranges are hypotheses to ablate, not published clinical optima.
