# From OCR evidence to a reviewable document

OCR does not become trustworthy because its output looks polished. This final post explains how Not So Smart OCR keeps the relationship between source pixels, generated text, model scores, and user corrections.

## Start with one evidence record

The pipeline stores recognized content in a canonical region representation. A record has a stable identity within the result, a page-relative box, text, kind, provider, resolution state, and optional structure and provenance. The active reader retains detector geometry, raw decoder output, and native generation metadata.

Table cells refer back to their source evidence. Pictures refer to source pixels. Alternative readings and user corrections remain distinguishable from the model's original output. This allows a reviewer to ask where a value came from rather than merely whether it looks plausible.

## What the displayed confidence means

The native Falcon service exposes probabilities for generated tokens. For a sequence of N scored tokens, the displayed token score is the geometric mean:

```text
score = exp((log p(token_1) + ... + log p(token_N)) / N)
```

Log probabilities are numerically convenient: multiplying many small probabilities can underflow, while summing their logarithms is stable. Averaging also avoids making a longer reading automatically look worse merely because it contains more factors. The score remains sensitive to tokenization, context, and the model's own distribution.

It is **not** the probability that an extracted field is correct. A model can confidently drop a digit or recognize the wrong punctuation. The UI therefore labels the score as an uncalibrated decoder token score. A figure's detection score is labeled separately. Unknown scores remain unknown.

A minimum token score can expose a locally uncertain step that an average hides, but it is still not a correctness oracle. Calibration would require held-out labels and measurement of how predictions correspond to observed errors. That has not been established for every document category here.

Token offsets also do not imply word-level pixel coordinates. The native service validates token/text alignment before exposing offsets, and the evidence representation does not turn those offsets into fabricated word boxes.

## One result, several views

The demo offers text, visual inspection, Markdown, and JSON around the same result. The text view provides a readable document; the visual view links elements to source geometry; JSON retains structure and provenance for downstream consumers.

Markdown generation is deterministic rendering, not another model call. Literal text is escaped for its output context. Pipes, angle brackets, and line breaks inside table cells must not silently change cell content or be interpreted as arbitrary markup. Structured tables with spans require a representation that preserves their structure; plain pipe tables cannot express every merged-cell arrangement.

Math uses local KaTeX assets. A formula that cannot be rendered should not disappear. A formula that can be rendered still needs recognition review. The renderer's role is to display the reading faithfully, not to rewrite it into a more convincing answer.

Source and presentation ownership must remain separate: retaining evidence for inspection should not itself create a second visible copy. The restored route still has known duplication from overlapping layout predictions, so this is a remaining limitation rather than a universal guarantee.

## Review should preserve the context being judged

A reviewer can inspect a region alongside its source crop. The new crop download uses the full-resolution source pixels within the selected box. It does not download a scaled screenshot or include colored overlay labels. Browser checks compared the generated PNG pixels directly with the source for multiple element types.

Feedback stores the judged image together with the complete result and feedback. The current retention setting keeps the latest 1,000 records. This is durable feedback storage; ordinary browser sessions are a different mechanism and can expire when the service restarts. Feedback remains private and is excluded from the public repository.

Corrections carry revisions so an update based on an obsolete result can be rejected. The browser guards against out-of-order responses, and the API exposes the service composition and start time. A live update is verified by a fresh upload and an actual model run. Refreshing HTML alone does not prove that a long-running Python process loaded new OCR code.

## Evaluate the complete outcome

A lower character error rate can coexist with worse exact table cells. A duplicated label can be a software error, but the same label printed in two source locations is legitimate content. Removing every repeated string would improve one superficial metric while destroying the document.

Useful evaluation therefore separates:

- Recognition errors, including missing symbols, digits, and text.
- Repeated or missing source instances.
- Table structure and exact cell values.
- Reading order and source geometry.
- Checkbox states and unresolved content.
- End-to-end time, memory, and processing failures.

Public benchmark labels and model-generated private annotations are not equivalent. A GPT-generated reference is a draft until reviewed. Training and evaluation must keep source documents separate, and failures must remain in the denominator. The previous fine-tuning trials were not promoted because exact table cells regressed despite gains elsewhere.

The release checks cover service behavior, source ownership, table assembly, browser rendering, exports, and persistent feedback. Real examples also exposed remaining handwriting, formula, and small-text errors. The current serving choice prioritizes the earlier extraction quality. The replacement route was withdrawn; duplicate readings remain a known limitation.

Return to [image-to-text processing](01-from-image-to-text.md) or [layout and tables](02-layout-tables-and-crops.md).

Implementation: `contracts.py`, `rendering.py`, `demo.py`, `demo.html`, and `experiments/serve_falcon_layout.py`, inspected on 9 September 2026. The explanations above describe this implementation; they are not a general benchmark claim about its underlying models.
