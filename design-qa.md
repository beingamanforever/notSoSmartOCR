# Design QA

## Final surface

- Demo: `http://127.0.0.1:8080/`
- Desktop result: `artifacts/research/design-qa/demo-release-result.png`
- Full-page result: `artifacts/research/design-qa/demo-release-full.png`
- State: the generated clinical table parsed successfully with timing, evidence diagnostics, overlays, rendered output, and downloads visible.

## Verified workflow

- Generated example selection and Parse complete end to end.
- The client timer starts on Parse and stops on completion.
- Rendered, Layout, Processed, Raw, JSON, and Failures tabs expose distinct views.
- Copy JSON writes the result to the browser clipboard.
- JSON and Markdown downloads become available after parsing.
- Region selection exposes kind, confidence, provider, geometry, provenance, alternatives, structure, and literal text.
- Selected-crop handwriting review returns an evidence-linked candidate or an explicit abstention.
- Keyboard focus states, labels, tab roles, selected states, and live status regions are present.
- Browser console warnings and errors: none.

## Visual review

The final UI uses three semantic roles: warm neutral surfaces, navy structure and success, and amber actions and review states. The input, examples, status, diagnostics, preview, and output follow a clear top-to-bottom hierarchy. At the desktop breakpoint, preview and output remain visible together and the result panel has enough width for structured tables. Text wraps inside cards, document imagery remains sharp, and controls remain visually distinct without decorative color noise.

The layout was also checked at a 390 x 844 CSS viewport. The document and body widths remained within the viewport with no horizontal overflow.

No actionable P0, P1, or P2 findings remain.

final result: passed
