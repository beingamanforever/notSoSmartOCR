# Private OCR weak-supervision index

## Outcome

The downloaded corpus is indexed offline by source PDF, provider, and source kind. The index stores paths and aggregate agreement only. It never writes document text and never labels a model or Textract output as ground truth.

## Corpus shape

- 50 source PDFs
- Seven model or pipeline transcript collections with 50 Markdown files each
- One Textract collection with 49 Markdown transcripts and paired run/layout metadata
- Direct-model collections use shortened names for most `IF` documents
- Textract adds timestamp suffixes

The completed local run produced 400 expected document-provider alignments: 399 successful transcripts and one missing Textract transcript. No duplicates, empty transcripts, unreadable files, unmatched names, or provider page failures were observed. All 50 source documents remained in the denominator.

The indexer resolves only exact contract stems and unique shortened aliases. Ambiguous aliases remain unmatched. It does not use fuzzy names or document content.

## Outputs

`prepare_s3_weak_supervision.py` writes a caller-selected private directory containing:

- `alignments.jsonl`: every expected document-provider pair, including missing, empty, duplicate, unreadable, and partial-failure states
- `train.jsonl`, `dev.jsonl`, and `audit.jsonl`: family-grouped path-only candidates
- `summary.json`: failure-inclusive counts and privacy semantics

Agreement is mean pairwise token F1 across successful providers. It is recorded only as a weak-supervision ranking signal. Development and audit candidates are never marked eligible for weak supervision, and every candidate requires human review before it can become reference data.

With seed 17 and 15 percent development and audit ratios, the family split contains 34 training documents, 8 development documents, and 8 audit documents. Each of the 50 document families occurs in exactly one split.

## Safety boundary

- No network or model calls
- No S3 writes
- No credentials read or stored
- No document text copied into the index or this report
- No consensus, provider output, or Textract output treated as held-out truth

Accessed locally on 2026-09-02.
