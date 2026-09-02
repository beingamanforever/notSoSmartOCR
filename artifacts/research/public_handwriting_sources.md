# Public handwriting sources

Accessed 2026-09-02. This note separates source-reported facts, local URL checks,
and adoption decisions. No dataset archive was downloaded during this review.

## Decision

Use READ 2017 Train-A as the first small external training slice. It is 50 pages,
21,429,672 bytes, CC BY 4.0, and includes PAGE annotations. Keep every READ test
archive evaluation-only.

For writer-sensitive experiments, prefer NIST SD19 `by_write.zip` or full HSF
pages. Do not use `by_field`, `by_class`, or `by_merge` for writer-disjoint
splits because the official guide says those organizations discard writer
identity. NIST SD19 and SD6 are much larger than the default download limit and
must be selected explicitly with a byte cap that covers the requested archive.

Do not automate GNHK or CENSUS-HWR downloads yet. Their papers remain useful,
but their original official data endpoints no longer expose a stable archive on
the accessed date. CENSUS-HWR also lacks an explicit dataset license in the
paper and current project page.

## Source review

### GNHK

- Primary sources: [official repository](https://github.com/GoodNotes/GNHK-dataset),
  [official paper](https://github.com/GoodNotes/GNHK-dataset/blob/main/gnhk_paper.pdf),
  and the original [dataset endpoint](https://www.goodnotes.com/gnhk).
- Source-reported scope: 687 camera-captured document images, 39,026 text
  instances, 9,363 lines, and 172,936 characters. At most five images came from
  one writer. The paper reports a random 75/25 train/test split.
- Labels: per-image JSON contains literal text, quadrilateral polygons, line
  indexes, and handwritten or printed type. Special labels cover math, illegible
  scribbles, and detected regions without text.
- License: the official repository states CC BY 4.0 and asks users to contact
  the authors if a document should be removed.
- Availability check: the repository still points to `goodnotes.com/gnhk`, but
  the canonical endpoint redirected to the Goodnotes homepage on 2026-09-02.
  No archive URL or archive size is published in the repository. The downloader
  therefore has no GNHK entry.
- Grouping caution: the paper limits pages per writer, but its published JSON
  schema does not document a writer identifier. Do not claim a writer-disjoint
  split until the actual archive is recovered and inspected.

### NIST Special Database 19

- Primary sources: [official dataset page](https://www.nist.gov/srd/nist-special-database-19),
  [second-edition user guide](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf),
  and [NIST data terms](https://www.nist.gov/open/license).
- Source-reported scope: the landing page summarizes full HSF pages from 3,600
  writers and more than 800,000 hand-checked character images. The second-edition
  guide describes 3,669 full forms and writer partitions numbered through 4,169.
  These are different official summaries, so preserve the exact release and
  partition identifiers instead of collapsing them into one writer count.
- Grouping: `by_write` retains HSF partition and writer directories. `hsf_page`
  retains the full form file stem. The guide explicitly says `by_field` discards
  writer information and `by_class` discards writer and field information.
- Use terms: SD19 is a NIST Standard Reference Data product. The NIST terms page
  warns that SRD compilations can be copyright protected and directs users to
  product-specific terms. Free download is not the same as a permissive license.
  Keep attribution and the original archive terms, and obtain a rights review
  before redistributing the data or derived public releases.
- Exact official downloads and HEAD `Content-Length` observed 2026-09-02:

| Organization | URL | Bytes | Grouping use |
| --- | --- | ---: | --- |
| by writer | [by_write.zip](https://s3.amazonaws.com/nist-srd/SD19/by_write.zip) | 568,113,446 | Preferred for writer-disjoint character work |
| full pages | [hsf_page.zip](https://s3.amazonaws.com/nist-srd/SD19/hsf_page.zip) | 348,196,072 | Preferred for form and full-page work |
| by field | [by_field.zip](https://s3.amazonaws.com/nist-srd/SD19/by_field.zip) | 540,793,335 | Writer identity discarded |
| by class | [by_class.zip](https://s3.amazonaws.com/nist-srd/SD19/by_class.zip) | 1,031,576,378 | Writer and field identity discarded |
| merged class | [by_merge.zip](https://s3.amazonaws.com/nist-srd/SD19/by_merge.zip) | 542,733,164 | Not suitable for writer grouping |

### NIST Special Database 6

- Primary sources: [official dataset page](https://www.nist.gov/srd/nist-special-database-6),
  [official user guide](https://s3.amazonaws.com/nist-srd/SD6/SD06_users_guide.pdf),
  [archive](https://s3.amazonaws.com/nist-srd/SD6/sd06.zip), and
  [NIST data terms](https://www.nist.gov/open/license).
- Source-reported scope: 5,595 binary form pages and matching answer files from
  900 simulated tax submissions, spanning 20 form faces and 12 tax forms. NIST
  states that the pages are synthesized and contain no real tax data.
- Grouping: each simulated submission is a directory such as `r0200`; page and
  answer pairs share a stem such as `r0200_00`. Keep the submission directory as
  the document-family key when splitting.
- Size: the official archive returned `Content-Length: 969640744` on 2026-09-02.
- Use terms: the same NIST SRD caution applies. Do not treat synthesized content
  or a free download as evidence of a permissive dataset license.
- Best role: form layout, empty-field, entry-field, and hand-print stress data.
  It is not a substitute for real cursive clinical handwriting.

### READ 2017 HTR

- Primary source: [official Zenodo record](https://zenodo.org/records/835489),
  DOI [10.5281/zenodo.835489](https://doi.org/10.5281/zenodo.835489), and
  [record metadata API](https://zenodo.org/api/records/835489).
- License: CC BY 4.0 in the official record metadata.
- Source-reported scope: Train-A has 50 pages with manually revised baselines
  and transcripts. Train-B has 10,000 pages with page-level transcripts. Test-A
  has 65 pages, Test-B1 reuses Test-A geometry, and Test-B2 has 57 pages.
- Annotation caution: Train-A baselines and transcripts were manually revised,
  but its polygons were not. The Zenodo record does not publish writer IDs.
  Preserve PAGE/image paths as document keys and inspect the archive before
  asserting author-disjoint partitions.
- Exact files published by the Zenodo API:

| File | URL | Bytes | Use here |
| --- | --- | ---: | --- |
| Train-A.tbz2 | [download](https://zenodo.org/api/records/835489/files/Train-A.tbz2/content) | 21,429,672 | Named small training subset |
| Train-B_batch1.tbz2 | [download](https://zenodo.org/api/records/835489/files/Train-B_batch1.tbz2/content) | 1,889,479,353 | Explicit large download only |
| Train-B_batch2.tbz2 | [download](https://zenodo.org/api/records/835489/files/Train-B_batch2.tbz2/content) | 1,897,592,301 | Explicit large download only |
| Test-A.tgz | [download](https://zenodo.org/api/records/835489/files/Test-A.tgz/content) | 70,860,990 | Evaluation only |
| Test-B1.tgz | [download](https://zenodo.org/api/records/835489/files/Test-B1.tgz/content) | 70,772,074 | Evaluation only |
| Test-B2.tgz | [download](https://zenodo.org/api/records/835489/files/Test-B2.tgz/content) | 48,043,177 | Evaluation only |
| Baseline.tgz | [download](https://zenodo.org/api/records/835489/files/Baseline.tgz/content) | 22,060,822 | Reference system, not training data |

### CENSUS-HWR

- Primary sources: [paper](https://arxiv.org/abs/2305.16275) and the paper's
  original [data URL](https://censustree.org/data.html).
- Source-reported scope: 1,812,014 grayscale word images, 1,865,134 texts, and a
  10,711-word vocabulary extracted from the 1930 and 1940 US censuses. The paper
  estimates about 70,000 enumerators in each year.
- Availability check: `censustree.org/data.html` returned 404 on 2026-09-02.
  The site's current `/data` page is a census-linkage repository selector and no
  longer exposes the HWR archive. No current direct archive or published byte
  size was found on the official project site.
- License caution: the paper says the data was freely downloadable, but neither
  that statement nor the arXiv paper license establishes a dataset license. Keep
  this source disabled until an official archive and explicit reuse terms are
  recovered.
- Grouping requirement if recovered: retain census year and enumerator as the
  author-family keys. Split by enumerator, not by word crop, to prevent writer
  leakage.

## Downloader behavior

`experiments/download_handwriting_data.py` uses only the Python standard
library. It never downloads by default. It prints the selected files, exact
declared bytes, output path, and total cap. `--download` is required for network
I/O. The default 64 MiB total cap permits only the named READ Train-A slice.

```bash
python experiments/download_handwriting_data.py --list
python experiments/download_handwriting_data.py data/public/handwriting \
  --item read-2017-train-a
python experiments/download_handwriting_data.py data/public/handwriting \
  --item read-2017-train-a --download
```

NIST requires an explicit item and a raised exact cap. This command remains a
plan until `--download` is added:

```bash
python experiments/download_handwriting_data.py data/public/nist-sd19 \
  --item nist-sd19-by-write --max-bytes 568113446
```

For any other public source, pass a JSONL selection file. Every row must state
the source, URL, basename, declared byte count, and writer/document grouping:

```json
{"id":"page-1","source":"public source","url":"https://example.org/page-1.png","filename":"page-1.png","declared_bytes":1234,"groups":{"writer":"writer-7","document":"document-3"}}
```

The tool rejects credential-bearing URLs, signed or tokenized URLs, local IPs,
private/manual S3 URLs, duplicate filenames, unexpected byte counts, and a
selection whose declared total exceeds the cap. Only the exact public NIST SRD
catalog URLs may use S3. Archives are left intact, and `downloads.jsonl` records
the original source and grouping fields without computing hashes.
