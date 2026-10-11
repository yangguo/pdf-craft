# Source-faithful hybrid EPUB conversion

The hybrid helper converts readable native PDF text to selectable XHTML and
embeds local visual regions. It does not proofread every word or recognize
scanned pages. Structural validation and text preservation are separate checks.

## Intake and coverage

Inspect the text layer, bookmarks, images, drawings, and representative pages.
Check chapter starts, dense body pages, page turns, notes, and image-adjacent
text. A high text-layer ratio does not prove correct decoding or reading order.
Use OCR for scanned or corrupted ranges; retain sound native prose.

Automatic level-one bookmarks include pages before the first bookmark as front
matter and collapse duplicate page starts. Compare the resulting chapter list
with printed contents. Explicit `--section` ranges override automatic coverage:
check that they cover every intended source page exactly once, with no gaps or
overlaps. Empty separators may be omitted only after checking their content.

For MinerU, check the current service page limit. Submit ordered inclusive ranges
in separate calls and separate output directories. Same-basename range requests
have produced duplicate/overwritten outputs in observed workflows. Check hashes,
file sizes, boundary text, headings, and approximate character counts; a success
response alone does not establish distinct or complete extraction.

## Paragraphs and reading order

A PDF block is not necessarily a paragraph. CJK line fragments are joined using
horizontal indentation transitions, explicit leading whitespace, and vertical
spacing. Repeated inset lines stay together; same-line fragments do not create a
new paragraph. Ordinary non-CJK block boundaries remain separate.

`--layout flow` is the default. It removes PDF page wrappers and forced page
breaks, while preserving `p0001`-style source-page anchors. `--layout pages`
retains the wrappers and page breaks for review or page-oriented reading.

`--join-page-continuations` is opt-in. It joins only candidate CJK page turns with
an unfinished full-width last line near the bottom and a same-size unindented
first line near the top. It preserves inline markup, links, and page IDs; it does
not join across chapters, figures, or explicit column mode. This heuristic can
miss continuations and cannot establish authorial paragraph boundaries for
poetry, lists, or unusual layouts. Review every changed boundary against the PDF.
A sentence ending does not itself prove a paragraph ending.

There is no page-number-specific column switch. For a source with a consistent
two-column layout, use `--column-split X`, where X is a PDF coordinate in points.
Mixed layouts need separate reviewed conversion ranges or a dedicated extractor.

## Visuals and notes

Full-page raster artwork is retained by default. Large image area does not prove
that an image is a scanned text page. After inspecting a scan-heavy source,
`--skip-page-rasters` excludes uncaptioned large rasters from automatic selection.
`--visual-pages` supplies explicit selections and takes precedence over that
heuristic. Review the cover and every substantive source visual independently.
Tiny raster marks and recurring vector furniture may be excluded by detection.

Captioned figures/tables can be split into separate local crops with searchable
captions. Text inside a retained crop is omitted from surrounding prose to avoid
duplication; a table snapshot is visually preserved, but its cell text is not a
semantic searchable table. A snapshot can also be appended after the page's
prose rather than at its exact original position. Use a dedicated table/layout
extractor when those distinctions matter.

Smaller text near the bottom is treated as a candidate note and collected at the
chapter end. This is a font/position heuristic, not reference matching: no
callout backlinks are created. Review notes and low-font-size body text against
the source. Misencoded emphasis digits are removed only when their font
identifies them as layout markers. Legitimate numeric citations, including
repeated digits, remain text. Replacement-character corruption needs source
inspection/OCR rather than speculative substitution.

## Repairs and content audit

Keep book-specific repairs outside the generic converter. A repair ledger should
record source identity/hash, PDF page number, exact old/new strings, expected
match count, reason, and the original-page evidence reviewed. Apply a repair only
when its exact preconditions match; never replace a repeated character or
speaker name globally based on a single example.

Compare normalized extracted text and packaged XHTML by section. Ignore only
recorded formatting changes, reconstructed navigation, duplicate source
headings, and approved repairs. Investigate unmatched substrings instead of
accepting similar aggregate character counts. Include notes and captions; when
text moves into an image, record and visually check that region separately.
Matching an extraction transcript proves packaging preservation, not that the
transcript faithfully recognized the source. Full proofreading requires an
independent source comparison, including OCR-sensitive names and punctuation.

## Validation commands

```bash
python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py inspect book.pdf
python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py convert book.pdf book.epub \
  --language zh-Hant --layout flow --translator "Translator Name"
python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py verify book.epub \
  --expected-visual-pages 12,30-31 --expected-label "Figure 1.1"
python3 -m zipfile -t book.epub
java -jar /path/to/epubcheck.jar book.epub
ebook-meta book.epub
ebook-convert book.epub /tmp/book-epub-audit.txt
python3 -m pytest tests/ -q
git diff --check
```

The helper checks ZIP layout, XML, local resources, expected visual pages/labels,
and remote image references. EPUBCheck checks EPUB conformance. Calibre checks
reader parsing; inspect rendered chapter starts, captions, notes, and repaired
boundaries as well. Record tool versions and actual results. If a tool is absent,
report that check as not run. Neither a green test suite nor EPUBCheck establishes
whole-book completeness or perfect OCR.

Do not commit source books, generated EPUBs, page renders, OCR output, credentials,
or temporary book-specific repair scripts. Commit reusable code, synthetic
fixtures, regression tests, and workflow documentation.
