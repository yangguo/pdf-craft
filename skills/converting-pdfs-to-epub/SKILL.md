---
name: converting-pdfs-to-epub
description: Use when converting PDF files to EPUB, especially books that need selectable text, cover metadata, OCR decisions, Calibre compatibility, retained diagrams/photos/tables, or fixes for image-only, missing-cover, remote-image, or broken-layout EPUB outputs.
---

# Converting PDFs to EPUB

## Overview

Convert PDFs into EPUBs that are readable in Calibre and usable on e-readers. The default target is selectable prose text plus local embedded images only where the PDF has covers, figures, photos, charts, diagrams, or tables that text extraction cannot preserve.

## Workflow

1. Inspect the PDF before converting: page count, metadata, text layer, embedded outline/bookmarks, embedded images, drawing-heavy pages, list of figures/tables, and representative chapter starts.
2. Choose the path:
   - Use readable native text as the source of truth when its order and encoding are sound; treat the text-layer ratio as a screening signal, not a fixed threshold.
   - Use MinerU to cross-check structure or repair complex, missing, or corrupted ranges instead of replacing trustworthy prose.
   - Use `pdf2epub_paddle.py --api mineru` (faster) or `--api paddle` (default) when pages are scanned or text length is near zero across body pages.
   - Use page images only as a last resort; Calibre and e-readers handle text EPUBs better.
3. Preserve local visuals selectively: cover, photos, diagrams, charts, and table pages. Do not link remote images from XHTML.
4. Build the EPUB with title, author, language, cover metadata, nav, NCX, CSS, and local manifest items.
5. Verify with fresh commands before reporting success.

## OCR Decision

Call `pdf2epub_paddle.py` only when OCR is actually needed. Use `--api mineru` for faster processing with the MinerU v4 API (set `MINERU_API_TOKEN` in `.env`).

| Inspection result | Use OCR? | Backend |
|---|---|---|
| User explicitly requests Paddle/OCR | Yes | `paddle` (default) |
| User explicitly requests MinerU | Yes | `mineru` |
| Body pages are scanned images with little/no text | Yes | `paddle` or `mineru` |
| `inspect` shows a low text-layer ratio or many body pages with `text_len` near zero | Yes | `mineru` (faster) |
| Extracted text is unreadable garbage, wrong script, or badly corrupted | Yes | `mineru` (faster) |
| PDF has readable prose text but cover/photos/tables are images | No | Use hybrid text+visual conversion |
| PDF has vector charts/tables with searchable captions/text | No | Add visual snapshots for those pages |
| Only the cover page is an image | No | Embed the cover locally |

For mixed PDFs, OCR only the scanned page ranges if the project OCR workflow supports it; otherwise use OCR for the source pass, then still run EPUB validation and visual-retention checks.

## MinerU-Assisted Extraction

When using MinerU MCP for a large PDF, honor the service's current page limit and split into ordered, inclusive ranges when necessary. Submit different ranges of the same source PDF as **separate MCP calls with separate output directories**. In observed workflows, batching same-basename ranges in one `file_sources` request duplicated or overwrote output while reporting success; separate calls avoid that collision.

Treat each saved result as untrusted until checked. Confirm that ranges cover the source exactly once and in order; compare file sizes, hashes, opening and closing text, representative headings, and approximate character counts against native extraction. Identical hashes for different ranges, missing boundary text, or implausibly small output require an independent rerun rather than packaging the Markdown.

Prefer the PDF's embedded outline for chapter boundaries when it matches printed contents and actual heading pages. Use MinerU headings only to fill gaps or challenge a suspect outline. If the converter's automatic outline handling is too coarse, pass explicit `--section` ranges for the useful chapter level.

## Helper Script

Use `skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py` for text-layer PDFs.

```bash
python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py inspect book.pdf

python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py convert book.pdf book.epub \
  --title "Book Title" \
  --author "Author Name" \
  --translator "Translator Name" \
  --language zh-Hant --layout flow \
  --isbn "978-1-23456-789-0" \
  --publisher "Example Press" \
  --published "2026-07-18" \
  --description "A concise catalog description." \
  --visual-pages 66,72-73,148,210 \
  --section "Front Matter:1-12" \
  --section "1 Introduction:13-42"

python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py verify book.epub \
  --expected-visual-pages 66,72-73,148,210 \
  --expected-label "Figure 1.1" \
  --expected-label "Table 2.1"
```

If the repository has a better project-specific converter, prefer that. For Paddle/OCR requests or scanned PDFs, inspect `pdf2epub_paddle.py` or the project OCR workflow, but still apply the verification checklist below. If the generated EPUB has CJK garbling, missing sentences, broken chapter boundaries, or cross-page/figure-adjacent OCR loss, switch to the `repairing-ocr-epubs` skill.

The hybrid converter automatically calibrates vector-drawing complexity against the document baseline so recurring crop marks and page furniture do not turn every page into a visual snapshot. On CJK-dominant pages it also joins wrapped line blocks into paragraphs using indentation and vertical spacing, while leaving non-CJK block boundaries unchanged. Explicit `--visual-pages` and `--section` values remain the authoritative override when automatic inspection is uncertain.

## Reflow and Source-Specific Overrides

The helper defaults to `--layout flow`: continuous prose with source-page anchors. Use `--layout pages` to retain PDF page breaks. `--join-page-continuations` enables conservative CJK joins based on native geometry, font size, and indentation; review changed boundaries. Joins preserve inline markup and stop at chapters and figures. Never merge all adjacent paragraphs simply because one lacks final punctuation.

Use `--column-split X` only for a reviewed, consistent two-column layout (X is in PDF points). Full-page raster artwork is retained automatically; use `--skip-page-rasters` only after checking that those regions are scans rather than illustrations. Explicit visual-page selections take precedence. Automatic chapter boundaries retain front matter before the first bookmark and deduplicate identical starts.

Keep book-specific titles, page mappings, spelling corrections, and repair ledgers outside the generic converter. Record exact repair preconditions and original-page evidence. See [the source-faithful validation guide](../../docs/hybrid-epub-validation.md) for workflow details and limits.

## Visual Retention

Use visual snapshots for:

| Content | EPUB handling |
|---|---|
| Cover | Embed as `cover.jpg` and set EPUB cover metadata |
| Photos/scans/diagrams | Local image item in the EPUB |
| Charts/vector figures | Snapshot the PDF page or tight crop |
| Complex tables | Local snapshot and searchable caption; use a dedicated extractor for semantic table cells |
| Plain prose | Extract as XHTML text, not images |

Find visual pages from the PDF’s list of figures/tables, `page.get_images()`, `page.get_drawings()`, and caption searches like `Figure 4.2` or `Table 6.1`. Multi-page tables need every continuation page.

## Required Verification

Run these before saying the EPUB is complete:

```bash
python3 -m zipfile -t output.epub
ebook-meta output.epub
ebook-convert output.epub /tmp/output_verify.txt
python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py verify output.epub \
  --expected-visual-pages "..." \
  --expected-label "..."
```

Confirm:

- Calibre can parse the EPUB.
- Cover metadata exists and the cover image is local.
- Normal prose is XHTML text, not one image per page.
- Expected captions are searchable text.
- All expected visual pages are embedded.
- EPUB contains no remote `http://` or `https://` image references.
- The `mimetype` member is first and uncompressed, package XML parses, and every local `href`/`src` resolves to an archive member.

## Common Failures

| Symptom | Fix |
|---|---|
| Calibre cannot open EPUB | Validate zip, OPF manifest, spine, nav, NCX; rebuild instead of patching blindly |
| EPUB has no cover | Use `book.set_cover()` and verify OPF cover metadata |
| Text-only EPUB lost charts/tables | Add local snapshots for figure/table pages |
| EPUB is huge or slow | Do not snapshot every page; preserve prose as text |
| Captions missing after visual cleanup | Keep all extracted page text, then add visual snapshots |
| Remote image timeout | Download/embed images locally or remove remote references |
| OCR produces noisy layout | Use PDF text layer when available; reserve OCR for scanned pages |
| OCR text has CJK garbling or missing spans | Use `repairing-ocr-epubs`: scan, map to PDF pages, run targeted MinerU reruns, then rebuild and verify |
| Different MinerU ranges contain identical content | Rerun each range in a separate MCP call and output directory; compare hashes and boundary text before use |
| EPUB chapters follow broad parts instead of articles | Cross-check the embedded outline with printed contents, then pass explicit `--section` ranges |

## Reporting

Report the output path and the evidence, not assumptions: archive check, Calibre parse result, metadata, image count, visual page list, and any known limitations.
