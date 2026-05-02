---
name: converting-pdfs-to-epub
description: Use when converting PDF files to EPUB, especially books that need selectable text, cover metadata, OCR decisions, Calibre compatibility, retained diagrams/photos/tables, or fixes for image-only, missing-cover, remote-image, or broken-layout EPUB outputs.
---

# Converting PDFs to EPUB

## Overview

Convert PDFs into EPUBs that are readable in Calibre and usable on e-readers. The default target is selectable prose text plus local embedded images only where the PDF has covers, figures, photos, charts, diagrams, or tables that text extraction cannot preserve.

## Workflow

1. Inspect the PDF before converting: page count, metadata, text layer, embedded images, drawing-heavy pages, list of figures/tables, and chapter starts.
2. Choose the path:
   - Use a text-layer hybrid conversion when most pages already contain extractable text.
   - Use `pdf2epub_paddle.py` or the project OCR workflow first when pages are scanned or text length is near zero across body pages.
   - Use page images only as a last resort; Calibre and e-readers handle text EPUBs better.
3. Preserve local visuals selectively: cover, photos, diagrams, charts, and table pages. Do not link remote images from XHTML.
4. Build the EPUB with title, author, language, cover metadata, nav, NCX, CSS, and local manifest items.
5. Verify with fresh commands before reporting success.

## OCR Decision

Call `pdf2epub_paddle.py` only when OCR is actually needed:

| Inspection result | Use Paddle OCR? |
|---|---|
| User explicitly requests Paddle/OCR | Yes |
| Body pages are scanned images with little/no text | Yes |
| `inspect` shows a low text-layer ratio or many body pages with `text_len` near zero | Yes |
| Extracted text is unreadable garbage, wrong script, or badly corrupted | Yes |
| PDF has readable prose text but cover/photos/tables are images | No; use hybrid text+visual conversion |
| PDF has vector charts/tables with searchable captions/text | No; add visual snapshots for those pages |
| Only the cover page is an image | No; embed the cover locally |

For mixed PDFs, OCR only the scanned page ranges if the project OCR workflow supports it; otherwise use OCR for the source pass, then still run EPUB validation and visual-retention checks.

## Helper Script

Use `skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py` for text-layer PDFs.

```bash
python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py inspect book.pdf

python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py convert book.pdf book.epub \
  --title "Book Title" \
  --author "Author Name" \
  --visual-pages 66,72-73,148,210 \
  --section "Front Matter:1-12" \
  --section "1 Introduction:13-42"

python3 skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py verify book.epub \
  --expected-visual-pages 66,72-73,148,210 \
  --expected-label "Figure 1.1" \
  --expected-label "Table 2.1"
```

If the repository has a better project-specific converter, prefer that. For Paddle/OCR requests or scanned PDFs, inspect `pdf2epub_paddle.py` or the project OCR workflow, but still apply the verification checklist below.

## Visual Retention

Use visual snapshots for:

| Content | EPUB handling |
|---|---|
| Cover | Embed as `cover.jpg` and set EPUB cover metadata |
| Photos/scans/diagrams | Local image item in the EPUB |
| Charts/vector figures | Snapshot the PDF page or tight crop |
| Complex tables | Keep table text searchable and add a page snapshot |
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

## Reporting

Report the output path and the evidence, not assumptions: archive check, Calibre parse result, metadata, image count, visual page list, and any known limitations.
