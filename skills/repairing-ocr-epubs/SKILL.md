---
name: repairing-ocr-epubs
description: Use when reviewing or repairing generated EPUB files with OCR problems: garbled Chinese/CJK text, missing sentences, broken chapters or TOC, cross-page omissions, figure-adjacent text loss, or PaddleOCR/MinerU recognition disagreements.
---

# Repairing OCR EPUBs

## Operating Rule

Do not patch prose blindly. Every text change needs evidence from the PDF rendering, an OCR checkpoint, a MinerU rerun, or the existing EPUB context. Prefer targeted MinerU sentence-level patching for PaddleOCR recognition errors; reserve manual EPUB edits for structure, TOC, or cases where OCR evidence is explicit and local.

## Workflow

1. Baseline scan:

```bash
python3 -m paddle_pipeline.ocr_review book.epub --limit 80 --min-score 0.68
python3 auto_fix_garbled.py book.pdf --output book.epub --scan-only --json-report repair_scan.json
```

2. Classify the failure:

| Symptom | First action |
|---|---|
| Short unreadable CJK spans | Run `auto_fix_garbled.py --dry-run` to map EPUB spans to PDF pages |
| Paddle text is bad but page exists in checkpoint | Run targeted MinerU rerun through `auto_fix_garbled.py` |
| Whole OCR page is unusable | Use `pdf2epub-mineru-rerun ... --replace-page` only for that page |
| Missing sentence at a page boundary | Render both adjacent PDF pages, inspect checkpoints, then rerun MinerU for both pages |
| Text near images/tables is missing or reordered | Inspect page image and OCR layout blocks before editing |
| TOC/chapter break is wrong | Inspect `nav.xhtml`, `toc.ncx`, OPF spine, and target XHTML anchors |

3. MinerU repair path:

```bash
python3 auto_fix_garbled.py book.pdf \
  --output book.epub \
  --title "Book Title" \
  --author "Author Name" \
  --dry-run

python3 auto_fix_garbled.py book.pdf \
  --output book.epub \
  --title "Book Title" \
  --author "Author Name"
```

`auto_fix_garbled.py` maps suspicious EPUB spans to 1-based PDF pages, reruns MinerU only for those pages, applies sentence-level checkpoint patches, rebuilds the EPUB, and runs package validation.

## Manual Patch Rules

- Patch the EPUB only after locating the affected XHTML file and source sentence.
- Preserve EPUB ZIP rules: `mimetype` must remain first and uncompressed.
- Keep `nav.xhtml`, `toc.ncx`, OPF manifest, and spine consistent after structure edits.
- For cross-page cases, verify both the end of the previous XHTML and the start of the next XHTML.
- For figure-adjacent text, compare the rendered PDF crop with OCR blocks; do not infer prose from context alone.

## Required Verification

Run these before saying the repair is done:

```bash
python3 auto_fix_garbled.py book.pdf --output book.epub --scan-only
python3 -m zipfile -t book.epub
python3 - <<'PY'
from ebooklib import epub
for path in ["book.epub"]:
    book = epub.read_epub(path)
    print(path, len(list(book.get_items())))
PY
```

Confirm candidate count is zero or every remaining candidate has been inspected and accepted as valid text. Report the exact files delivered and the verification evidence.
