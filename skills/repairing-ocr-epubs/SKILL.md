---
name: repairing-ocr-epubs
description: "Use when reviewing or repairing generated EPUB files with OCR problems: garbled Chinese/CJK text, missing sentences, broken chapters or TOC, cross-page omissions, figure-adjacent text loss, or PaddleOCR/MinerU recognition disagreements."
---

# Repairing OCR EPUBs

## Operating Rule

Do not patch prose blindly. Every text change needs evidence from the PDF rendering, an OCR checkpoint, a MinerU rerun, or the existing EPUB context. Prefer targeted MinerU sentence-level patching for PaddleOCR recognition errors; reserve manual EPUB edits for structure, TOC, or cases where OCR evidence is explicit and local.

## Workflow

1. Baseline scan:

```bash
python3 -m paddle_pipeline.ocr_review book.epub --limit 80 --min-score 0.68
python3 auto_fix_garbled.py book.pdf --output book.epub --scan-only \
  --scan-boundaries --json-report repair_scan.json
```

2. Classify the failure:

| Symptom | First action |
|---|---|
| Short unreadable CJK spans | Run `auto_fix_garbled.py --dry-run` to map EPUB spans to PDF pages |
| Paddle text is bad but page exists in checkpoint | Run targeted MinerU rerun through `auto_fix_garbled.py` |
| Whole OCR page is unusable | Use `pdf2epub-mineru-rerun ... --replace-page` only for that page |
| Missing sentence at a page boundary | Run `--scan-boundaries`, then render both adjacent PDF pages and compare checkpoint page starts |
| Text near images/tables is missing or reordered | Inspect page image and OCR layout blocks; image-first pages can hide text before/after figures |
| TOC/chapter break is wrong | Inspect `nav.xhtml`, `toc.ncx`, OPF spine, target XHTML anchors, then rebuild from checkpoint with a manual TOC JSON |

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

## Cross-Page and Page-Start Omissions

Use the checkpoint boundary scanner before manual reading:

```bash
python3 -m paddle_pipeline.page_boundary_review paddle_epub_work_xxxxxxxx \
  --epub book.epub --min-score 0.70 --limit 80 --json > boundary_candidates.json
```

Treat this as high-recall triage, not proof. `epub_status` means:

| Status | Meaning |
|---|---|
| `checkpoint_boundary_present` | EPUB contains the checkpoint page-tail/page-head join; PDF page-start OCR can still be wrong |
| `checkpoint_boundary_missing_from_epub` | EPUB does not contain the checkpoint join; inspect XHTML assembly, chapter splits, and checkpoints |
| `epub_unavailable` / `insufficient_boundary_text` | The EPUB could not be checked or the candidate is too short for automatic matching |

A missing-page-start candidate is confirmed only when PDF rendering, a second OCR pass, or MinerU shows text that is absent from the checkpoint/EPUB. For vertical Traditional Chinese pages, render the next page and use Tesseract as a quick second pass:

```bash
pdftoppm -f PAGE -l PAGE -png -r 220 book.pdf tmp/page
tesseract tmp/page-PAGE.png stdout -l chi_tra_vert --psm 5
```

If the checkpoint page head appears later in the second OCR output, the page start probably lost text. Check image-adjacent candidates even when Tesseract cannot align, because figure pages often start with `<img>` before the missing prose.

For confirmed page-start omissions, prefer MinerU page-start patching over whole-page replacement:

```bash
python3 -m paddle_pipeline.mineru_rerun book.pdf \
  --work-dir paddle_epub_work_xxxxxxxx \
  --pages 30,34,40 \
  --chunk-size 5 \
  --patch-page-start
```

This inserts only the MinerU-detected prefix before the existing checkpoint page text. If MinerU also misses the rightmost column, use the rendered PDF image plus Tesseract text as evidence and make a narrow manual checkpoint patch.

## Manual TOC Rebuild

When automatic heading detection promotes body sentences into chapter titles, do not patch only `nav.xhtml`. Rebuild the EPUB from the repaired checkpoint with an explicit TOC:

```json
{
  "headings": [
    {"page": 1, "title": "封面與目次", "page_start": true},
    {"page": 9, "title": "第七章 雅爾達、東北和戰後戰略", "match": "雅爾達、東北和戰後戰略"},
    {"page": 291, "title": "第十三章 尼克森和晚年歲月", "aliases": ["尼克森及晚年"]}
  ]
}
```

Use:

```bash
python3 -m paddle_pipeline.rebuild_from_checkpoint paddle_epub_work_xxxxxxxx \
  --output book.epub \
  --title "Book Title" \
  --author "Author" \
  --toc manual_toc/book.json
```

Manual TOC fields:

| Field | Use |
|---|---|
| `page` | 1-based PDF/checkpoint page |
| `title` | Correct display title used in nav, NCX, spine item title, and chapter heading |
| `match` / `matches` | OCR line(s) that should trigger this split when the source heading is wrong or incomplete |
| `aliases` | Wrong or alternate OCR subtitle lines to remove as duplicate headings, not split triggers |
| `page_start` | Insert a chapter/part/notes heading at the beginning of the page without deleting page text |

After rebuilding, compare `nav.xhtml`, `toc.ncx`, and OPF spine labels against the manual TOC and spot-check chapter XHTML starts. This catches cases where the TOC label is correct but the content still starts at the wrong OCR line.

## Manual Patch Rules

- Patch the EPUB only after locating the affected XHTML file and source sentence.
- Preserve EPUB ZIP rules: `mimetype` must remain first and uncompressed.
- Keep `nav.xhtml`, `toc.ncx`, OPF manifest, and spine consistent after structure edits.
- For cross-page cases, verify both the end of the previous XHTML and the start of the next XHTML.
- For figure-adjacent text, compare the rendered PDF crop with OCR blocks; do not infer prose from context alone.
- When direct evidence supports a manual prose patch, update both the EPUB XHTML and the matching `chunk_*.pdf.json` checkpoint so future rebuilds do not reintroduce the defect.

## Required Verification

Run these before saying the repair is done:

```bash
python3 auto_fix_garbled.py book.pdf --output book.epub --scan-only --scan-boundaries
python3 -m zipfile -t book.epub
python3 - <<'PY'
from ebooklib import epub
for path in ["book.epub"]:
    book = epub.read_epub(path)
    print(path, len(list(book.get_items())))
PY
```

Confirm candidate count is zero or every remaining candidate has been inspected and accepted as valid text. Report the exact files delivered and the verification evidence.
