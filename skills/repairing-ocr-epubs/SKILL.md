---
name: repairing-ocr-epubs
description: "Use when reviewing or repairing generated EPUB files with OCR problems: garbled Chinese/CJK text, missing sentences, broken chapters or TOC, cross-page omissions, figure-adjacent text loss, or PaddleOCR/MinerU recognition disagreements."
---

# Repairing OCR EPUBs

## Operating Rule

Do not patch prose blindly. Every text change needs evidence from the PDF rendering, an OCR checkpoint, a MinerU rerun, or the existing EPUB context. Prefer targeted MinerU sentence-level patching for PaddleOCR recognition errors; reserve manual EPUB edits for structure, TOC, or cases where OCR evidence is explicit and local.

## Garbled CJK Text Classification

Three tiers of garbled OCR text, in order of detectability:

| Tier | Type | Example | Detected by |
|------|------|---------|-------------|
| 1 | Random rare-char clusters | `居之者忽音但一通乃一祔一筆八` | `ocr_review.py` (singleton bigram + density bonus) |
| 2 | Single-char repetition | `三三三三三三...` (4048 times) | `ocr_noise.py` (`_find_repeated_char_spans`), `ocr_review.py` (`_repetition_ratio`) |
| 3 | Semantic garble | `誰誰歸白一工任合室什其林人然已將則一語至不遂金方是任見` | `semantic_ocr_review.py` + `--llm` (statistical candidate + LLM filter) |

Tier 3 characters are mostly common, bigrams look statistically plausible, but the sentence is nonsense. The standard detectors score them below threshold because common_char_ratio is high and anchor_bigram_ratio is high.

## Workflow

1. Baseline scan:

```bash
# Standard detection (Tiers 1+2)
python3 -m paddle_pipeline.ocr_review book.epub --limit 80 --min-score 0.68

# Semantic detection (Tier 3 — statistical windows + optional LLM filter)
python3 -m paddle_pipeline.semantic_ocr_review book.epub --llm --json semantic_review.json

# Deep semantic detection (statistical + sentence-level chunks + LLM)
# Use --deep when Tier-3 garble is suspected but not caught by standard scan
python3 -m paddle_pipeline.semantic_ocr_review book.epub --llm --deep --only-garbled --strict-llm --json deep_review.json

# Full auto-fix scan
python3 auto_fix_garbled.py book.pdf --output book.epub --scan-only \
  --scan-boundaries --json-report repair_scan.json
```

2. Classify the failure:

| Symptom | First action |
|---|---|
| Short unreadable CJK spans | Run `auto_fix_garbled.py --dry-run` to map EPUB spans to PDF pages |
| Semantic garble (common chars, nonsense sentence) | Run `semantic_ocr_review.py --llm --deep --strict-llm`, then search MinerU checkpoints and PDF crops for correct text |
| Paddle text is bad but page exists in checkpoint | Run targeted MinerU rerun through `auto_fix_garbled.py` |
| Whole OCR page is unusable | Use `pdf2epub-mineru-rerun ... --replace-page` only for that page |
| Missing sentence at a page boundary | Run `--scan-boundaries`, then render both adjacent PDF pages and compare checkpoint page starts |
| Text near images/tables is missing or reordered | Inspect page image and OCR layout blocks; image-first pages can hide text before/after figures |
| TOC/chapter break is wrong | Inspect `nav.xhtml`, `toc.ncx`, OPF spine, target XHTML anchors, then rebuild from checkpoint with a manual TOC JSON |

3. **Finding correct text via MinerU checkpoints (preferred)**:

The fastest way to get correct text for any garbled passage is to search existing MinerU rebuild checkpoints. If you have run `pdf2epub_paddle.py --api mineru` (even a failed/cancelled run), the `paddle_epub_work_*/chunk_*.json` files contain complete OCR for every PDF page. Search them for context keywords:

```python
import json, glob
for path in sorted(glob.glob("paddle_epub_work_*/chunk_*.json")):
    with open(path) as f: r = json.load(f)
    for page in r.get("result", {}).get("layoutParsingResults", []):
        text = page.get("markdown", {}).get("text", "")
        if "keyword" in text:
            print(f"Found in {path}!")
```

This avoids ad-hoc single-page OCR — one search covers the entire book.

4. **Patching an EPUB XHTML file**:

```python
import zipfile, shutil, tempfile, os
epub = "book.epub"
fname = "EPUB/Chapter.xhtml"
with zipfile.ZipFile(epub) as z: orig = z.read(fname).decode("utf-8")
new = orig.replace(old_garbled, correct_text)

td = tempfile.mkdtemp()
with zipfile.ZipFile(epub) as z: z.extractall(td)
with open(os.path.join(td, fname), "w", encoding="utf-8") as f: f.write(new)
with zipfile.ZipFile(epub, "w", zipfile.ZIP_DEFLATED) as z:
    mt = os.path.join(td, "mimetype")
    if os.path.exists(mt): z.write(mt, "mimetype", compress_type=zipfile.ZIP_STORED)
    for root, _, files in os.walk(td):
        for file in files:
            if file == "mimetype": continue
            z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), td).replace(os.sep, "/"))
shutil.rmtree(td)
```

**Critical**: Use raw XHTML matching (regex with `\r+\n` for newlines), not plain-text matching.

5. MinerU repair path:

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

## LLM-Positive Semantic Garble Loop

LLM review is a filter, not a source of replacement text. For every LLM-positive candidate:

1. Locate the checkpoint page by searching the candidate span in `paddle_epub_work_*/chunk_*.json`.
2. Find the matching `prunedResult.parsing_res_list[*].block_bbox`.
3. Render that PDF page/crop and inspect the printed text; optionally run Tesseract as a second opinion.
4. Patch the checkpoint, not just the EPUB. Update every copy of the bad text, especially both `markdown.text` and `prunedResult.parsing_res_list[*].block_content`.
5. Rebuild from checkpoint and search exact bad strings in both checkpoint JSON and the rebuilt EPUB.
6. Rerun `semantic_ocr_review --llm --deep --only-garbled --strict-llm` until it returns zero LLM-positive candidates, or document any remaining candidate as intentionally accepted with evidence.

Use `--strict-llm` for final verification so network/API failures cannot be mistaken for a clean result. The LLM caller retries transient HTTP/SSL errors, but a strict failure still means the review did not complete.

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
- In checkpoint patches, replace every duplicate OCR text store: `markdown.text` plus matching `prunedResult` block content. A rebuild may use one while later audits read the other.

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

Confirm candidate count is zero or every remaining candidate has been inspected and accepted as valid text. Also run semantic scan to catch Tier 3 issues:

```bash
python3 -m paddle_pipeline.semantic_ocr_review book.epub --limit 30 --min-score 0.60 --json semantic_postfix.json
```

## NEVER Do

- **Never guess correct text from context** — always use source OCR from MinerU checkpoints or PDF rerun
- **Never delete PaddleOCR/MinerU checkpoint directories** — they are the primary repair resource (complete source text for every page)
- **Never rebuild an EPUB from scratch to fix a few garbled passages** — MinerU/Paddle builds produce MORE new garble than they fix
- **Never apply LLM-suggested replacement text directly** — LLM only flags candidates for review; source OCR provides the confirmed text
- **Never use `.bak` restore without re-applying all known fixes** — `.bak` files pre-date intermediate repair sessions
- **Never match EPUB text as plain strings** — raw XHTML has `\r\n` newlines that break exact matching; use regex with `\r+\n`

## Key Repair Patterns (from jiang1/jiang2 experience)

| Pattern | Solution |
|---|---|
| "Paddle text + random rare chars" scribble | Singleton bigram density triggers `ocr_review.py` |
| Single character repeated 100s of times | `_repetition_ratio` ≥ 0.80 fires detection boost |
| Common chars forming nonsense sentence | Semantic scan → LLM filter → checkpoint search |
| Fix lost after `.bak` restore | Batch re-apply all known replacements from memory record |
| XHTML newline mismatch | Match raw `repr()` output, use `\r+\n` in regex |
| EPUB-to-PDF page non-linear mapping | Search mineru checkpoints instead of guessing pages |
