# Bird OCR EPUB Pipeline Design

**Goal:** Convert 165 page images in `bird_pages/` into OCR text, chapter-merged Markdown with embedded images, and a polished image-heavy EPUB using PaddleOCR MCP.

**Architecture:** A single orchestrator script drives three phases: OCR each page image with PaddleOCR MCP, build chapter-merged Markdown that embeds every page image, then generate an EPUB with TOC entries per detected chapter and all images embedded. The process is resumable via `bird_ocr_results.json` and deterministic via a stable page order.

**Tech Stack:** Python, PaddleOCR MCP (OCR), `epub_generator` (EPUB), JSON and Markdown artifacts.

## Data Flow
1. `bird_ocr_skeleton.json` (page list + image paths)
2. PaddleOCR MCP per page image
3. `bird_ocr_results.json` (page, file, text)
4. `bird.md` (chapter headings + image blocks + OCR text)
5. `bird.epub` (TOC by chapter, cover set to page 1)

## Components
- **OCR runner:** Reads skeleton, calls PaddleOCR MCP per `bird_pages/page_###.png`, writes incremental results, and skips pages that already have non-empty text in `bird_ocr_results.json`.
- **Chapterizer:** Reuses heading heuristics from existing bird scripts: start new chapter on `# ` headings, treat leading `目次`/`はじめに` as section boundaries. Pages without headings append to the current chapter.
- **Markdown builder:** Writes `bird.md` with chapter headings and per-page blocks:
  - image block `![](bird_pages/page_###.png)`
  - OCR text paragraphs
- **EPUB builder:** Uses `epub_generator` to build TOC entries per chapter, embed all images, and apply CSS for readable typography and image scaling. Cover image set to `bird_pages/page_001.png`.

## Error Handling
- OCR errors on a page are logged; the page is still included with its image so content is preserved.
- Empty OCR text is allowed; the image remains as the source of truth.
- Malformed or missing `bird_ocr_results.json` fails fast with a clear message.

## Outputs
- `bird_ocr_results.json` (165 pages of OCR)
- `bird.md` (chapter-merged markdown with embedded images)
- `bird.epub` (image-heavy EPUB with TOC by chapter)

## Testing
- Optional smoke checks: parse skeleton count, verify chapterizer output on a small synthetic set, and confirm EPUB contains at least one image asset.

## Success Criteria
- All 165 images appear in the EPUB in correct order.
- OCR text is present and merged into chapters by heading heuristics.
- EPUB opens without errors in common readers (Apple Books, Calibre).
