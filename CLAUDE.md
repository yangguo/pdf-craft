# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaddleOCR or MinerU cloud-API pipeline for converting scanned PDFs to EPUB. Uses the PaddleOCR or MinerU v4 cloud API for layout parsing and OCR, then assembles results into a structured EPUB with auto-detected chapters, footnotes, and embedded images.

## Commands

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_pdf2epub_paddle_ocr_cleanup.py -v

# Run a single test method
python3 -m pytest tests/test_pdf2epub_paddle_ocr_cleanup.py::TestPdf2EpubPaddleOcrCleanup::test_clean_ocr_noise -v

# Type checking (requires pip install pyright)
python3 -m pyright paddle_pipeline/

# Convert a PDF to EPUB (PaddleOCR, default)
python3 pdf2epub_paddle.py input.pdf --title "Book Title" --author "Author" --auto-toc --language zh-Hant

# Convert a PDF to EPUB (MinerU)
python3 pdf2epub_paddle.py input.pdf --title "Book Title" --author "Author" --auto-toc --api mineru

# Skip interactive prompts (non-interactive mode)
python3 pdf2epub_paddle.py input.pdf --title "Title" --author "Author" --auto-toc

# Disable TOC detection (single-chapter EPUB)
python3 pdf2epub_paddle.py input.pdf --title "Title" --no-toc

# Strict OCR noise validation
python3 pdf2epub_paddle.py input.pdf --title "Title" --strict-ocr-noise
```

## Architecture

### Runtime dependency pattern
All third-party imports (`requests`, `fitz`/PyMuPDF, `ebooklib`) are optional — imported in a `try/except` in `paddle_pipeline/config.py` and set to `None` on failure. `check_dependencies()` in `paddle_api.py` provides actionable install instructions before the pipeline runs.

### Module dependency graph (no cycles)

```
config.py                  ← constants, regexes, env vars (leaf module)
  ↑
epub_href.py              ← EPUB href resolution, XHTML element parsing
  ↑
heading_cleanup.py        ← garbled heading detection, bibliography removal
  ↑
toc_retarget.py           ← TOC retargeting, page-break CSS injection
  ↑
epub_validate.py          ← scan/validate EPUB, write_validated_epub
  ↑
ocr_noise.py   footnotes.py   paddle_api.py   mineru_api.py   metadata.py
  ↑              ↑               ↑                ↑                ↑
  └──────────────┴───────────────┴────────────────┴────────────────┘
                        ↑
                  epub_builder.py  ← create_epub (assembly)
                        ↑
                      main.py      ← CLI entry point
```

### Key design decisions
- **Private-by-default**: All functions are prefixed `_` unless used across modules or by tests. Public API re-exports are in `__init__.py`.
- **Wrapper at root**: `pdf2epub_paddle.py` is a thin 10-line shim calling `paddle_pipeline.main.main()` — backward compat with the original usage.
- **Test compatibility**: Tests import from `paddle_pipeline` as a package (not `importlib` loading a raw script). Mock targets use `mod.submodule.function` paths matching the new module structure.
- **Checkpoint/resume**: API results are saved as JSON in `paddle_epub_work_<hash>/` per chunk. Re-running the same PDF resumes from the last checkpoint. MinerU checkpoints stash `_mineru_zip_url` for re-parse without re-uploading when parsing logic changes.

### EPUB post-processing pipeline (in `ensure_toc_targets_start_pages`)
1. Clean garbled headings → 2. Remove bibliography titles → 3. Retarget numbered TOC links → 4. Collect TOC fragment targets → 5. Mark targets with CSS page-break class → 6. Remove next-chapter heading previews from previous XHTML files.

## Garbled CJK Detection & Repair Workflow

Three tiers of garbled OCR text, in order of detectability:

**Tier 1 — Random rare-character garble** (detected by `ocr_review.py`):
- Example: `居之者忽音但一通乃一祔一筆八`
- Characters are rare, bigrams are mostly singletons → `_score_fragment()` catches via singleton ratio + density bonus.

**Tier 2 — Repeated character garble** (detected by `ocr_noise.py`):
- Example: `三三三三三三...` (4048 repetitions)
- `_repetition_ratio()` in `ocr_review.py` and `_find_repeated_char_spans()` in `ocr_noise.py`.

**Tier 3 — Semantic garble** (needs LLM or source-text comparison):
- Example: `誰誰歸白一工任合室什其林人然已將則一語至不遂金方是任見八百仁素今月新用至不`
- Characters are common, bigrams look statistically plausible, but sentence is nonsense.
- `semantic_ocr_review.py` generates candidates → LLM filters via `--llm`.
- **Variant**: Common-bigram semantic garble scores too low for statistical detection (e.g. `安事變凸顯出壽、罰之罰均固人關系…` scored 0.091). Use `--deep` to add sentence-level chunks that bypass scoring.

### Repair workflow

```
1. Run semantic scan with LLM:
   python3 -m paddle_pipeline.semantic_ocr_review book.epub --llm --json review.json

   # Deep scan: adds sentence-level chunks (catches Tier-3 garble with common bigrams)
   python3 -m paddle_pipeline.semantic_ocr_review book.epub --llm --deep --only-garbled --json deep.json
   
   # With --llm, min_score auto-defaults to 0.30 (vs 0.50 without LLM)

2. For each confirmed garble, find correct text using MinerU checkpoints:
   - MinerU rebuild generates complete OCR checkpoints in paddle_epub_work_*/
   - Search chunk_*.json for context keywords to find correct text
   - This is faster than one-by-one page OCR

3. Apply fix: extract ZIP → patch XHTML → re-zip EPUB
   - Use raw XHTML matching (account for \r\n newlines)
   - Verify old text gone and new text present
   - Standard detectors: ocr_review.py (score>=0.55) + repeated  CJK

4. Verify with standard scan:
   python3 -m paddle_pipeline.ocr_review book.epub
```

### NEVER do

- Do NOT guess correct text from context — always use source OCR
- Do NOT delete MinerU/PaddleOCR checkpoint directories — they're the key repair resource
- Do NOT rebuild EPUB from scratch to fix garble — produces more new issues than it solves
- Do NOT apply LLM-suggested text directly — LLM only flags candidates, source OCR confirms
