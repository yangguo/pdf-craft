# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaddleOCR-based pipeline for converting scanned PDFs to EPUB. Uses the PaddleOCR cloud API for layout parsing and OCR, then assembles results into a structured EPUB with auto-detected chapters, footnotes, and embedded images.

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

# Convert a PDF to EPUB
python3 pdf2epub_paddle.py input.pdf --title "Book Title" --author "Author" --auto-toc --language zh-Hant

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
ocr_noise.py   footnotes.py   paddle_api.py   metadata.py
  ↑              ↑               ↑                ↑
  └──────────────┴───────────────┴────────────────┘
                        ↑
                  epub_builder.py  ← create_epub (assembly)
                        ↑
                      main.py      ← CLI entry point
```

### Key design decisions
- **Private-by-default**: All functions are prefixed `_` unless used across modules or by tests. Public API re-exports are in `__init__.py`.
- **Wrapper at root**: `pdf2epub_paddle.py` is a thin 10-line shim calling `paddle_pipeline.main.main()` — backward compat with the original usage.
- **Test compatibility**: Tests import from `paddle_pipeline` as a package (not `importlib` loading a raw script). Mock targets use `mod.submodule.function` paths matching the new module structure.
- **Checkpoint/resume**: API results are saved as JSON in `paddle_epub_work_<hash>/` per chunk. Re-running the same PDF resumes from the last checkpoint.

### EPUB post-processing pipeline (in `ensure_toc_targets_start_pages`)
1. Clean garbled headings → 2. Remove bibliography titles → 3. Retarget numbered TOC links → 4. Collect TOC fragment targets → 5. Mark targets with CSS page-break class → 6. Remove next-chapter heading previews from previous XHTML files.
