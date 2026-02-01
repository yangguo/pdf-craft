# PDF Craft - Copilot Instructions

PDF Craft converts PDF files (especially scanned books) into Markdown or EPUB using DeepSeek OCR for local document recognition.

## Build & Test Commands

```bash
# Install dependencies
poetry install --with dev

# Run all tests
poetry run python test.py

# Run a specific test file
poetry run python test.py test_parser

# Type checking
poetry run pyright pdf_craft tests

# Linting
poetry run pylint pdf_craft tests
```

## Architecture

### Core Processing Pipeline

The conversion flow is: **PDF → OCR → Pages → Chapters → Output (Markdown/EPUB)**

1. **PDF Handler** (`pdf/handler.py`): Renders PDF pages to images using Poppler via `pdf2image`
2. **OCR** (`pdf/ocr.py`): Uses DeepSeek OCR models to extract text, tables, and formulas from page images
3. **Sequence Processing** (`sequence/`): Analyzes OCR output to detect document structure, joins content across pages, and groups into chapters
4. **TOC Analysis** (`toc/`): Extracts or generates table of contents from recognized pages
5. **Rendering**: Converts processed chapters to Markdown (`markdown/`) or EPUB (`epub/`) using `epub-generator`

### Key Classes

- `Transform` (`transform.py`): Main orchestrator exposing `transform_markdown()` and `transform_epub()`
- `OCR` (`pdf/ocr.py`): Handles model loading and page recognition
- `PDFHandler` protocol (`pdf/handler.py`): Abstraction for PDF parsing (default uses Poppler)

### Module Structure

- `pdf_craft/common/`: Shared utilities (XML handling, folder management, statistics)
- `pdf_craft/sequence/`: Chapter generation, content joining, level analysis
- `pdf_craft/expression.py`: LaTeX expression handling
- `pdf_craft/llm/`: Optional LLM integration for TOC extraction

## Code Conventions

### Type Annotations

All public APIs use Python type hints. The codebase targets Python 3.11+ and uses `PathLike | str` for path parameters.

### Error Handling Pattern

Errors follow a two-tier pattern:
- `PDFError` and `OCRError` are inline errors that can be optionally ignored via `ignore_pdf_errors` / `ignore_ocr_errors` callbacks
- `InterruptedError` wraps aborted operations and includes token metering

### Configuration Style

Functions accept many optional parameters with sensible defaults. The pattern is to have a simple call signature with extensive keyword arguments:

```python
transform_markdown(
    pdf_path="input.pdf",
    markdown_path="output.md",
    # ... many optional params
)
```

### Lazy Loading

The `doc-page-extractor` dependency uses lazy loading - don't import its types at module level outside of `pdf/` module.

### Testing

Tests use `unittest` (not pytest runner). Test files are in `tests/` and follow the `test_*.py` naming pattern.
