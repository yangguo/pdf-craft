# PDF to EPUB

Convert scanned PDF books to EPUB using PaddleOCR or MinerU cloud APIs — no local GPU required.

## Installation

```bash
pip install pymupdf ebooklib requests python-dotenv markdown numpy
```

## Setup

Copy `.env.example` to `.env` and add your API token(s):

```
PADDLE_API_TOKEN=your_token_here   # for --api paddle (default)
MINERU_API_TOKEN=your_token_here   # for --api mineru
```

Get a PaddleOCR token from [Baidu AIStudio](https://aistudio.baidu.com/) (Layout Parsing API).
Get a MinerU token from [MinerU](https://mineru.net/) (v4 API).

## Usage

```bash
# PaddleOCR (default)
python pdf2epub_paddle.py book.pdf --title "Book Title" --author "Author" --auto-toc

# MinerU (faster, 20-page chunks)
python pdf2epub_paddle.py book.pdf --title "Book Title" --author "Author" --auto-toc --api mineru

# Basic — prompts for title and author
python pdf2epub_paddle.py book.pdf

# Single-chapter EPUB (no chapter splitting)
python pdf2epub_paddle.py book.pdf --title "Title" --no-toc

# Fail on OCR noise artifacts
python pdf2epub_paddle.py book.pdf --title "Title" --strict-ocr-noise

# Review suspicious short CJK OCR spans without failing the build
pdf2epub-ocr-review book.epub --limit 50 --min-score 0.68

# Re-run MinerU for selected 1-based PDF pages.
# By default, only suspicious OCR spans are patched back into the checkpoint.
pdf2epub-mineru-rerun book.pdf --work-dir paddle_epub_work_<hash> --pages 71,75-79,82
python pdf2epub_paddle.py book.pdf --title "Book Title" --author "Author" --auto-toc --api mineru

# Replace an entire page only when the whole page OCR is unusable
pdf2epub-mineru-rerun book.pdf --work-dir paddle_epub_work_<hash> --pages 82 --replace-page
```

### Options

| Flag | Description |
|---|---|
| `--output`, `-o` | Output EPUB path (default: `<input>.epub`) |
| `--title` | Book title (skips interactive prompt) |
| `--author` | Author name |
| `--language` | EPUB language tag (default: `zh-Hant`) |
| `--api` | OCR backend: `paddle` (default) or `mineru` |
| `--auto-toc` | Skip TOC review, use auto-detected chapters |
| `--no-toc` | No chapter splitting — single-chapter EPUB |
| `--strict-ocr-noise` | Fail if OCR artifacts remain in output |
| `--cover-max-edge` | Max cover image edge in pixels (default: 2000) |
| `--cover-quality` | JPEG cover quality 1–100 (default: 82) |

### MinerU Configuration

| Env Var | Default | Description |
|---|---|---|
| `MINERU_API_TOKEN` | — | MinerU v4 API Bearer token |
| `MINERU_CHUNK_SIZE` | `20` | Pages per chunk (max 200) |
| `MINERU_POLL_INTERVAL` | `5` | Seconds between batch result polls |
| `MINERU_MAX_POLL_TIME` | `1800` | Max seconds to wait for a batch |
| `MINERU_LANGUAGE` | `chinese_cht` | MinerU OCR language pack; use `ch_server` for handwritten-heavy content |
| `MINERU_VERIFY_SSL` | `1` | Set to `0` to disable SSL verification |

### Checkpointing

Progress is saved per chunk in `paddle_epub_work_<hash>/`. Re-run the same command to resume after interruption.
`pdf2epub-mineru-rerun` updates the existing checkpoint JSON for the selected pages only; it does not re-upload completed chunks. By default it patches only suspicious OCR spans from the MinerU rerun, preserving the rest of the original page text. Use `--replace-page` only when you want a full-page replacement.

## Features

- **Layout parsing**: Uses PaddleOCR API to detect paragraphs, headings, images, and tables
- **Chapter detection**: Auto-detects chapter headings from OCR output with interactive review
- **Cover extraction**: Renders first PDF page as EPUB cover image
- **Image embedding**: Preserves figures, photos, and illustrations from the original PDF
- **Footnote handling**: Extracts and links OCR-detected footnotes
- **OCR noise cleanup**: Filters out common PaddleOCR artifacts (numeric-only tables, false LaTeX)
- **OCR review scan**: Ranks suspicious short CJK spans for manual checking or targeted second OCR
- **Targeted MinerU reruns**: Re-OCR selected PDF pages and patch their existing checkpoints
- **TOC retargeting**: Fixes TOC fragment links to point at the correct chapter headings

## License

MIT
