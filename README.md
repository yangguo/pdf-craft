# PDF to EPUB

Convert scanned PDF books to EPUB using PaddleOCR or MinerU cloud APIs — no local GPU required.

## Installation

```bash
pip install pymupdf ebooklib requests python-dotenv markdown
```

## Setup

Copy `.env.example` to `.env` and add your API token(s):

```
PADDLE_API_TOKEN=your_token_here   # for --api paddle (default)
MINERU_API_TOKEN=your_token_here   # for --api mineru
MINERU_LANGUAGE=ch_tra            # traditional Chinese OCR
MINERU_ENABLE_TABLE=0             # avoid vertical text being treated as tables
PADDLE_LAYOUT_THRESHOLD=0.35      # better retain narrow vertical text blocks
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
| `MINERU_LANGUAGE` | `ch_tra` | OCR language for traditional Chinese books |
| `MINERU_ENABLE_TABLE` | `0` | Set to `1` only when table extraction is needed |
| `MINERU_VERIFY_SSL` | `1` | Set to `0` to disable SSL verification |

### PaddleOCR Tuning

| Env Var | Default | Description |
|---|---|---|
| `PADDLE_LAYOUT_THRESHOLD` | `0.35` | Lower threshold to keep narrow vertical text blocks |
| `PADDLE_TEMPERATURE` | `0.1` | Lower decoding randomness for traditional text |
| `PADDLE_REPETITION_PENALTY` | `1.05` | Preserve legitimate repeated classical Chinese characters |
| `PADDLE_TOP_P` | `0.75` | Narrow decoding candidates to reduce garbled output |

### Checkpointing

Progress is saved per chunk in `paddle_epub_work_<hash>/`. Re-run the same command to resume after interruption.

## Features

- **Layout parsing**: Uses PaddleOCR API to detect paragraphs, headings, images, and tables
- **Chapter detection**: Auto-detects chapter headings from OCR output with interactive review
- **Cover extraction**: Renders first PDF page as EPUB cover image
- **Image embedding**: Preserves figures, photos, and illustrations from the original PDF
- **Footnote handling**: Extracts and links OCR-detected footnotes
- **OCR noise cleanup**: Filters out common PaddleOCR artifacts (numeric-only tables, false LaTeX)
- **TOC retargeting**: Fixes TOC fragment links to point at the correct chapter headings

## License

MIT
