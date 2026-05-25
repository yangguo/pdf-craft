#!/usr/bin/env python3
"""PDF to EPUB converter — supports PaddleOCR and MinerU backends.

Usage:
    python pdf2epub_paddle.py input.pdf [--output output.epub] [--title "..."] [--author "..."] [--api mineru]
"""
from paddle_pipeline.main import main

if __name__ == "__main__":
    main()
