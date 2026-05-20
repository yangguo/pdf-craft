#!/usr/bin/env python3
"""PaddleOCR PDF to EPUB converter — thin wrapper.

Usage:
    python pdf2epub_paddle.py input.pdf [--output output.epub] [--title "..."] [--author "..."]
"""
from paddle_pipeline.main import main

if __name__ == "__main__":
    main()
