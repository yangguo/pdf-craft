# Hybrid EPUB Quality Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generalize the reliable visual selection, CJK paragraph reflow, metadata, and package checks learned from converting `duoyuan.pdf` without adding book-specific rules.

**Architecture:** Keep the existing hybrid converter CLI and add small pure helpers around its current extraction and verification seams. Visual selection will calibrate drawing complexity against the document median, CJK reflow will use block geometry only when the page is predominantly CJK, and verification will inspect EPUB ZIP/XML/local-reference invariants before reporting success.

**Tech Stack:** Python 3, PyMuPDF, EbookLib, Pillow, `zipfile`, `xml.etree.ElementTree`, pytest.

### Task 1: Calibrate visual-page detection

**Files:**
- Modify: `skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py`
- Create: `tests/test_pdf_to_epub_hybrid.py`

**Steps:**
1. Add a failing regression test showing recurring decorative drawings are ignored while a drawing outlier and an embedded-image page are retained.
2. Run the focused test and confirm it fails against the fixed `>= 5` drawing threshold.
3. Implement a median-derived drawing threshold and bilingual figure/table caption recognition.
4. Run the focused test and the existing suite.

### Task 2: Reflow CJK line blocks into paragraphs

**Files:**
- Modify: `skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py`
- Modify: `tests/test_pdf_to_epub_hybrid.py`

**Steps:**
1. Add failing tests for joining unindented continuation lines without inserting spaces between CJK characters and starting a new paragraph at a first-line indent.
2. Run the tests and confirm the current line-block behavior fails.
3. Add a geometry-aware CJK-only reflow helper and integrate it into `page_text_blocks` without changing non-CJK extraction.
4. Run focused and full tests.

### Task 3: Enrich metadata and package verification

**Files:**
- Modify: `skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py`
- Modify: `tests/test_pdf_to_epub_hybrid.py`
- Modify: `skills/converting-pdfs-to-epub/SKILL.md`

**Steps:**
1. Add failing tests for publisher/date/description/ISBN metadata options, EPUB mimetype ordering/compression, XML parsing, and missing local resource detection.
2. Run the tests and confirm the options/checks are absent.
3. Add the CLI options and standard-library verifier checks, and verify generated XHTML language attributes use the selected language.
4. Update the skill examples and verification description.
5. Run focused tests, all pytest tests, Pyright, `git diff --check`, and a real sample conversion/verification.
