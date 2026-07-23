# EPUB Chapter-End Notes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move interleaved EPUB footnote blocks to the end of their existing chapter while preserving prose order, note contents, links, metadata, and package validity.

**Architecture:** Add a standard-library EPUB postprocessor that rewrites XHTML body children marked as `footnotes`, collecting them into one `endnotes` section at the end of each chapter. Repackage the EPUB without changing unrelated members, enforce the required uncompressed first `mimetype` entry, and add a small CSS rule for the chapter-end notes section.

**Tech Stack:** Python 3.11, `zipfile`, `xml.etree.ElementTree`, `pytest`, existing EPUB verifier and Calibre CLI.

### Task 1: Define XHTML note relocation

**Files:**
- Create: `tests/test_relocate_epub_notes.py`
- Create: `skills/converting-pdfs-to-epub/scripts/relocate_epub_notes.py`

1. Write a failing test containing prose, two interleaved `aside.footnotes` elements, note IDs, and links.
2. Run the focused test and confirm it fails because the relocation API is absent.
3. Implement the minimal XML transformation:
   - leave non-note body children in their original order;
   - remove direct chapter footnote asides;
   - append one `section.chapter-notes[epub:type=endnotes]`;
   - preserve each aside and its descendants unchanged.
4. Run the focused test and confirm it passes.
5. Add and pass tests for no-op XHTML and idempotent input.

### Task 2: Repackage EPUB safely

**Files:**
- Modify: `tests/test_relocate_epub_notes.py`
- Modify: `skills/converting-pdfs-to-epub/scripts/relocate_epub_notes.py`

1. Write a failing test for a miniature EPUB with two XHTML chapters, CSS, metadata, an image, and a valid `mimetype`.
2. Implement archive transformation using a temporary output file:
   - rewrite only XHTML members containing direct footnote asides;
   - append the chapter-notes CSS once;
   - copy all other bytes and ZIP metadata;
   - write `mimetype` first and without compression;
   - reject identical input and output paths.
3. Assert counts for changed chapters and moved note groups.
4. Verify the output parses as XML and unrelated binary members are byte-identical.

### Task 3: Adjust and verify `duoyuan.epub`

**Files:**
- Output: `/Users/vyang/Desktop/spaces/pdf-craft/duoyuan-chapter-end-notes.epub`

1. Run the postprocessor against the existing `duoyuan.epub` without overwriting it.
2. Confirm every changed chapter has no direct interleaved footnote aside and ends with one endnotes section.
3. Compare non-note prose order before and after.
4. Run the full test suite.
5. Run `zipfile -t`, the project EPUB verifier, `ebook-meta`, and `ebook-convert`.
6. Inspect extracted text around representative chapter boundaries and report any limitations.
