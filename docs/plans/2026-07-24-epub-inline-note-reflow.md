# EPUB Inline Note Reflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore continuous chapter prose by converting falsely block-level `[number]` note markers into inline superscripts and joining only the paragraph continuations they split.

**Architecture:** Extend the existing standard-library EPUB note postprocessor. First collect notes and narrowly identified page-bottom fragments at each chapter end. Then convert direct `blockquote` children whose complete text starts with `[number]`, plus direct paragraphs beginning with the same marker, to inline superscripts. Preserve genuine quotations, append block markers to the preceding paragraph when available, and join the following paragraph only when the marker block contains an unfinished text fragment. Handle zero-height page anchors without losing source-page IDs. For the rare PDF reading-order ambiguity that cannot be inferred safely, accept an explicit audited JSON correction manifest before using the existing atomic EPUB repackaging path.

**Tech Stack:** Python 3.11, `zipfile`, `xml.etree.ElementTree`, `pytest`, Pyright, existing EPUB verifier and Calibre CLI.

### Task 1: Repair ordinary inline marker splits

**Files:**
- Modify: `tests/test_relocate_epub_notes.py`
- Modify: `skills/converting-pdfs-to-epub/scripts/relocate_epub_notes.py`

1. Add a failing XHTML test reproducing `[35] 可見即` between two paragraphs.
2. Assert one continuous paragraph containing an inline `sup.inline-note-marker`, with CJK fragments joined without spaces.
3. Implement the minimal direct-body transformation.
4. Run the focused test to green.
5. Add a test proving genuine blockquotes remain unchanged.

### Task 2: Preserve real paragraph and page boundaries

**Files:**
- Modify: `tests/test_relocate_epub_notes.py`
- Modify: `skills/converting-pdfs-to-epub/scripts/relocate_epub_notes.py`

1. Add failing cases for a marker-only citation, a sentence-ending marker fragment, and page anchors before/after markers.
2. Keep the next paragraph separate for marker-only and sentence-ending cases.
3. Inline page anchors only when joining text across the PDF boundary, retaining their IDs.
4. Add a safe fallback that retags an orphan marker block as a paragraph rather than deleting or guessing surrounding prose.
5. Run all focused tests to green.

### Task 3: Integrate into EPUB repackaging

**Files:**
- Modify: `tests/test_relocate_epub_notes.py`
- Modify: `skills/converting-pdfs-to-epub/scripts/relocate_epub_notes.py`

1. Add a failing archive-level test for marker repair counts and CSS.
2. Relocate chapter notes before marker repair so page anchors become adjacent to their prose.
3. Report note, marker, paragraph-join, fragment, and reading-order correction counts.
4. Append inline-marker CSS independently of the existing chapter-notes CSS.
5. Verify unrelated archive members, metadata, images, ZIP ordering, and permissions remain unchanged.

### Task 4: Repair and validate the real book

**Files:**
- Input: `/Users/vyang/Desktop/spaces/pdf-craft/duoyuan.epub`
- Output: `/Users/vyang/Desktop/spaces/pdf-craft/duoyuan-reflowed.epub`

1. Generate a distinct output without overwriting prior artifacts.
2. Confirm all 779 direct numbered marker blockquotes are removed and genuine blockquotes remain.
3. Check the reported example and representative marker-only, sentence-ending, cross-page, and true-quotation cases.
4. Compare normalized text before and after after ignoring presentation-only marker brackets.
5. Run the full repository tests, Pyright, ZIP/XML/reference validation, `ebook-meta`, and `ebook-convert`.
6. Review the diff, commit, fast-forward `main`, verify again, and remove the temporary worktree.

### Task 5: Resolve verified page-bottom and reading-order exceptions

**Files:**
- Modify: `tests/test_relocate_epub_notes.py`
- Modify: `skills/converting-pdfs-to-epub/scripts/relocate_epub_notes.py`
- Add: `docs/plans/duoyuan-reading-order-corrections.json`

1. Move only note-like fragments interposed between an unfinished numbered marker and a page anchor to chapter endnotes.
2. Preserve fragment content as valid XHTML without nested paragraphs.
3. Style direct body paragraphs beginning with `[number]` as superscript markers without merging them.
4. Add an explicit correction format for PDF-verified displaced prefixes, and fail if any declared correction does not match.
5. Verify the two corrected sequences against PDF pages 185 and 297.
