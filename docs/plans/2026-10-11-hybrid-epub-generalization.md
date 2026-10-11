# Hybrid EPUB Generalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Commit reusable text, visual, and validation improvements without book-specific material.

**Architecture:** Keep native PDF extraction authoritative. Replace source-specific assumptions with explicit options. Flatten generated page wrappers into continuous XHTML while retaining page anchors; join across pages only with conservative native-layout evidence and preserve inline elements.

**Tech Stack:** Python, PyMuPDF, EbookLib, ElementTree, pytest.

### Task 1: Remove source-specific extraction assumptions

- Modify `skills/converting-pdfs-to-epub/scripts/pdf_to_epub_hybrid.py`.
- Test `tests/test_pdf_to_epub_hybrid.py`.
- Add failing tests for late-page reading order, explicit column boundaries, legitimate repeated digits in notes, and full-page artwork retention.
- Remove the page-300 switch; expose `--column-split` in PDF points.
- Filter known emphasis-marker fonts, retain ambiguous numeric text, and reject only substantial replacement-character corruption.
- Preserve large raster visuals by default; expose `--skip-page-rasters` for reviewed scan-heavy sources.
- Run targeted tests, then commit extraction and existing visual/endnote improvements together.

### Task 2: Generalize flow output

- Add regressions for continuous inset paragraphs and for XHTML page flattening, inline links, page IDs, structural barriers, and geometry-based cross-page continuation.
- Add `--layout flow|pages` (flow default), conservative continuation detection, and a tree-based flattening function.
- Respect section boundaries, indentation, figures, index sections, and page-layout mode.
- Add repeatable translator metadata with the EPUB translator role.
- Run targeted tests and full suite before committing the flow feature.

### Task 3: Document and package the workflow

- Update `skills/converting-pdfs-to-epub/SKILL.md` and add `docs/hybrid-epub-validation.md`.
- Document text-layer screening, OCR range verification, source-scoped repair ledgers, image review, and the limits of structural/content checks.
- Ignore `/output/`; keep temporary scripts and copyrighted artifacts out of Git.
- Run the full suite and `git diff --check`; inspect staged paths and commit the docs/config separately.


## Execution results

- Extraction changes committed separately; legitimate numeric citations and full-page raster artwork are retained.
- Flow mode preserves page IDs and inline markup. Cross-page joining remains opt-in and uses native geometry; the test fixture verifies both join modes and section barriers.
- Additional coverage regression found and fixed omitted front matter and empty ranges caused by same-page bookmarks.
- Full suite: 273 passed, with 5 pre-existing PyMuPDF/SWIG deprecation warnings.
- Synthetic flow and pages EPUBs: helper verifier OK, EPUBCheck 5.4.0 zero fatals/errors/warnings; Calibre converted the flow EPUB to text.
- `git diff --check` passed. Source books, generated outputs, and temporary repair scripts were excluded. These checks do not establish whole-book proofreading.
