---
name: weread-epub-chapterizing
description: "Use when an EPUB imported into 微信读书/WeRead AI朗读 shows a mixed or random chapter list: formal TOC sub-sections, nested headings, paper table-of-contents rows, or multiple book chapters grouped under one AI reading item."
---

# WeRead EPUB Chapterizing

## Overview

Restructure finished EPUB files so WeRead receives one consistent chapter signal from `nav.xhtml`, `toc.ncx`, OPF spine files, and body `h1` headings.

Use this when WeRead AI朗读 lists small sections such as `阿美士德使團，1816年`, `戰爭爆發`, or `新文化運動的展開`, or when a few large XHTML files contain many book chapters.

## Workflow

1. Inspect the EPUB before modifying it:

```bash
python3 -m paddle_pipeline.weread_chapterize book.epub --verify-only
```

If the verifier reports nested nav entries, `spine/nav count mismatch`, or non-boundary labels, use chapterizing.

2. For a fresh PDF conversion, run the main pipeline with WeRead post-processing:

```bash
python3 pdf2epub_paddle.py book.pdf --title "Book Title" --author "Author" --auto-toc --weread-chapterize
```

3. For an existing EPUB, patch in place with a backup:

```bash
pdf2epub-weread-chapterize book.epub --backup tmp/book.before_weread_chapterize.epub
```

Or write a separate output:

```bash
pdf2epub-weread-chapterize input.epub -o output.epub
```

4. Verify again:

```bash
pdf2epub-weread-chapterize book.epub --verify-only
python3 -m zipfile --test book.epub
```

Required result: `TOTAL_ISSUES=0` and `Done testing`.

## What The Tool Does

- Selects only front matter, `第X編`, `第X章`, and `索引` TOC entries as reading boundaries.
- Rewrites `nav.xhtml` and `toc.ncx` into a flat one-level reading TOC.
- Replaces large source XHTML files with `EPUB/weread_001.xhtml`, `EPUB/weread_002.xhtml`, etc.
- Keeps each generated XHTML file to exactly one reading-level `h1`.
- Removes old source XHTML files from the EPUB package so WeRead cannot scan stale section headings.
- Preserves visible body text under the nearest reading boundary.

## Do Not

- Do not only demote `h2`/`h3` headings when WeRead is reading from the formal TOC; that leaves nested `nav.xhtml` / `toc.ncx` entries intact.
- Do not patch `nav.xhtml` without updating `toc.ncx`, `content.opf`, and the spine together.
- Do not delete body text for unwanted AI entries. The unwanted text should remain in the chapter body, just not as a TOC/spine/h1 boundary.
- Do not reuse old WeRead imports for validation; delete the old imported copy and import the patched EPUB fresh.

## Code Entry Points

- Library: `paddle_pipeline.weread_chapterize`
- Main API: `chapterize_epub_for_weread(...)`
- Verifier API: `verify_weread_chapterized_epub(...)`
- Main pipeline flag: `python3 pdf2epub_paddle.py ... --weread-chapterize`
- CLI: `pdf2epub-weread-chapterize ...`
- Module CLI: `python3 -m paddle_pipeline.weread_chapterize ...`
