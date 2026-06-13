# WeRead AI Chapter Splitting Design

## Goal

Adjust the existing `xu1.epub` and `xu2.epub` files so WeChat Read / 微信读书 AI 朗读 is more likely to split by normal book-level chapters rather than mixing in paper-TOC entries and lower-level section headings.

The target AI朗读 structure is **chapter-level only**: front matter, part titles (`第X編`), and chapter titles (`第X章 ...`). Lower-level sections such as `阿美士德使團，1816年`, `戰爭爆發`, and `新文化運動的展開` should not be promoted to AI朗读 chapter entries.

## Scope

Only patch the existing finished EPUB files:

- `xu1.epub`
- `xu2.epub`

Do not change the EPUB generation pipeline in this pass. Do not split XHTML files into one file per chapter.

## Current Problem

The EPUBs contain a valid formal TOC (`EPUB/nav.xhtml` and `EPUB/toc.ncx`), but WeRead AI朗读 appears to infer its own chapter list from a mix of:

1. formal TOC entries,
2. XHTML heading tags (`h1`, `h2`), and
3. OCR text from the printed paper table-of-contents pages.

This causes incomplete and inconsistent AI朗读 chapter lists. In `xu1.epub`, examples include lower-level entries such as `阿美士德使團，1816年`, `戰爭爆發`, and `新文化運動的展開` appearing alongside part and chapter titles.

## Recommended Approach

Use a minimal structural patch:

1. Keep formal EPUB TOC links intact.
2. Keep all content text visible in reading order.
3. Preserve existing `id` anchors so directory links still resolve.
4. Promote only desired AI朗读 boundaries to strong heading structure.
5. Demote lower-level section headings from `h2` to paragraph-like title elements.
6. Demote heading-like tags in paper table-of-contents pages so WeRead AI does not treat OCR TOC rows as chapter boundaries.

## Chapter Boundary Rules

Treat these as AI朗读 chapter candidates:

- front-matter headings such as `出版者言`, `原著者中文版序`, `郭序`, `第六版序`, `第一版序`, `歷代紀元表`, `貨幣及度量衡折算表`, and map/table directories when present;
- part headings matching `第X編 ...`;
- chapter headings matching `第X章 ...`.

Treat these as non-chapter section headings:

- ordinary topical section headings under chapters;
- nested section entries under TOC items;
- `參考書目`;
- OCR paper-TOC headings such as repeated `目錄` markers and embedded TOC row titles.

## Implementation Shape

Patch EPUB ZIP contents in place with backups under `tmp/`.

For each target EPUB:

1. Read `nav.xhtml` to identify formal TOC entries and known anchors.
2. Inspect each XHTML content document.
3. For body headings:
   - keep or convert desired front matter / part / chapter boundaries to `h1` where appropriate;
   - demote lower-level `h2` headings to `<p id="..." class="section-title">...</p>` or an equivalent paragraph element preserving the same `id`.
4. For paper TOC pages:
   - demote heading tags used only for printed TOC/OCR layout to paragraph elements;
   - preserve text and anchors.
5. Repack the EPUB preserving `mimetype`, required members, and existing resource files.

## Verification

After patching both EPUBs, verify:

- ZIP integrity passes for each EPUB.
- Required files exist: `mimetype`, `META-INF/container.xml`, `EPUB/content.opf`, `EPUB/nav.xhtml`, `EPUB/toc.ncx`.
- Formal TOC links still resolve to existing files and IDs.
- No formal TOC entry points to an empty span.
- Desired top-level AI朗读 candidate headings remain as `h1` or equivalent strong chapter headings.
- Known unwanted AI朗读 entries are no longer represented as `h1`/`h2` headings:
  - `阿美士德使團，1816年`
  - `戰爭爆發`
  - `新文化運動的展開`

## Risks

WeRead AI朗读 is a black box. This patch improves the structural signals it appears to use, but cannot guarantee exact behavior without re-importing into WeRead and checking the app.

The recommended user verification is to delete the old imported book from WeRead, import the patched EPUB as a fresh book, and inspect the AI朗读 chapter list again.
