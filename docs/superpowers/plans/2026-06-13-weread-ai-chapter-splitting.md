# WeRead AI Chapter Splitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch `xu1.epub` and `xu2.epub` so 微信读书 AI朗读 is more likely to split by front matter, part, and chapter boundaries only.

**Architecture:** This is a finished-EPUB patch, not a pipeline change. A one-off Python script will inspect the two EPUB ZIPs, back them up, rewrite XHTML heading structure in place, preserve formal TOC anchors, and verify structural invariants.

**Tech Stack:** Python 3 standard library (`zipfile`, `re`, `html`, `shutil`, `pathlib`); existing EPUB files in `/Users/vyang/Desktop/spaces/pdf-craft`.

---

## File Structure

- Create: `tmp/patch_weread_ai_chapters.py`
  - One-off patch script for `xu1.epub` and `xu2.epub`.
  - Creates backups in `tmp/` before modifying each EPUB.
  - Rewrites XHTML tags inside EPUB archives.
  - Runs built-in verification and exits non-zero on failure.
- Modify: `xu1.epub`
  - Demote paper-TOC heading tags and lower-level section headings.
  - Preserve existing anchors and visible text.
- Modify: `xu2.epub`
  - Same structural patch strategy as `xu1.epub`.
- Create backup: `tmp/xu1.before_weread_ai_chapter_patch.epub`
- Create backup: `tmp/xu2.before_weread_ai_chapter_patch.epub`

## Rules Implemented by the Patch Script

Desired AI chapter candidates:

- `h1` front matter headings that are already top-level front matter.
- `第X編` part titles, normalized into one `h1` where they currently appear as paragraph blocks.
- `第X章` chapter titles, represented as one `h1` containing both the chapter number and the chapter title when adjacent split headings exist.

Non-AI chapter candidates:

- Paper table-of-contents headings inside `Content_0.xhtml` / `中國近代史_1.xhtml` that are only OCR layout.
- Lower-level section headings currently in `h2` under chapters.
- Nested TOC entries such as `阿美士德使團，1816年`, `戰爭爆發`, and `新文化運動的展開`.
- `參考書目` headings.

---

### Task 1: Create a failing structural verification script

**Files:**
- Create: `tmp/patch_weread_ai_chapters.py`

- [ ] **Step 1: Write the verification-only script**

Create `tmp/patch_weread_ai_chapters.py` with this initial content:

```python
#!/usr/bin/env python3
from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path('/Users/vyang/Desktop/spaces/pdf-craft')
BOOKS = [ROOT / 'xu1.epub', ROOT / 'xu2.epub']

UNWANTED_HEADING_TEXTS = {
    '阿美士德使團，1816年',
    '戰爭爆發',
    '新文化運動的展開',
}

REQUIRED_MEMBERS = {
    'mimetype',
    'META-INF/container.xml',
    'EPUB/content.opf',
    'EPUB/nav.xhtml',
    'EPUB/toc.ncx',
}


def strip_tags(value: str) -> str:
    return html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', value))).strip()


def norm(value: str) -> str:
    value = value.replace('，', ',').replace('–', '-').replace('—', '-')
    value = value.replace('「', '').replace('」', '').replace('　', ' ')
    return re.sub(r'\s+', '', value)


def collect_nav_targets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    nav = zf.read('EPUB/nav.xhtml').decode('utf-8', errors='replace')
    targets: list[tuple[str, str]] = []
    for match in re.finditer(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', nav, re.S):
        text = strip_tags(match.group(2))
        href = html.unescape(match.group(1))
        if text:
            targets.append((text, href))
    return targets


def collect_body_headings(zf: zipfile.ZipFile) -> list[tuple[str, str, str, str]]:
    headings: list[tuple[str, str, str, str]] = []
    for name in zf.namelist():
        if not name.startswith('EPUB/') or not name.endswith('.xhtml'):
            continue
        if name.endswith('/nav.xhtml') or name.endswith('/cover.xhtml'):
            continue
        data = zf.read(name).decode('utf-8', errors='replace')
        for match in re.finditer(r'<h([1-6])\b([^>]*)>(.*?)</h\1>', data, re.S):
            attrs = match.group(2)
            ident = ''
            id_match = re.search(r'\bid="([^"]+)"', attrs)
            if id_match:
                ident = id_match.group(1)
            headings.append((name, 'h' + match.group(1), ident, strip_tags(match.group(3))))
    return headings


def verify_book(path: Path) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            issues.append(f'ZIP corrupt member: {bad_member}')
        names = set(zf.namelist())
        for member in sorted(REQUIRED_MEMBERS):
            if member not in names:
                issues.append(f'missing required member: {member}')

        for text, href in collect_nav_targets(zf):
            filename, sep, fragment = href.partition('#')
            if not sep or not fragment:
                issues.append(f'{text}: nav href has no fragment: {href}')
                continue
            member = 'EPUB/' + filename if not filename.startswith('EPUB/') else filename
            if member not in names:
                issues.append(f'{text}: nav target file missing: {href}')
                continue
            data = zf.read(member).decode('utf-8', errors='replace')
            if f'id="{fragment}"' not in data:
                issues.append(f'{text}: nav target id missing: {href}')

        headings = collect_body_headings(zf)
        for name, tag, ident, text in headings:
            if tag in {'h1', 'h2'} and any(norm(text) == norm(bad) for bad in UNWANTED_HEADING_TEXTS):
                issues.append(f'{path.name}: unwanted AI heading remains: {name} {tag} {ident} {text}')

        if path.name == 'xu1.epub':
            required_chapter_texts = {
                '第一章 「近代中國」的概念',
                '第二章 清帝國的興盛',
                '第三章 政治和經濟體制',
                '第十章 太平天國革命、捻軍叛亂及 回民叛亂',
                '第十八章 晚清的思想、社會和經濟變化 重點討論1895–1911年',
            }
        else:
            required_chapter_texts = {
                '第二十章 革命、共和與軍閥割據',
                '第二十一章 思想革命，1917-1923年',
                '第三十五章 四個現代化',
                '第四十二章 香港的回歸和中美關係',
            }
        h1_texts = {text for _, tag, _, text in headings if tag == 'h1'}
        for expected in required_chapter_texts:
            if not any(norm(expected) in norm(actual) or norm(actual) in norm(expected) for actual in h1_texts):
                issues.append(f'{path.name}: expected chapter-level h1 missing: {expected}')
    return issues


def main() -> int:
    all_issues: list[str] = []
    for book in BOOKS:
        issues = verify_book(book)
        print(f'{book}: issues={len(issues)}')
        for issue in issues:
            print('  -', issue)
        all_issues.extend(issues)
    print('TOTAL_ISSUES=', len(all_issues))
    return 1 if all_issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run verification to confirm it fails before the patch**

Run:

```bash
python3 tmp/patch_weread_ai_chapters.py
```

Expected: exit code `1`. The output should report unwanted AI headings such as `戰爭爆發` or `新文化運動的展開`, and missing combined chapter-level `h1` headings where chapter number/title are currently split.

---

### Task 2: Add the EPUB patcher

**Files:**
- Modify: `tmp/patch_weread_ai_chapters.py`

- [ ] **Step 1: Replace the script with patch + verify implementation**

Replace the entire file with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import html
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path('/Users/vyang/Desktop/spaces/pdf-craft')
BOOKS = [ROOT / 'xu1.epub', ROOT / 'xu2.epub']

REQUIRED_MEMBERS = {
    'mimetype',
    'META-INF/container.xml',
    'EPUB/content.opf',
    'EPUB/nav.xhtml',
    'EPUB/toc.ncx',
}

UNWANTED_HEADING_TEXTS = {
    '阿美士德使團，1816年',
    '戰爭爆發',
    '新文化運動的展開',
}

PAPER_TOC_FILES = {
    'EPUB/Content_0.xhtml',
    'EPUB/中國近代史_1.xhtml',
}

CHAPTER_NUMBER_RE = re.compile(r'^第[零〇一二三四五六七八九十百兩0-9]+章$')
PART_NUMBER_RE = re.compile(r'^第[零〇一二三四五六七八九十百兩0-9]+編$')
REFERENCE_RE = re.compile(r'^參考書目$')


def strip_tags(value: str) -> str:
    return html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', value))).strip()


def norm(value: str) -> str:
    value = value.replace('，', ',').replace('–', '-').replace('—', '-')
    value = value.replace('「', '').replace('」', '').replace('　', ' ')
    value = value.replace('闕', '閥').replace('现', '現').replace('满', '滿')
    return re.sub(r'\s+', '', value)


def attrs_without(attrs: str, names: set[str]) -> str:
    for name in names:
        attrs = re.sub(r'\s*' + re.escape(name) + r'="[^"]*"', '', attrs)
    return attrs


def get_attr(attrs: str, name: str) -> str | None:
    match = re.search(r'\b' + re.escape(name) + r'="([^"]*)"', attrs)
    return match.group(1) if match else None


def p_with_attrs(attrs: str, text: str, class_name: str) -> str:
    ident = get_attr(attrs, 'id')
    clean = attrs_without(attrs, {'class'})
    if 'class=' not in clean:
        clean = clean.rstrip() + f' class="{class_name}"'
    if ident and 'id=' not in clean:
        clean = clean.rstrip() + f' id="{ident}"'
    return f'<p{clean}>{text}</p>'


def demote_all_headings_in_paper_toc(data: str) -> str:
    def repl(match: re.Match[str]) -> str:
        attrs = match.group(2)
        body = match.group(3)
        return p_with_attrs(attrs, body, 'paper-toc-line')
    return re.sub(r'<h([1-6])\b([^>]*)>(.*?)</h\1>', repl, data, flags=re.S)


def demote_section_headings(data: str) -> str:
    def repl(match: re.Match[str]) -> str:
        level = match.group(1)
        attrs = match.group(2)
        body = match.group(3)
        text = strip_tags(body)
        if level == '1':
            return match.group(0)
        if CHAPTER_NUMBER_RE.match(text):
            return match.group(0)
        if PART_NUMBER_RE.match(text):
            return match.group(0)
        if REFERENCE_RE.match(text):
            return p_with_attrs(attrs, body, 'section-title reference-title')
        return p_with_attrs(attrs, body, 'section-title')
    return re.sub(r'<h([1-6])\b([^>]*)>(.*?)</h\1>', repl, data, flags=re.S)


def merge_adjacent_chapter_h1(data: str) -> str:
    pattern = re.compile(
        r'<h1\b(?P<num_attrs>[^>]*)>(?P<num_body>.*?)</h1>\s*'
        r'<h1\b(?P<title_attrs>[^>]*)>(?P<title_body>.*?)</h1>',
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        num_text = strip_tags(match.group('num_body'))
        title_text = strip_tags(match.group('title_body'))
        if not CHAPTER_NUMBER_RE.match(num_text):
            return match.group(0)
        attrs = match.group('num_attrs')
        title_id = get_attr(match.group('title_attrs'), 'id')
        anchor = f'<span id="{title_id}" class="chapter-title-anchor"></span>' if title_id else ''
        return f'<h1{attrs}>{num_text} {anchor}{match.group("title_body")}</h1>'

    previous = None
    while previous != data:
        previous = data
        data = pattern.sub(repl, data)
    return data


def merge_adjacent_chapter_h2(data: str) -> str:
    pattern = re.compile(
        r'<h2\b(?P<num_attrs>[^>]*)>(?P<num_body>.*?)</h2>\s*'
        r'<h2\b(?P<title_attrs>[^>]*)>(?P<title_body>.*?)</h2>',
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        num_text = strip_tags(match.group('num_body'))
        title_text = strip_tags(match.group('title_body'))
        if not CHAPTER_NUMBER_RE.match(num_text):
            return match.group(0)
        attrs = match.group('num_attrs')
        title_id = get_attr(match.group('title_attrs'), 'id')
        anchor = f'<span id="{title_id}" class="chapter-title-anchor"></span>' if title_id else ''
        return f'<h1{attrs}>{num_text} {anchor}{match.group("title_body")}</h1>'

    previous = None
    while previous != data:
        previous = data
        data = pattern.sub(repl, data)
    return data


def merge_part_paragraph_blocks(data: str) -> str:
    pattern = re.compile(
        r'<p\b(?P<p1_attrs>[^>]*)>(?P<part>第[零〇一二三四五六七八九十百兩0-9]+編)</p>\s*'
        r'<p\b(?P<p2_attrs>[^>]*)>(?P<title>[^<]{2,80})</p>\s*'
        r'<p\b(?P<p3_attrs>[^>]*)>(?P<years>[0-9]{3,4}[–\-][0-9]{3,4}年?)</p>',
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        attrs = match.group('p1_attrs')
        return f'<h1{attrs}>{match.group("part")} {match.group("title")}，{match.group("years")}</h1>'

    return pattern.sub(repl, data)


def promote_standalone_chapter_h2(data: str) -> str:
    def repl(match: re.Match[str]) -> str:
        attrs = match.group(2)
        body = match.group(3)
        text = strip_tags(body)
        if CHAPTER_NUMBER_RE.match(text):
            return f'<h1{attrs}>{body}</h1>'
        return match.group(0)
    return re.sub(r'<h2\b([^>]*)>(.*?)</h2>', lambda m: repl(m), data, flags=re.S)


def patch_xhtml(name: str, data: str) -> str:
    if name in PAPER_TOC_FILES:
        return demote_all_headings_in_paper_toc(data)
    data = merge_part_paragraph_blocks(data)
    data = merge_adjacent_chapter_h1(data)
    data = merge_adjacent_chapter_h2(data)
    data = demote_section_headings(data)
    return data


def collect_nav_targets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    nav = zf.read('EPUB/nav.xhtml').decode('utf-8', errors='replace')
    targets: list[tuple[str, str]] = []
    for match in re.finditer(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', nav, re.S):
        text = strip_tags(match.group(2))
        href = html.unescape(match.group(1))
        if text:
            targets.append((text, href))
    return targets


def collect_body_headings(zf: zipfile.ZipFile) -> list[tuple[str, str, str, str]]:
    headings: list[tuple[str, str, str, str]] = []
    for name in zf.namelist():
        if not name.startswith('EPUB/') or not name.endswith('.xhtml'):
            continue
        if name.endswith('/nav.xhtml') or name.endswith('/cover.xhtml'):
            continue
        data = zf.read(name).decode('utf-8', errors='replace')
        for match in re.finditer(r'<h([1-6])\b([^>]*)>(.*?)</h\1>', data, re.S):
            attrs = match.group(2)
            ident = get_attr(attrs, 'id') or ''
            headings.append((name, 'h' + match.group(1), ident, strip_tags(match.group(3))))
    return headings


def verify_book(path: Path) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            issues.append(f'ZIP corrupt member: {bad_member}')
        names = set(zf.namelist())
        for member in sorted(REQUIRED_MEMBERS):
            if member not in names:
                issues.append(f'missing required member: {member}')

        for text, href in collect_nav_targets(zf):
            filename, sep, fragment = href.partition('#')
            if not sep or not fragment:
                issues.append(f'{text}: nav href has no fragment: {href}')
                continue
            member = 'EPUB/' + filename if not filename.startswith('EPUB/') else filename
            if member not in names:
                issues.append(f'{text}: nav target file missing: {href}')
                continue
            data = zf.read(member).decode('utf-8', errors='replace')
            if f'id="{fragment}"' not in data:
                issues.append(f'{text}: nav target id missing: {href}')

        headings = collect_body_headings(zf)
        for name, tag, ident, text in headings:
            if tag in {'h1', 'h2'} and any(norm(text) == norm(bad) for bad in UNWANTED_HEADING_TEXTS):
                issues.append(f'{path.name}: unwanted AI heading remains: {name} {tag} {ident} {text}')

        if path.name == 'xu1.epub':
            required_chapter_texts = {
                '第一章 「近代中國」的概念',
                '第二章 清帝國的興盛',
                '第三章 政治和經濟體制',
                '第十章 太平天國革命、捻軍叛亂及 回民叛亂',
                '第十八章 晚清的思想、社會和經濟變化 重點討論1895–1911年',
            }
        else:
            required_chapter_texts = {
                '第二十章 革命、共和與軍閥割據',
                '第二十一章 思想革命，1917-1923年',
                '第三十五章 四個現代化',
                '第四十二章 香港的回歸和中美關係',
            }
        h1_texts = {text for _, tag, _, text in headings if tag == 'h1'}
        for expected in required_chapter_texts:
            if not any(norm(expected) in norm(actual) or norm(actual) in norm(expected) for actual in h1_texts):
                issues.append(f'{path.name}: expected chapter-level h1 missing: {expected}')
    return issues


def patch_book(path: Path) -> None:
    backup = path.with_name(f'tmp/{path.stem}.before_weread_ai_chapter_patch.epub')
    shutil.copy2(path, backup)

    with zipfile.ZipFile(path, 'r') as zin:
        payloads: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info in zin.infolist():
            raw = zin.read(info.filename)
            new_raw = raw
            if info.filename.startswith('EPUB/') and info.filename.endswith('.xhtml'):
                text = raw.decode('utf-8')
                patched = patch_xhtml(info.filename, text)
                new_raw = patched.encode('utf-8')
            payloads.append((info, new_raw))

    tmp = path.with_suffix('.epub.tmp')
    with zipfile.ZipFile(tmp, 'w') as zout:
        for info, raw in payloads:
            zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            zi.comment = info.comment
            zi.extra = info.extra
            zi.internal_attr = info.internal_attr
            zi.external_attr = info.external_attr
            zi.create_system = info.create_system
            zi.compress_type = zipfile.ZIP_STORED if info.filename == 'mimetype' else info.compress_type
            zout.writestr(zi, raw)
    tmp.replace(path)
    print(f'backup: {backup}')


def main() -> int:
    for book in BOOKS:
        patch_book(book)

    all_issues: list[str] = []
    for book in BOOKS:
        issues = verify_book(book)
        print(f'{book}: issues={len(issues)}')
        for issue in issues:
            print('  -', issue)
        all_issues.extend(issues)
    print('TOTAL_ISSUES=', len(all_issues))
    return 1 if all_issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run the patch script**

Run:

```bash
python3 tmp/patch_weread_ai_chapters.py
```

Expected: exit code `0`, backup paths printed, and `TOTAL_ISSUES= 0`.

---

### Task 3: Inspect high-signal heading output

**Files:**
- Read-only inspection of `xu1.epub` and `xu2.epub`

- [ ] **Step 1: Run heading summary**

Run:

```bash
python3 - <<'PY'
import zipfile, re, html
for path in ['xu1.epub', 'xu2.epub']:
    print('\n###', path)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.startswith('EPUB/') or not name.endswith('.xhtml'):
                continue
            if name.endswith('/nav.xhtml') or name.endswith('/cover.xhtml'):
                continue
            data = zf.read(name).decode('utf-8', errors='replace')
            headings = []
            for match in re.finditer(r'<h([1-2])\b[^>]*>(.*?)</h\1>', data, re.S):
                text = html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', match.group(2)))).strip()
                headings.append('h' + match.group(1) + ' ' + text)
            if headings:
                print(name)
                for heading in headings[:80]:
                    print(' ', heading)
PY
```

Expected:

- `阿美士德使團，1816年`, `戰爭爆發`, and `新文化運動的展開` do not appear as `h1` or `h2`.
- Chapter boundaries such as `第十章 太平天國革命、捻軍叛亂及 回民叛亂` appear as `h1`.
- Paper-TOC files no longer contain many OCR `h2` headings from the printed table of contents.

---

### Task 4: Manual WeRead verification handoff

**Files:**
- No file changes

- [ ] **Step 1: Prepare user verification instructions**

Tell the user:

```text
Please delete the old imported copies of xu1.epub and xu2.epub from 微信读书, then import the patched files as fresh books. After import, open AI朗读 and check whether the chapter list now mainly contains front matter, parts, and 第X章 entries rather than small sections such as 阿美士德使團、戰爭爆發、新文化運動的展開.
```

- [ ] **Step 2: If WeRead still shows stale chapters, ask for the new AI朗读 list**

If the user reports mismatch, ask them to paste the new AI朗读 list. Compare the new list against the heading summary from Task 3 before making another patch.

---

## Self-Review

- Spec coverage: The plan patches only `xu1.epub` and `xu2.epub`, preserves TOC anchors, demotes paper TOC headings, demotes lower-level headings, keeps chapter-level `h1` signals, and verifies known unwanted entries.
- Placeholder scan: No `TBD`, `TODO`, or vague implementation steps remain.
- Type consistency: The script consistently uses `Path`, `zipfile.ZipFile`, string-based regex transformations, and the same `verify_book()` function before and after patching.
