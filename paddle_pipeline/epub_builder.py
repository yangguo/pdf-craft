"""Main EPUB assembly — creates the EPUB from aggregated OCR results."""

import os
import re

from typing import Any, Dict, List

from .config import DEFAULT_EPUB_LANGUAGE
from .config import epub  # Optional dependency
from .epub_validate import write_validated_epub
from .footnotes import (
    extract_page_footnotes,
    format_page_footnotes_html,
    link_page_footnote_references,
)
from .ocr_noise import clean_ocr_noise

# Chinese numerals used in page numbers
_CN_NUMERALS = set("零一二三四五六七八九十百千")
_MANUAL_HEADING_BRACKET_RE = re.compile(
    r"^[\(（〔【\[]\s*"
    r"(第[零一二三四五六七八九十百千0-9]+[章節节编編篇部卷])"
    r"\s*[\)）〕】\]]\s*"
)
_MANUAL_HEADING_PREFIX_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千0-9]+[章節节编編篇部卷])\s+"
)


def _normalize_manual_heading_text(text: Any) -> str:
    """Return a stable key for matching manually confirmed OCR headings."""
    value = str(text or "").strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", value).strip()
    value = re.sub(r"^\d+\s+", "", value).strip()
    value = _MANUAL_HEADING_BRACKET_RE.sub(r"\1 ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _manual_heading_sources(
    heading: Dict[str, Any],
    *,
    include_aliases: bool = True,
) -> list[str]:
    """Return the display title plus any OCR/source aliases for a heading."""
    values: list[str] = [str(heading.get("title", ""))]
    for key in ("match", "source", "source_title", "ocr", "ocr_title"):
        raw = heading.get(key)
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw)
    raw_matches = heading.get("matches")
    if isinstance(raw_matches, list):
        values.extend(str(item) for item in raw_matches)
    if include_aliases:
        raw_aliases = heading.get("aliases")
        if isinstance(raw_aliases, list):
            values.extend(str(item) for item in raw_aliases)
    return values


def _build_confirmed_heading_lookup(
    confirmed_headings: List[Dict] | None,
) -> dict[tuple[int, str], str]:
    """Map (page, normalized OCR heading) to the intended display title."""
    lookup: dict[tuple[int, str], str] = {}
    if confirmed_headings is None:
        return lookup

    for heading in confirmed_headings:
        title = str(heading.get("title", "")).strip()
        if not title:
            continue
        page = int(heading["page"])
        for source in _manual_heading_sources(heading, include_aliases=False):
            key = _normalize_manual_heading_text(source)
            if not key:
                continue
            lookup[(page, key)] = title
            prefix_match = _MANUAL_HEADING_PREFIX_RE.match(key)
            if prefix_match:
                lookup[(page, prefix_match.group(1))] = title
    return lookup


def _build_page_start_heading_lookup(
    confirmed_headings: List[Dict] | None,
) -> dict[int, str]:
    """Map pages that should begin a manual TOC entry to display titles."""
    lookup: dict[int, str] = {}
    if confirmed_headings is None:
        return lookup
    for heading in confirmed_headings:
        if not heading.get("page_start") and heading.get("at") != "page_start":
            continue
        title = str(heading.get("title", "")).strip()
        if title:
            lookup[int(heading["page"])] = title
    return lookup


def _lookup_confirmed_heading(
    lookup: dict[tuple[int, str], str],
    current_page: int,
    heading_text: Any,
) -> str | None:
    key = _normalize_manual_heading_text(heading_text)
    if not key:
        return None
    return lookup.get((current_page, key)) or lookup.get((current_page - 1, key))


def _build_header_fingerprints(confirmed_headings: List[Dict] | None,
                               book_title: str = "") -> set:
    """Build a set of whitespace-stripped chapter titles for page-header detection.

    Also extracts individual components (e.g. "第十四章" and "「六四」風雲"
    from "第十四章 「六四」風雲") so partial-title page headers are matched.
    Includes the book title so running-page-header copies are stripped.
    """
    fingerprints = set()
    if confirmed_headings:
        for h in confirmed_headings:
            for title in _manual_heading_sources(h):
                fp = re.sub(r"\s+", "", title)
                if fp:
                    fingerprints.add(fp)
                # Also fingerprint just the chapter-number prefix and title suffix
                # so running page headers like "第十四章" or "「六四」風雲" match.
                normalized = _normalize_manual_heading_text(title)
                m = re.match(
                    r"(第[零一二三四五六七八九十百千0-9]+[章節编編篇卷])\s*(.+)",
                    normalized,
                )
                if m:
                    num_fp = re.sub(r"\s+", "", m.group(1))
                    title_fp = re.sub(r"\s+", "", m.group(2))
                    fingerprints.add(num_fp)
                    fingerprints.add(title_fp)
    # Add book title so running page headers carrying the title are stripped.
    if book_title:
        fp = re.sub(r"\s+", "", book_title)
        if fp:
            fingerprints.add(fp)
    return fingerprints


def _strip_page_headers(
    lines: list,
    fingerprints: set,
    preserve_indexes: set[int] | None = None,
) -> list:
    """Remove OCR page headers / footers and Chinese page numbers from a page.

    Page headers appear as standalone lines and match:
    - A known chapter-title or book-title fingerprint (whitespace removed)
    - A short chapter-like line (e.g. '第X章 ...' ≤25 chars) without heading markup
    - A Chinese-numeral page number (e.g. '五', '二七')
    - A short line in book-title brackets 「...」 (≤20 chars)

    Fingerprint, chapter-like, and page-number checks apply to all lines
    because OCR reading order may place running headers anywhere in the
    output. Generic bracket-title stripping is limited to page edges or
    lines adjacent to page numbers to avoid deleting body dialogue.
    """
    if len(lines) < 2:
        return lines

    _chapterish = re.compile(r"第[零一二三四五六七八九十百千0-9]+[章節编編篇卷]")
    # Short lines that look like Chinese section titles or running headers.
    # Book-title brackets 「...」 or Western quotes with Chinese content.
    _titleish = re.compile(r'^(?:「[^」]{1,20}」|『[^』]{1,20}』|"[^"]{1,20}")$')

    def _is_page_number(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if s.isdigit():
            return True
        return all(c in _CN_NUMERALS for c in s) and len(s) <= 3

    def _is_header_line(s: str, idx: int) -> bool:
        s = s.strip()
        if not s:
            return False
        fp = re.sub(r"\s+", "", s)
        if fp in fingerprints:
            return True
        # Short chapter-like line (≤25 chars) — a standalone page header,
        # not body text like "第三章講述了中英談判的漫長過程..."
        if _chapterish.match(s) and not s.startswith("#") and len(s) <= 25:
            return True
        # Short standalone line in book-title brackets — almost
        # certainly a running page header at page edges or near page numbers,
        # but preserve mid-page quoted body/dialogue lines.
        if _titleish.match(s) and len(s) <= 20:
            near_edge_or_number = (
                idx == 0
                or idx == n - 1
                or (idx > 0 and _is_page_number(lines[idx - 1]))
                or (idx + 1 < n and _is_page_number(lines[idx + 1]))
            )
            return near_edge_or_number
        return False

    result = list(lines)
    n = len(result)

    preserved = preserve_indexes or set()
    for idx in range(n):
        if idx in preserved:
            continue
        line = result[idx]
        if not line.strip():
            continue
        if _is_page_number(line):
            result[idx] = ""
        elif _is_header_line(line, idx):
            result[idx] = ""
            if idx + 1 < n and _is_page_number(result[idx + 1]):
                result[idx + 1] = ""

    return [l for l in result if l.strip() or l == ""]


def _strip_missing_image_references(markdown_text: str,
                                    missing_image_paths: set[str]) -> str:
    """Remove Markdown and HTML image references whose assets were not packaged."""
    for missing in missing_image_paths:
        escaped = re.escape(missing)
        markdown_text = re.sub(
            r"!\[[^\]]*\]\(" + escaped + r"\)",
            "",
            markdown_text,
        )
        markdown_text = re.sub(
            r"<img\b[^>]*\bsrc=[\"']" + escaped + r"[\"'][^>]*/?>",
            "",
            markdown_text,
            flags=re.IGNORECASE,
        )
    return re.sub(
        r"<div\b[^>]*>\s*</div>",
        "",
        markdown_text,
        flags=re.IGNORECASE,
    )


def create_epub(title: str, results: List[Dict], output_file: str, image_dir: str,
                cover_image_path: str | None = None, author: str | None = None,
                confirmed_headings: List[Dict] | None = None,
                language: str = DEFAULT_EPUB_LANGUAGE,
                strict_ocr_validation: bool = False):
    """
    Creates an EPUB file from the aggregated API results.
    """
    book = epub.EpubBook()
    book.set_identifier(f"id_{title}")
    book.set_title(title)
    book.set_language(language)

    if author:
        book.add_author(author)

    # Set cover image
    if cover_image_path and os.path.exists(cover_image_path):
        with open(cover_image_path, "rb") as f:
            cover_data = f.read()
        book.set_cover(os.path.basename(cover_image_path), cover_data)

    chapters = []

    # CSS for the book
    style = """
    body { font-family: serif; line-height: 1.8; text-align: justify; margin: 1em; }
    h1 { text-align: center; margin: 1.5em 0 0.8em 0; font-size: 1.6em; }
    h2 { margin: 1.2em 0 0.6em 0; font-size: 1.3em; }
    h3 { margin: 1em 0 0.5em 0; font-size: 1.1em; }
    p { margin-bottom: 0.8em; }
    blockquote { margin: 1em 2em; font-style: italic; }
    img { max-width: 100%; height: auto; display: block; margin: 1em auto; }
    .page-footnotes {
        margin-top: 1.2em;
        padding-top: 0.6em;
        border-top: 1px solid #999;
        font-size: 0.85em;
        line-height: 1.5;
    }
    .page-footnotes p { margin: 0.35em 0; text-align: left; }
    .page-footnotes sup { font-size: 0.8em; vertical-align: super; }
    .footnote-ref a { text-decoration: none; }
    .unlinked-footnote-marker { font-size: 0.8em; vertical-align: super; }
    .footnote-backlink { margin-left: 0.4em; text-decoration: none; }
    """
    nav_css = epub.EpubItem(
        uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style
    )
    book.add_item(nav_css)

    # Process each chunk's result
    # The API returns 'result' -> 'layoutParsingResults' (list)
    # Each item in layoutParsingResults corresponds to a page (usually)

    # Combine all pages into chapters if they have headings, or just page-by-page?
    # Better to just append text. If we see a H1/H2, we can make a split potentially,
    # but for simplicity, let's dump pages into one big flow or chapter-per-chunk?
    # 50 pages per chapter is too much.
    # Let's try to detect headers in the markdown to split chapters.

    full_markdown = ""

    # Build chapter title fingerprints for page-header detection
    _header_fingerprints = _build_header_fingerprints(confirmed_headings, title)
    _confirmed_heading_lookup = _build_confirmed_heading_lookup(confirmed_headings)
    _page_start_heading_lookup = _build_page_start_heading_lookup(confirmed_headings)

    # Aggregated images to add to the book
    # Map API image path (e.g. 'images/tmp.jpg') to internal EPUB path

    global_page = 0  # tracks the same counter used by extract_candidate_headings

    for chunk_idx, result in enumerate(results):
        if not result or "result" not in result:
            continue

        layout_results = result["result"].get("layoutParsingResults", [])

        for i, page_res in enumerate(layout_results):
            global_page += 1
            # 1. Get Markdown Text
            page_md = clean_ocr_noise(page_res["markdown"]["text"])

            # 2. Handle Images
            # page_res["markdown"]["images"] is { relative_path: url }
            # The markdown text uses relative_path: ![](images/xxx.jpg)
            images_map = page_res["markdown"].get("images", {})

            # Track which image paths were successfully packaged so we can
            # strip dangling ![...](...) references from the markdown later.
            missing_image_paths: set[str] = set()

            for rel_path, img_url in images_map.items():
                # Define local path where we downloaded the image
                local_img_path = os.path.join(image_dir, rel_path)

                # Check if we successfully downloaded it
                if os.path.exists(local_img_path):
                    # Add to EPUB
                    # Read image data
                    with open(local_img_path, "rb") as img_f:
                        img_data = img_f.read()

                    # Create EPUB Image Item
                    # Use the same relative path as filename to match markdown links
                    # (assuming markdown is ![](images/xxx))
                    epub_img = epub.EpubImage()
                    epub_img.file_name = rel_path
                    ext = os.path.splitext(rel_path)[1].lower()
                    _MIME_MAP = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                        ".svg": "image/svg+xml",
                    }
                    epub_img.media_type = _MIME_MAP.get(ext, "image/jpeg")
                    epub_img.content = img_data

                    if rel_path not in [item.file_name for item in book.get_items()]:
                        book.add_item(epub_img)
                else:
                    # Download failed or not yet attempted — record so the markdown
                    # reference can be stripped to avoid broken <img> in the EPUB.
                    missing_image_paths.add(rel_path)
                    print(f"    [!] Image not available, removing from markdown: {rel_path}")

            # 3. Concatenate Text with paragraph reflow
            # Strip markdown image links whose assets were not packaged, so the
            # generated HTML does not contain broken <img src="images/..."> references.
            if missing_image_paths:
                page_md = _strip_missing_image_references(
                    page_md,
                    missing_image_paths,
                )

            # Reflow lines into continuous paragraphs to fix sentence splitting issues
            # Pages often break sentences mid-flow, and OCR can split paragraphs
            # into separate lines that should be joined.

            lines = page_md.split("\n")
            cleaned_lines = []
            for line in lines:
                # Basic Page Number Removal (digit only)
                if line.strip().isdigit():
                    continue
                cleaned_lines.append(line)

            # Remove page headers / footers and Chinese page numbers
            if _header_fingerprints:
                preserve_indexes = {
                    idx for idx, line in enumerate(cleaned_lines)
                    if _lookup_confirmed_heading(
                        _confirmed_heading_lookup,
                        global_page,
                        line,
                    )
                }
                cleaned_lines = _strip_page_headers(
                    cleaned_lines,
                    _header_fingerprints,
                    preserve_indexes=preserve_indexes,
                )

            # Reflow: join lines that appear to be part of the same paragraph
            reflowed_paragraphs = []
            current_para = ""

            # Terminal punctuation that indicates end of sentence/paragraph
            # Includes CJK full-width equivalents for Chinese/Japanese OCR.
            terminal_chars = ('.', '!', '?', '"', "'", ')', ']', '}', ':', ';',
                              '。', '！', '？', '：', '；', '」', '』', '）', '】', '〉', '》')
            # Patterns that indicate a line is a header or special block.
            # Includes plain-text chapter headings (第X章) that OCR output
            # without markdown markup — these must stay standalone for the
            # chapter split detection to find them.
            header_pattern = re.compile(
                r'^(#{1,6}\s|>|\s*[-*]\s|\d+\.)'
            )
            # Plain-text chapter headings (第X章) that OCR outputs without
            # markdown markup.  Only treat as standalone heading when the
            # line is short enough (≤25 chars) to be a real page/chapter
            # header rather than body text that happens to mention a chapter.
            _chapterish_heading = re.compile(
                r'^第[零一二三四五六七八九十百千0-9]+[章節编編篇卷]'
            )
            # Characters that indicate a line should not be joined with next
            non_join_endings = (':', ';', ',', '-', '：', '；', '，')

            consecutive_blanks = 0

            for i, line in enumerate(cleaned_lines):
                stripped = line.rstrip()
                if not stripped:
                    consecutive_blanks += 1
                    continue

                # Check if this line is a header/list/blockquote
                is_manual_heading = bool(
                    _lookup_confirmed_heading(
                        _confirmed_heading_lookup,
                        global_page,
                        stripped,
                    )
                )
                is_special = header_pattern.match(stripped) or (
                    _chapterish_heading.match(stripped) and len(stripped) <= 25
                ) or is_manual_heading

                if is_special:
                    if current_para:
                        reflowed_paragraphs.append(current_para)
                        current_para = ""
                    reflowed_paragraphs.append(stripped)
                    consecutive_blanks = 0
                elif not current_para:
                    current_para = stripped
                    consecutive_blanks = 0
                else:
                    current_ends = current_para.rstrip()

                    if consecutive_blanks >= 2:
                        # Multiple blank lines — real paragraph break
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif consecutive_blanks == 1:
                        # Single blank between text lines — often OCR line-wrap
                        # noise, but sometimes a real paragraph break.
                        if current_ends.endswith(terminal_chars):
                            # Previous ends a sentence — real break
                            reflowed_paragraphs.append(current_para)
                            current_para = stripped
                        elif (len(stripped) > 20
                              and stripped.endswith(terminal_chars)
                              and not current_ends.endswith('，')):
                            # Next line is a paragraph that ends with
                            # terminal punctuation — it is a self-contained
                            # sentence/paragraph, not a mid-sentence
                            # continuation.  But if the current line ends
                            # with a CJK comma, the next line is likely still
                            # the same sentence (just broken by a page/line
                            # boundary).
                            reflowed_paragraphs.append(current_para)
                            current_para = stripped
                        else:
                            current_para = current_ends + " " + stripped
                    elif current_ends.endswith(terminal_chars):
                        # No blank but terminal punctuation — new sentence
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif current_ends and current_ends[-1] in non_join_endings:
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif len(stripped) < 20 and not stripped.endswith(terminal_chars):
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    else:
                        current_para = current_ends + " " + stripped

                    consecutive_blanks = 0

            # Don't forget the last paragraph
            if current_para:
                reflowed_paragraphs.append(current_para)

            # Join paragraphs and append to full_markdown
            # But first, check if we need to merge with the previous page's content
            # (sentences can be split across page boundaries)
            page_markdown = "\n\n".join(reflowed_paragraphs)
            page_footnotes = extract_page_footnotes(page_res)
            page_markdown, footnote_refs = link_page_footnote_references(
                page_markdown,
                page_footnotes,
                global_page,
            )
            footnote_html = format_page_footnotes_html(
                page_footnotes,
                global_page,
                footnote_refs=footnote_refs,
            )
            if footnote_html:
                page_markdown = (page_markdown.rstrip() + "\n\n" + footnote_html).strip()

            if full_markdown and page_markdown:
                # Check if full_markdown ends mid-sentence (no terminal punctuation)
                # and page_markdown starts with a continuation (not a header)
                full_stripped = full_markdown.rstrip()
                page_lines = page_markdown.split('\n')
                first_page_line = page_lines[0].strip() if page_lines else ""

                # Check if we should merge (previous doesn't end with terminal, next isn't a header)
                ends_mid_sentence = full_stripped and not full_stripped.endswith(terminal_chars)
                next_is_header = (
                    header_pattern.match(first_page_line)
                    or (_chapterish_heading.match(first_page_line) and len(first_page_line) <= 25)
                ) if first_page_line else False

                # Prepend page marker AFTER first_page_line is captured so merge logic is unaffected
                # The marker must be on its own line so the later ^\x00PAGE:\d+\x00$ regex matches it;
                # in the merge branch we ensure a newline precedes it before joining.
                tagged_page_markdown = f"\x00PAGE:{global_page}\x00\n" + page_markdown

                if ends_mid_sentence and not next_is_header:
                    # Merge: strip trailing whitespace, insert marker on its own line,
                    # then join the first content line of the new page with a space.
                    full_markdown = full_markdown.rstrip() + "\n" + f"\x00PAGE:{global_page}\x00\n" + page_markdown.lstrip() + "\n\n"
                else:
                    full_markdown += tagged_page_markdown + "\n\n"
            else:
                full_markdown += f"\x00PAGE:{global_page}\x00\n" + page_markdown + "\n\n"

    # Split markdown into chapters based on headers (# Header)
    # If no headers found, put everything in one chapter.

    # Split by H1 or H2, BUT only if it looks like a real Chapter/Part title
    # Regex for headers: ^#+ \s* (Title)

    # Collapse accumulated page-boundary newlines back to paragraph breaks.
    # Each page appends "\n\n" at assembly time; when pages stack the gap can
    # grow to 4+ newlines which later leak into XHTML as excessive \r\n runs.
    full_markdown = re.sub(r"\n{3,}", "\n\n", full_markdown)

    md_lines = full_markdown.split("\n")
    current_chapter_title = None  # None means no chapter header seen yet
    current_chapter_content = []
    chapter_count = 0

    # Regex to identify "Major" headers (Chapters/Parts) to split on (legacy fallback)
    major_header_pattern = re.compile(
        r"^(#{1,2})\s+(?:Chapter|Part|Lecture|Preface|Intro|Appendix|Prologue|Epilogue|Conclusion|Book|Acknowledgements|Contents|Abstract|序|前言|导论|目录|第[零一二三四五六七八九十百千0-9]+[篇章讲]).*"
    )

    # Regex for ANY header to format as H1/H2 in HTML but not necessarily split
    any_header_pattern = re.compile(r"^(#{1,2})\s+(.+)$")

    _page_marker_re = re.compile(r"^\x00PAGE:(\d+)\x00$")

    current_page = 0
    last_split_page = -1  # guard against duplicate splits on the same page

    for line in md_lines:
        # Consume page boundary markers (injected during markdown assembly)
        _pm = _page_marker_re.match(line)
        if _pm:
            current_page = int(_pm.group(1))
            page_start_title = _page_start_heading_lookup.get(current_page)
            if page_start_title and current_page != last_split_page:
                if current_chapter_content:
                    display_title = current_chapter_title or "Content"
                    safe_title = "".join(
                        c for c in display_title
                        if c.isalnum() or c in (" ", "_", "-")
                    ).strip()
                    if not safe_title:
                        safe_title = f"chap_{chapter_count}"

                    c = epub.EpubHtml(
                        title=display_title,
                        file_name=f"{safe_title}_{chapter_count}.xhtml",
                        lang=language,
                    )

                    try:
                        import markdown
                        html_content = markdown.markdown(
                            "\n".join(current_chapter_content)
                        )
                    except ImportError:
                        html_content = (
                            "<p>" + "</p><p>".join(current_chapter_content) + "</p>"
                        )

                    c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
                    c.add_item(nav_css)
                    book.add_item(c)
                    chapters.append(c)
                    chapter_count += 1

                current_chapter_title = page_start_title
                current_chapter_content = [f"# {page_start_title}"]
                last_split_page = current_page
            continue

        # Check if line is a header (before LaTeX cleanup, to match extracted candidates)
        match = any_header_pattern.match(line)
        is_split_point = False
        heading_text = None
        lookup_title = None
        if match:
            heading_text = match.group(2).strip()
            # Clean LaTeX artifacts for matching
            heading_text = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", heading_text).strip()
            heading_text = re.sub(r"^\d+\s+", "", heading_text).strip()
            # Determine if this heading is a chapter split point
            if confirmed_headings is not None:
                lookup_title = _lookup_confirmed_heading(
                    _confirmed_heading_lookup,
                    current_page,
                    heading_text,
                )
                is_split_point = lookup_title is not None
            else:
                is_split_point = bool(major_header_pattern.match(line))
        elif confirmed_headings is not None and current_page != last_split_page:
            # Detect chapter headings that OCR output as plain text (no # prefix).
            # Skip if we already split on this page (guards against OCR outputting
            # the same heading as both # H1 and plain text on the same page).
            stripped = line.strip()
            if stripped and not stripped.isdigit():
                lookup_title = _lookup_confirmed_heading(
                    _confirmed_heading_lookup,
                    current_page,
                    stripped,
                )
                if lookup_title is not None:
                    heading_text = stripped
                    is_split_point = True

        # Clean up LaTeX artifacts
        line = re.sub(r"\$\s*\^\{(.+?)\}\s*\$", r"", line)  # superscripts
        line = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", line)  # underlines

        if is_split_point:
            # Use the full merged title from confirmed_map when available
            new_title = lookup_title or (match.group(2) if match else heading_text)
            chapter_heading_line = f"# {new_title}"

            # If this split point has the same title as the current chapter,
            # it is a page-boundary duplicate (e.g. page header repeating
            # the chapter title).  Merge instead of splitting.
            if current_chapter_title and new_title == current_chapter_title:
                current_chapter_content.append(chapter_heading_line)
                last_split_page = current_page
                continue

            # Save previous chapter
            if current_chapter_content:
                display_title = current_chapter_title or "Content"
                safe_title = "".join(
                    c for c in display_title
                    if c.isalnum() or c in (" ", "_", "-")
                ).strip()
                if not safe_title:
                    safe_title = f"chap_{chapter_count}"

                c = epub.EpubHtml(
                    title=display_title,
                    file_name=f"{safe_title}_{chapter_count}.xhtml",
                    lang=language,
                )

                try:
                    import markdown
                    html_content = markdown.markdown(
                        "\n".join(current_chapter_content)
                    )
                except ImportError:
                    html_content = (
                        "<p>" + "</p><p>".join(current_chapter_content) + "</p>"
                    )

                c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
                c.add_item(nav_css)
                book.add_item(c)
                chapters.append(c)
                chapter_count += 1

            current_chapter_title = new_title
            current_chapter_content = [chapter_heading_line]
            last_split_page = current_page
        else:
            current_chapter_content.append(line)

    # Add last chapter
    if current_chapter_content:
        # Determine filename
        display_title = current_chapter_title or "Content"
        safe_title = "".join(
            [c for c in display_title if c.isalnum() or c in (" ", "_", "-")]
        ).strip()
        if not safe_title:
            safe_title = f"chap_{chapter_count}"

        c = epub.EpubHtml(
            title=display_title,
            file_name=f"{safe_title}_{chapter_count}.xhtml",
            lang=language,
        )
        try:
            import markdown

            html_content = markdown.markdown("\n".join(current_chapter_content))
        except ImportError:
            html_content = "<p>" + "</p><p>".join(current_chapter_content) + "</p>"

        c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
        c.add_item(nav_css)
        book.add_item(c)
        chapters.append(c)

    # If no chapters were created (no headers found), create one big chapter
    if not chapters and full_markdown:
        c = epub.EpubHtml(title="Content", file_name="content.xhtml", lang=language)
        try:
            import markdown

            html_content = markdown.markdown(full_markdown)
        except ImportError:
            html_content = "<p>" + "</p><p>".join(full_markdown.split("\n")) + "</p>"
        c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
        book.add_item(c)
        chapters.append(c)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters

    write_validated_epub(
        book,
        output_file,
        strict_ocr_validation=strict_ocr_validation,
    )
    print(f"[*] EPUB saved to {output_file}")
