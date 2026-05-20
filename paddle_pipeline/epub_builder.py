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
                for _missing in missing_image_paths:
                    page_md = re.sub(
                        r"!\[[^\]]*\]\(" + re.escape(_missing) + r"\)",
                        "",
                        page_md,
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

            # Reflow: join lines that appear to be part of the same paragraph
            reflowed_paragraphs = []
            current_para = ""

            # Terminal punctuation that indicates end of sentence/paragraph
            # Includes CJK full-width equivalents for Chinese/Japanese OCR.
            terminal_chars = ('.', '!', '?', '"', "'", ')', ']', '}', ':', ';',
                              '。', '！', '？', '：', '；', '」', '』', '）', '】', '〉', '》')
            # Patterns that indicate a line is a header or special block
            header_pattern = re.compile(r'^(#{1,6}\s|>|\s*[-*]\s|\d+\.)')
            # Characters that indicate a line should not be joined with next
            non_join_endings = (':', ';', ',', '-', '：', '；', '，')

            for i, line in enumerate(cleaned_lines):
                stripped = line.rstrip()
                if not stripped:
                    # Empty line indicates paragraph break
                    if current_para:
                        reflowed_paragraphs.append(current_para)
                        current_para = ""
                    continue

                # Check if this line is a header/list/blockquote
                is_special = header_pattern.match(stripped)

                if is_special:
                    # Flush current paragraph before special line
                    if current_para:
                        reflowed_paragraphs.append(current_para)
                        current_para = ""
                    reflowed_paragraphs.append(stripped)
                elif not current_para:
                    # Start of new paragraph
                    current_para = stripped
                else:
                    # Check if current_para ends with terminal punctuation
                    # or if it ends with colon/comma/semicolon (likely list/address item)
                    current_ends = current_para.rstrip()

                    if current_ends.endswith(terminal_chars):
                        # Previous paragraph ends with terminal punctuation
                        # This line starts a new paragraph
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif current_ends and current_ends[-1] in non_join_endings:
                        # Previous line ends with :, ;, , or - - likely list item or address
                        # Don't join, start new paragraph
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif len(stripped) < 20 and not stripped.endswith(terminal_chars):
                        # Short line that doesn't end with punctuation - might be
                        # a title, caption, or deliberate short line (poetry, etc.)
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    else:
                        # Likely continuation of same paragraph
                        current_para = current_ends + " " + stripped

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
                next_is_header = header_pattern.match(first_page_line) if first_page_line else False

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

    # Build a dict mapping (page, heading_text) → full_title so split-point
    # detection can use exact lookups like the original, while merged titles
    # (e.g. '第二十二章 思潮澎湃...') still resolve correctly.
    _page_marker_re = re.compile(r"^\x00PAGE:(\d+)\x00$")
    _chapter_num_re = re.compile(r"^(第[零一二三四五六七八九十百千0-9]+章)\s+")
    confirmed_map: Dict[tuple[int, str], str] = {}
    if confirmed_headings is not None:
        for h in confirmed_headings:
            title = h["title"]
            page = h["page"]
            confirmed_map[(page, title)] = title
            # For merged titles like '第二十二章 思潮澎湃...', also register
            # the bare chapter-number component (e.g. '第二十二章') so that the
            # H2 number heading triggers the chapter split with the full title.
            cn_match = _chapter_num_re.match(title)
            if cn_match:
                confirmed_map[(page, cn_match.group(1))] = title
    else:
        confirmed_map = {}  # use legacy regex fallback

    current_page = 0

    for line in md_lines:
        # Consume page boundary markers (injected during markdown assembly)
        _pm = _page_marker_re.match(line)
        if _pm:
            current_page = int(_pm.group(1))
            continue

        # Check if line is a header (before LaTeX cleanup, to match extracted candidates)
        match = any_header_pattern.match(line)
        is_split_point = False
        if match:
            heading_text = match.group(2).strip()
            # Clean LaTeX artifacts for matching
            heading_text = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", heading_text).strip()
            heading_text = re.sub(r"^\d+\s+", "", heading_text).strip()
            # Determine if this heading is a chapter split point
            if confirmed_headings is not None:
                lookup_title = confirmed_map.get((current_page, heading_text))
                if lookup_title is None:
                    lookup_title = confirmed_map.get((current_page - 1, heading_text))
                is_split_point = lookup_title is not None
            else:
                is_split_point = bool(major_header_pattern.match(line))

        # Clean up LaTeX artifacts
        line = re.sub(r"\$\s*\^\{(.+?)\}\s*\$", r"", line)  # superscripts
        line = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", line)  # underlines

        if match:
            if is_split_point:
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

                # Use the full merged title from confirmed_map when available
                current_chapter_title = lookup_title or match.group(2)
                current_chapter_content = [line]
            else:
                # Minor header or non-split heading — keep it in flow
                current_chapter_content.append(line)
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


