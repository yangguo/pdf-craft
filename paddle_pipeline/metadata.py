"""Cover image extraction, metadata, heading candidates, and TOC review."""

import os
import re

from typing import Any, Dict, List

from .config import (
    DEFAULT_COVER_MAX_EDGE,
    DEFAULT_COVER_JPEG_QUALITY,
    fitz,      # Optional dependency
    requests,  # Optional dependency
)
from .ocr_noise import clean_ocr_noise


def extract_cover_image(
    pdf_path: str,
    output_path: str,
    max_edge: int = DEFAULT_COVER_MAX_EDGE,
    jpg_quality: int = DEFAULT_COVER_JPEG_QUALITY,
) -> str | None:
    """Renders the first PDF page as a bounded JPEG image for use as an EPUB cover."""
    doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            print("[!] PDF has no pages; skipping cover extraction.")
            return None
        page = doc.load_page(0)
        if max_edge <= 0:
            zoom = 2.0
        else:
            zoom = min(2.0, max_edge / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(output_path, jpg_quality=max(1, min(jpg_quality, 100)))
        print(f"[*] Cover image extracted to {output_path}")
        return output_path
    except Exception as e:
        print(f"[!] Failed to extract cover image: {e}")
        return None
    finally:
        doc.close()


def extract_metadata_interactive(results: List[Dict], default_title: str) -> Dict[str, str | None]:
    """Shows the first page OCR text and prompts the user for title and author."""
    first_page_text = ""
    try:
        first_page_text = results[0]["result"]["layoutParsingResults"][0]["markdown"]["text"]
    except (IndexError, KeyError, TypeError):
        pass

    if first_page_text:
        print("\n--- First page OCR text ---")
        print(first_page_text.strip())
        print("----------------------------\n")
    else:
        print("\n[!] Could not extract text from the first page.\n")

    title = input(f"Enter book title (or press Enter to use '{default_title}'): ").strip()
    if not title:
        title = default_title

    author = input("Enter author name (or press Enter to skip): ").strip()

    return {"title": title, "author": author if author else None}


def extract_candidate_headings(results: List[Dict]) -> List[Dict]:
    """Scans API results and extracts candidate chapter headings with page numbers.

    Merges chapter-number H2 headings (e.g. '## 第二十二章') on page N-1 with
    the following H1 title heading on page N (e.g. '# 思潮澎湃...'), producing a
    combined title like '第二十二章 思潮澎湃...'.
    """
    candidates = []
    any_header_pattern = re.compile(r"^(#{1,2})\s+(.+)$")
    latex_pattern = re.compile(r"\$\s*\\underline\{(.+?)\}\s*\$")
    chapter_number_pattern = re.compile(r"^第[零一二三四五六七八九十百千0-9]+章$")
    global_page = 0

    for result in results:
        if not result or "result" not in result:
            continue
        for page_res in result["result"].get("layoutParsingResults", []):
            global_page += 1
            page_md = clean_ocr_noise(page_res["markdown"]["text"])
            for line in page_md.split("\n"):
                if line.strip().isdigit():
                    continue
                match = any_header_pattern.match(line)
                if match:
                    title = match.group(2).strip()
                    # Clean LaTeX artifacts (e.g. "$ \underline{3} $ Title" -> "Title")
                    title = latex_pattern.sub("", title).strip()
                    # Strip leading bare numbers left after LaTeX cleanup (e.g. "3 What Is..." -> "What Is...")
                    title = re.sub(r"^\d+\s+", "", title).strip()
                    candidates.append({
                        "title": title,
                        "page": global_page,
                        "level": len(match.group(1)),
                        "md_line": line,
                    })

    # Merge chapter-number H2 with the following H1 title.
    # PaddleOCR may place them on the same page or adjacent pages.
    merged = []
    i = 0
    while i < len(candidates):
        current = candidates[i]
        if (
            current["level"] == 2
            and chapter_number_pattern.match(current["title"])
            and i + 1 < len(candidates)
        ):
            next_c = candidates[i + 1]
            if next_c["level"] == 1 and next_c["page"] in (
                current["page"],
                current["page"] + 1,
            ):
                merged.append({
                    "title": f"{current['title']} {next_c['title']}",
                    "page": current["page"],
                    "level": 1,
                    "md_line": next_c["md_line"],
                })
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged


def filter_heading_candidates(candidates: List[Dict]) -> List[Dict]:
    """Adaptively filters headings by trying H1-only, then keyword match, then all."""
    chapter_keyword = re.compile(
        r"^(?:Chapter|Part|Lecture|Preface|Intro|Appendix|Prologue|Epilogue|"
        r"Conclusion|Acknowledgements|Contents|Abstract|序|前言|导论|目录|附录|后记|"
        r"第[零一二三四五六七八九十百千0-9]+[篇章节讲])",
        re.IGNORECASE,
    )

    # Strategy 1: H1-only (skip front-matter on first 2 pages)
    h1 = [h for h in candidates if h["level"] == 1 and h["page"] > 2]
    if len(h1) >= 4:
        return h1

    # Strategy 2: Keyword-matched headings (any level)
    keyword_matches = [h for h in candidates if chapter_keyword.match(h["title"])]
    if len(keyword_matches) >= 3:
        return keyword_matches

    # Strategy 3: All headings (last resort)
    return candidates


def review_toc_interactive(candidates: List[Dict], all_candidates: List[Dict] = None) -> List[Dict]:
    """Interactive prompt for reviewing and editing the auto-detected TOC."""
    if not candidates:
        print("\n[!] No chapter headings detected. The book will be a single chapter.")
        return []

    def _print_heading_list(headings):
        print(f"  {'#':>3}  | {'Pg':>4} | Heading")
        print(f"  {'---':>3}--+------+{'-' * 40}")
        for i, h in enumerate(headings):
            indent = "  " if h["level"] == 1 else "    "
            print(f"  {i+1:>3}  | {h['page']:>4} | {indent}{h['title']}")

    def _show_options():
        print()
        print("Options:")
        print("  [Enter]    Accept all headings as chapter split points")
        print("  1,3,5      Remove headings by number (comma-separated)")
        print("  +1,3,5     Keep ONLY these headings (comma-separated)")
        if all_candidates and len(all_candidates) > len(candidates):
            print(f"  all        Show all {len(all_candidates)} headings (including sub-sections)")
        print("  none       No chapters (entire book as single chapter)")
        print()

    print("\n--- Detected Chapter Headings ---")
    _print_heading_list(candidates)
    _show_options()

    while True:
        choice = input("Your choice: ").strip()

        if choice == "":
            return list(candidates)
        elif choice.lower() == "none":
            return []
        elif choice.lower() == "all" and all_candidates and len(all_candidates) > len(candidates):
            candidates = all_candidates
            all_candidates = None
            print(f"\n--- All Headings ({len(candidates)}) ---")
            _print_heading_list(candidates)
            _show_options()
            continue
        else:
            try:
                # Detect keep mode (+) vs remove mode (default)
                keep_mode = choice.startswith("+")
                if keep_mode:
                    choice = choice[1:]  # strip the +

                nums = set()
                valid = True
                for part in choice.split(","):
                    part = part.strip().lstrip("-")  # allow optional - prefix
                    num = int(part)
                    if 1 <= num <= len(candidates):
                        nums.add(num)
                    else:
                        print(f"  [!] Invalid number: {num} (must be 1-{len(candidates)})")
                        valid = False
                        break

                if not valid:
                    continue

                if keep_mode:
                    confirmed = [h for i, h in enumerate(candidates) if (i + 1) in nums]
                else:
                    confirmed = [h for i, h in enumerate(candidates) if (i + 1) not in nums]

                if confirmed:
                    print(f"\nUpdated TOC ({len(confirmed)} chapters):")
                    _print_heading_list(confirmed)
                else:
                    print("\n  All headings removed. Book will be a single chapter.")

                confirm = input("\nConfirm? [Y/n]: ").strip().lower()
                if confirm in ("", "y", "yes"):
                    return confirmed
                else:
                    print("\n--- Detected Chapter Headings ---")
                    _print_heading_list(candidates)
                    _show_options()
                    continue

            except ValueError:
                print("  [!] Invalid input. Use numbers like '1,3,5' or '+1,3,5' for keep mode.")
                continue


def download_image(url: str, save_path: str):
    """Downloads an image from a URL to a local path."""
    try:
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"[!] Failed to download image {url}: {e}")
    return False


