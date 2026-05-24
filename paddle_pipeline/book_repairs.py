"""Book-specific OCR repairs applied before TOC detection and EPUB assembly.

TODO: Consolidate duplicate helper functions (_flatten_layout_pages, _render_page_png)
when integrating book_repairs into the main pipeline. These are copy-pasted from
page_order_repair.py and page_image_fallback.py and should be extracted to a shared
utilities module to avoid maintenance burden.
"""

import os

from typing import Any, Dict, List

from .config import fitz  # Optional dependency


_HONG2_VISUAL_PAGES = {
    17: "地圖",
    18: "地圖",
    19: "年表",
    20: "年表",
    21: "年表",
    22: "年表",
    23: "年表",
}

_HONG2_PHOTO_CROP_PAGES = {
    36: {
        "source_size": (761, 1169),
        "photos": (
            ((0, 1, 761, 536), "imgs/hong2_page_0036_photo_01.png", "插圖 1"),
            ((52, 609, 647, 1097), "imgs/hong2_page_0036_photo_02.png", "插圖 2"),
        ),
    },
}

_HONG2_OPENING_HEADINGS = {
    42: (2, "# 一、三寸金蓮——嫁給軍閥為妾（一九○九—一九三三年）"),
    59: (2, "# 二、「喝涼水也是甜的」——成為滿族醫生的妻子（一九三三—一九三八年）"),
    74: (2, "# 三、「人人都說好滿洲」——在日本人統治下（一九三八—一九四五年）"),
    86: (2, "# 四、「亡國奴」——走馬燈似的換政府（一九四五—一九四七年）"),
    100: (1, "# 五、「十歲女兒，十公斤大米」——為中國的前途而戰（一九四七—一九四八年）"),
    118: (2, "# 六、「談戀愛」——革命的婚姻（一九四八—一九四九年）"),
    138: (1, "# 七、「過五關」——我母親的長征（一九四九—一九五○年）"),
    146: (2, "# 八、「衣錦還鄉」——歸故里，遭逢土匪（一九五○—一九五一年）"),
    162: (2, "# 九、「一人得道，雞犬升天」——與清官共同生活（一九五一—一九五三年）"),
    180: (1, "# 十、「磨難會使妳成為真正的共產黨員」——我母親受審查（一九五三—一九五六年）"),
    190: (2, "# 十一、「反右以後莫發言」——中國沉默了（一九五六—一九五八年）"),
    204: (1, "# 十二、「巧婦能為無米炊」——大饑荒（一九五八—一九六二年）"),
    222: (2, "# 十三、「千金小姐」——我的世界（一九五八—一九六五年）"),
    236: (1, "# 十四、「爹親娘親，不如毛主席親」——對毛澤東的個人崇拜（一九六四—一九六五年）"),
    252: (2, "# 十五、「破字當頭，立在其中」——文化大革命開始（一九六五—一九六六年）"),
    260: (2, "# 十六、「天不怕，地不怕」——毛的紅衛兵（一九六六年六月—八月）"),
    274: (1, "# 十七、「你要我們的孩子變成黑五類嗎？」——父母進退兩難（一九六六年八月—十月）"),
    284: (2, "# 十八、「特大喜訊」——進京朝聖（一九六六年十月—十二月）"),
    298: (2, "# 十九、「欲加之罪，何患無辭」——父母受折磨（一九六六年十二月—一九六七年）"),
    314: (2, "# 二十、「我不出賣靈魂」——父親被捕（一九六七—一九六八年）"),
    334: (2, "# 二十一、「雪中送炭」——姐弟們、朋友們（一九六七—一九六八年）"),
    350: (2, "# 二十二、「勞動改造」——到喜馬拉雅山邊去（一九六九年一月—六月）"),
    376: (2, "# 二十三、「書讀得越多越蠢」——我當農民，也當赤腳醫生（一九六九年六月—一九七一年）"),
    398: (1, "# 二十四、「容我朝暮謝過，以贖前愆」——父母在幹校（一九六九—一九七二年）"),
    412: (3, "# 二十五、「香風味」——與《電工手册》、《六次危機》為伴的新生活（一九七二—一九七三年）"),
    424: (2, "# 二十六、「外國人放個屁都是香的」——在毛澤東治下學英語（一九七二—一九七四年）"),
    440: (1, "# 二十七、「如果這是天堂，地獄又是什麼樣子呢？」——父親之死（一九七四—一九七六年）"),
    458: (1, "# 二十八、長上翅膀飛——（一九七六—一九七八年）"),
}

_HONG2_TEXT_REPLACEMENTS = (
    (
        2,
        "# 江 三代中國女人的故事 張戎·著／張樸·譯",
        "鴻——三代中國女人的故事\n\n張戎·著／張樸·譯",
    ),
    (
        8,
        "# 江 三代中國女人的故事 張戎·著／張樸·譯",
        "鴻——三代中國女人的故事\n\n張戎·著／張樸·譯",
    ),
    (40, "## 江", "## 鴻"),
    (
        250,
        "此外，努力工作還有個更重",
        "此外，努力工作還有個更重要的目的——當上勞動模範去北京見毛主席，這成了我生活的目標。",
    ),
    (
        259,
        "就是在這種氣氛裡，八月份來臨了，百萬紅衛兵乍現，似狂風暴雨般席捲了整個中國",
        "就是在這種氣氛裡，八月份來臨了，百萬紅衛兵乍現，似狂風暴雨般席捲了整個中國。",
    ),
    (
        283,
        "她是假紅衛兵之手整那個可憐的女人，紅衛兵被利用",
        "她是假紅衛兵之手整那個可憐的女人，紅衛兵被利用來算舊帳、洩私憤。我爬上卡車，滿心是厭惡和狂怒。",
    ),
    (470, "# 覈刖訏諡莕菶", "## 報刊評論薈萃"),
)


def _flatten_layout_pages(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages = []
    for result in results:
        pages.extend(result.get("result", {}).get("layoutParsingResults", []))
    return pages


def _is_hong2_book(pdf_path: str, pages: List[Dict[str, Any]]) -> bool:
    basename = os.path.basename(pdf_path).lower()
    if basename == "hong2.pdf":
        return True

    front_text = "\n".join(
        page.get("markdown", {}).get("text", "")
        for page in pages[:20]
    )
    return "三代中國女人" in front_text and "張戎" in front_text


def _render_page_png(pdf_path: str, page_number: int, output_path: str,
                     zoom: float = 2.0) -> None:
    if fitz is None:
        raise RuntimeError("pymupdf is required for known book repair rendering")

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pix.save(output_path)
    finally:
        doc.close()


def _render_page_crop_png(pdf_path: str, page_number: int,
                          source_box: tuple[int, int, int, int],
                          source_size: tuple[int, int], output_path: str,
                          zoom: float = 2.0) -> None:
    if fitz is None:
        raise RuntimeError("pymupdf is required for known book repair rendering")

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        page_width = getattr(page.rect, "width", source_size[0])
        page_height = getattr(page.rect, "height", source_size[1])
        source_width, source_height = source_size
        x0, y0, x1, y1 = source_box
        clip = fitz.Rect(
            x0 * page_width / source_width,
            y0 * page_height / source_height,
            x1 * page_width / source_width,
            y1 * page_height / source_height,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pix.save(output_path)
    finally:
        doc.close()


def _replace_page_with_image(pdf_path: str, page_res: Dict[str, Any],
                             page_number: int, image_dir: str,
                             alt_text: str) -> int:
    markdown = page_res.setdefault("markdown", {})
    rel_path = f"imgs/hong2_page_{page_number:04d}.png"
    local_path = os.path.join(image_dir, rel_path)
    html = (
        f'<div style="text-align: center;"><img src="{rel_path}" '
        f'alt="{alt_text}" width="100%" /></div>'
    )

    if not os.path.exists(local_path):
        _render_page_png(pdf_path, page_number, local_path)

    changed = markdown.get("text") != html or markdown.get("images") != {rel_path: ""}
    markdown["text"] = html
    markdown["images"] = {rel_path: ""}
    return 1 if changed else 0


def _replace_page_with_photo_crops(pdf_path: str, page_res: Dict[str, Any],
                                   page_number: int, image_dir: str,
                                   spec: Dict[str, Any]) -> int:
    markdown = page_res.setdefault("markdown", {})
    images: Dict[str, str] = {}
    html_blocks = []
    source_size = spec["source_size"]

    for source_box, rel_path, alt_text in spec["photos"]:
        local_path = os.path.join(image_dir, rel_path)
        if not os.path.exists(local_path):
            _render_page_crop_png(
                pdf_path,
                page_number,
                source_box,
                source_size,
                local_path,
            )
        images[rel_path] = ""
        html_blocks.append(
            f'<div style="text-align: center;"><img src="{rel_path}" '
            f'alt="{alt_text}" width="100%" /></div>'
        )

    text = "\n\n".join(html_blocks)
    changed = markdown.get("text") != text or markdown.get("images") != images
    markdown["text"] = text
    markdown["images"] = images
    return 1 if changed else 0


def _replace_leading_nonempty_lines(text: str, replacement: str,
                                    drop_nonempty: int) -> tuple[str, bool]:
    if text.lstrip().startswith(replacement):
        return text, False

    lines = text.split("\n")
    index = 0
    dropped = 0
    while index < len(lines) and dropped < drop_nonempty:
        if lines[index].strip():
            dropped += 1
        index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    rest = "\n".join(lines[index:]).lstrip("\n")
    if rest:
        return f"{replacement}\n\n{rest}", True
    return replacement, True


def _apply_opening_heading(page_res: Dict[str, Any], replacement: str,
                           drop_nonempty: int) -> int:
    markdown = page_res.setdefault("markdown", {})
    text = markdown.get("text", "")
    new_text, changed = _replace_leading_nonempty_lines(
        text,
        replacement,
        drop_nonempty,
    )
    if changed:
        markdown["text"] = new_text
        return 1
    return 0


def _replace_text(page_res: Dict[str, Any], old: str, new: str) -> int:
    markdown = page_res.setdefault("markdown", {})
    text = markdown.get("text", "")
    if new in text:
        return 0
    if old not in text:
        return 0
    markdown["text"] = text.replace(old, new, 1)
    return 1


def _prepend_heading(page_res: Dict[str, Any], heading: str) -> int:
    markdown = page_res.setdefault("markdown", {})
    text = markdown.get("text", "")
    if text.lstrip().startswith(heading):
        return 0
    markdown["text"] = f"{heading}\n\n{text.lstrip()}"
    return 1


def apply_known_book_repairs(pdf_path: str, results: List[Dict[str, Any]],
                             image_dir: str) -> int:
    """Apply deterministic repairs for known OCR-problem scans."""
    pages = _flatten_layout_pages(results)
    if not _is_hong2_book(pdf_path, pages):
        return 0

    repairs = 0

    for page_number, alt_text in _HONG2_VISUAL_PAGES.items():
        if page_number <= len(pages):
            repairs += _replace_page_with_image(
                pdf_path,
                pages[page_number - 1],
                page_number,
                image_dir,
                alt_text,
            )

    for page_number, spec in _HONG2_PHOTO_CROP_PAGES.items():
        if page_number <= len(pages):
            repairs += _replace_page_with_photo_crops(
                pdf_path,
                pages[page_number - 1],
                page_number,
                image_dir,
                spec,
            )

    for page_number, (drop_nonempty, replacement) in _HONG2_OPENING_HEADINGS.items():
        if page_number <= len(pages):
            repairs += _apply_opening_heading(
                pages[page_number - 1],
                replacement,
                drop_nonempty,
            )

    for page_number, old, new in _HONG2_TEXT_REPLACEMENTS:
        if page_number <= len(pages):
            repairs += _replace_text(pages[page_number - 1], old, new)

    if 468 <= len(pages):
        repairs += _prepend_heading(pages[467], "# 跋")

    if repairs:
        print(f"[*] Applied {repairs} known-book repair(s)")
    return repairs
