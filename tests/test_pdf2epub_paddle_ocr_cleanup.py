import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile
from unittest import mock


def _load_script_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paddle_pipeline
    return paddle_pipeline


def _load_fresh_script_module():
    for module_name in list(sys.modules):
        if module_name == "paddle_pipeline" or module_name.startswith("paddle_pipeline."):
            del sys.modules[module_name]
    return _load_script_module()


class TestPdf2EpubPaddleOcrCleanup(unittest.TestCase):
    def test_clean_ocr_noise_preserves_display_array_and_keeps_prose(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            "# 中国要变什么？\n\n"
            r"$$ \begin{array}{ccc} 2 & 3 & 4 \\ 5 & 6 & 7 \\ 8 & 9 & 10 \end{array} $$"
            "\n\n过去学术界习惯把鸦片战争作为近代中国之起点。"
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("# 中国要变什么？", cleaned)
        self.assertIn("过去学术界习惯", cleaned)
        self.assertIn(r"\begin{array}", cleaned)
        self.assertIn("2 & 3 & 4", cleaned)
        self.assertIn(r"\end{array}", cleaned)

    def test_clean_ocr_noise_preserves_valid_aligned_equations(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            "Before equation.\n\n"
            r"$$\begin{aligned} a &= b + c \\ d &= e + f \end{aligned}$$"
            "\n\nAfter equation."
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Before equation.", cleaned)
        self.assertIn(r"\begin{aligned}", cleaned)
        self.assertIn("a &= b + c", cleaned)
        self.assertIn(r"\end{aligned}", cleaned)
        self.assertIn("After equation.", cleaned)

    def test_clean_ocr_noise_preserves_prose_after_unclosed_array_marker(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            "Intro paragraph.\n"
            r"\[\begin{array}{c} noisy ornament"
            "\n\nThis prose should remain in the chapter."
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Intro paragraph.", cleaned)
        self.assertIn("This prose should remain in the chapter.", cleaned)

    def test_clean_ocr_noise_normalizes_inline_false_latex(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = (
            r"20 $ \frac{1}{2} $7年，香港三联书店出版。"
            r"海关内部衍生了邮政、海巡 $ \downarrow $气象、天文等多个部门。"
            r"不惜使用相当夸张的说法 $ \underset{\cdot}{颂} $圣。"
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("2017年，香港三联书店出版", cleaned)
        self.assertIn("海巡、气象、天文", cleaned)
        self.assertIn("颂圣", cleaned)
        self.assertNotIn("\\frac", cleaned)
        self.assertNotIn("\\downarrow", cleaned)
        self.assertNotIn("\\underset", cleaned)

    def test_clean_ocr_noise_preserves_isolated_downarrow_math(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        raw = r"The diagram uses the symbol $ \downarrow $ between two nodes."

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn(r"$ \downarrow $", cleaned)
        self.assertNotIn("、", cleaned)

    def test_clean_ocr_noise_removes_numeric_only_year_table_from_blank_page(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        rows = []
        for year in range(2021, 2045):
            rows.append(
                "<tr>"
                f"<td>{year}</td><td>{year + 1}</td><td>{year + 2}</td>"
                "</tr>"
            )
        raw = "Before.\n\n<table>" + "".join(rows) + "</table>\n\nAfter."

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Before.", cleaned)
        self.assertIn("After.", cleaned)
        self.assertNotIn("<table", cleaned)
        self.assertNotIn("2044", cleaned)

    def test_clean_ocr_noise_removes_dotted_numeric_table_from_blank_page(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "clean_ocr_noise"))

        cells = "".join(
            f"<td>1.1.1.{index}</td>"
            for index in range(377, 397)
        )
        raw = (
            "<p>Before.</p>"
            "<table><tr><td>序：章之三</td>"
            + cells
            + "</tr></table>"
            "<p>After.</p>"
        )

        cleaned = mod.clean_ocr_noise(raw)

        self.assertIn("Before.", cleaned)
        self.assertIn("After.", cleaned)
        self.assertNotIn("<table", cleaned)
        self.assertNotIn("1.1.1.382", cleaned)

    def test_extract_page_footnotes_keeps_ocr_footnote_blocks_only(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "extract_page_footnotes"))

        page_res = {
            "prunedResult": {
                "parsing_res_list": [
                    {"block_label": "text", "block_content": "正文"},
                    {
                        "block_label": "footnote",
                        "block_content": " $ ^{14} $ Ping-ti Ho, 189.",
                    },
                    {"block_label": "footnote", "block_content": "  "},
                    {
                        "block_label": "vision_footnote",
                        "block_content": "15 另一条脚注。",
                    },
                ]
            }
        }

        footnotes = mod.extract_page_footnotes(page_res)

        self.assertEqual(["$ ^{14} $ Ping-ti Ho, 189.", "15 另一条脚注。"], footnotes)

    def test_extract_page_footnotes_infers_unnumbered_note_from_inline_marker(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "extract_page_footnotes"))

        page_res = {
            "markdown": {
                "text": "正文 $ ^{9} $ 继续 $ ^{10} $ 结束。",
            },
            "prunedResult": {
                "parsing_res_list": [
                    {"block_label": "footnote", "block_content": "Leonard S. Hsü, 62-63."},
                    {"block_label": "footnote", "block_content": "$ ^{10} $ 即杨文炳。"},
                ]
            },
        }

        footnotes = mod.extract_page_footnotes(page_res)

        self.assertEqual(
            ["$ ^{9} $ Leonard S. Hsü, 62-63.", "$ ^{10} $ 即杨文炳。"],
            footnotes,
        )

    def test_format_page_footnotes_html_renders_numbered_notes(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "format_page_footnotes_html"))

        html_text = mod.format_page_footnotes_html(
            ["$ ^{14} $ Ping-ti Ho, 189.", "15 另一条脚注。"],
            page_number=42,
        )

        self.assertIn('class="page-footnotes"', html_text)
        self.assertIn('data-source-page="42"', html_text)
        self.assertIn("<sup>14</sup> Ping-ti Ho, 189.", html_text)
        self.assertIn("<sup>15</sup> 另一条脚注。", html_text)

    def test_link_page_footnote_references_connects_matching_notes(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "link_page_footnote_references"))

        linked_text, footnote_refs = mod.link_page_footnote_references(
            "正文 $ ^{14} $ 继续 $ ^{99} $。",
            ["$ ^{14} $ Ping-ti Ho, 189.", "15 另一条脚注。"],
            page_number=42,
        )
        html_text = mod.format_page_footnotes_html(
            ["$ ^{14} $ Ping-ti Ho, 189.", "15 另一条脚注。"],
            page_number=42,
            footnote_refs=footnote_refs,
        )

        self.assertIn('id="fnref-p42-14"', linked_text)
        self.assertIn('epub:type="noteref"', linked_text)
        self.assertIn('href="#fn-p42-14"', linked_text)
        self.assertIn('class="unlinked-footnote-marker"', linked_text)
        self.assertIn('<p id="fn-p42-14" class="footnote">', html_text)
        self.assertIn('epub:type="backlink"', html_text)
        self.assertIn('href="#fnref-p42-14"', html_text)
        self.assertNotIn('id="fn-p42-15"', html_text)

    def test_toc_page_start_css_keeps_part_and_chapter_together(self):
        mod = _load_script_module()

        self.assertIn(
            "p.toc-page-start + p + h1.toc-page-start",
            mod.TOC_PAGE_START_CSS,
        )
        self.assertIn("break-before: auto", mod.TOC_PAGE_START_CSS)

    def test_scan_epub_for_ocr_noise_reports_specific_false_latex_artifacts(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>$ \underset{\cdot}{颂} $圣</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(any(item["file"] == "EPUB/chapter.xhtml" for item in findings))
        self.assertTrue(any(item["token"] == "\\underset{\\cdot}" for item in findings))

    def test_scan_epub_for_ocr_noise_reports_numeric_only_year_table(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        rows = []
        for year in range(2021, 2045):
            rows.append(
                "<tr>"
                f"<td>{year}</td><td>{year + 1}</td><td>{year + 2}</td>"
                "</tr>"
            )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><table>"
                    + "".join(rows)
                    + "</table></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(
            any(item["token"] == "numeric-only OCR table" for item in findings)
        )

    def test_scan_epub_for_ocr_noise_reports_dotted_numeric_table(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        cells = "".join(
            f"<td>1.1.1.{index}</td>"
            for index in range(377, 397)
        )

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><table><tr><td>序：章之三</td>"
                    + cells
                    + "</tr></table></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(
            any(item["token"] == "dotted numeric OCR table" for item in findings)
        )

    def test_scan_epub_for_ocr_noise_allows_valid_math_latex(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "math.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>Area is $a \times b$ and half is $\frac{1}{2}$.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertEqual([], findings)

    def test_scan_epub_for_ocr_noise_allows_valid_downarrow_math(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "math.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>The sequence satisfies $a_n \downarrow 0$.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertEqual([], findings)

    def test_validate_epub_no_ocr_noise_allows_isolated_downarrow_in_strict_mode(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "validate_epub_no_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "math.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>The symbol is $ \downarrow $.</p></body></html>",
                )

            findings = mod.validate_epub_no_ocr_noise(epub_path, strict=True)

        self.assertEqual([], findings)

    def test_scan_epub_for_ocr_noise_ignores_structural_metadata_files(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "metadata.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    r"<metadata><dc:title>Arrow $ \downarrow $ Theory</dc:title></metadata>",
                )
                zf.writestr(
                    "EPUB/toc.ncx",
                    r"<navLabel><text>Arrow $ \downarrow $ Theory</text></navLabel>",
                )
                zf.writestr(
                    "EPUB/nav.xhtml",
                    r"<html><body><nav>Arrow $ \downarrow $ Theory</nav></body></html>",
                )
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><p>Clean OCR chapter content.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertEqual([], findings)

    def test_scan_epub_for_ocr_noise_detects_garbled_cjk_text(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        # The detector is intentionally conservative: it should only report
        # long runs where nearly all bigrams are novel.
        cjk_start = 0x4E00
        garbled_run = "".join(chr(cjk_start + i) for i in range(60))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>Normal intro text. {garbled_run} Normal outro.</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        self.assertTrue(
            any("garbled CJK" in item.get("token", "") for item in findings),
            f"Expected garbled CJK finding, got: {findings}",
        )

    def test_scan_epub_for_ocr_noise_allows_normal_chinese_prose(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        # The self-calibrating detector builds a bigram model from the full
        # book text.  A short passage would have too many singleton bigrams,
        # so we repeat the prose enough times to simulate a book-length
        # corpus where common character transitions reappear.
        normal_passage = (
            "今天天氣很好，我和朋友一起去公園散步。公園裡有很多花草樹木，"
            "還有一些小朋友在玩耍。我們找了一張長椅坐下來，欣賞周圍的風景。"
            "朋友說他最近工作很忙，很少有時間出來走走。我告訴他要多注意休息，"
            "不要總是加班到很晚。身體健康比什麼都重要，這是大家都知道的道理。"
        )
        normal_text = (normal_passage + "\n") * 20

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>{normal_text}</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        garbled_findings = [
            item for item in findings if "garbled CJK" in item.get("token", "")
        ]
        self.assertEqual(
            [], garbled_findings,
            f"Normal Chinese should not trigger garbled detection, got: {garbled_findings}",
        )

    def test_scan_epub_for_ocr_noise_allows_single_one_off_chinese_sentence(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        # Use a large, varied text corpus (≈5K CJK chars) so the self-calibrating
        # bigram model has enough coverage for common character transitions.
        # Real books have 100K+ CJK chars; a single unique sentence should not
        # trigger garbled detection against that background.
        varied_corpus = (
            "今天天氣很好，我和朋友一起去公園散步。公園裡有很多花草樹木，"
            "還有一些小朋友在玩耍。我們找了一張長椅坐下來，欣賞周圍的風景。"
            "朋友說他最近工作很忙，很少有時間出來走走。我告訴他要多注意休息，"
            "不要總是加班到很晚。身體健康比什麼都重要，這是大家都知道的道理。"
            "遠處傳來鳥兒的叫聲，讓人感到心情愉悅。春天裡的花朵爭奇鬥艷，"
            "吸引了不少遊客駐足拍照。湖面上的小船輕輕搖晃，水面泛起陣陣漣漪。"
            "午後的陽光透過樹葉灑落下來，在地面上形成斑駁的光影交錯。"
            "我們沿著小路繼續往前走，路邊的攤販正在叫賣各種小吃和飲料。"
            "城市裡的高樓大廈在夕陽下閃閃發光，顯得格外壯觀美麗。"
            "回到家中，我泡了一杯熱茶，坐在沙發上回想今天的所見所聞，"
            "覺得生活雖然忙碌，但還是充滿了許多美好的時刻值得珍惜。"
            "中國的經濟發展在過去幾十年取得了舉世矚目的成就，從農業社會"
            "轉型為工業大國，數億人口脫離了貧困。基礎設施建設日新月異，"
            "高速鐵路網絡遍布全國，城市地鐵系統不斷擴張。科技創新能力"
            "顯著提升，在人工智能和電子商務領域處於世界領先地位。"
            "教育事業蓬勃發展，大學數量不斷增加，科研論文發表量位居全球前列。"
            "醫療衛生條件大幅改善，人均預期壽命從建國初期的三十五歲提高到"
            "如今的七十七歲以上。社會保障體系逐步完善，養老保險和醫療保險"
            "覆蓋率持續擴大。文化產業欣欣向榮，電影票房屢創新高。"
            "國際交流日益頻繁，留學生人數不斷增長。對外貿易規模持續擴大，"
            "已成為世界第一大貿易國。人民幣國際化進程穩步推進，越來越多"
            "國家將人民幣納入外匯儲備。一帶一路倡議促進了沿線國家的"
            "基礎設施建設和經貿合作。亞洲基礎設施投資銀行的成立為區域發展"
            "注入了新的動力。金磚國家合作機制不斷深化，新興市場國家在國際"
            "事務中的發言權日益增強。國內消費市場蓬勃發展，電子商務和移動支付"
            "普及率位居世界前列。新能源汽車產業快速崛起，光伏發電和風力發電"
            "裝機容量持續增長。互聯網用戶規模超過十億，數字經濟佔國內生產總值"
            "的比重不斷提高。傳統文化傳承創新並舉，非物質文化遺產保護工作"
            "取得顯著成效。文學藝術創作繁榮，優秀作品不斷湧現。體育事業"
            "健康發展，競技體育和群眾體育協調推進。生態文明建設力度加大，"
            "藍天保衛戰取得階段性成果。空氣質量持續改善，森林覆蓋率穩步提高。"
            "改革開放是決定當代中國命運的關鍵一招，也是實現中華民族偉大復興的"
            "必由之路。堅持和完善社會主義市場經濟體制，使市場在資源配置中"
            "起決定性作用，更好發揮政府作用。全面依法治國深入推進，"
            "中國特色社會主義法律體系不斷完善。黨的建設新的偉大工程全面推進，"
            "反腐敗鬥爭取得壓倒性勝利。外交工作堅持獨立自主的和平外交政策，"
            "推動構建人類命運共同體。國防和軍隊現代化建設取得重大進展，"
            "維護國家主權和領土完整的能力顯著增強。鄉村振興戰略全面實施，"
            "農業農村現代化步伐加快。區域協調發展戰略深入推進，東中西部地區"
            "發展差距逐步縮小。城鎮化水平不斷提高，城市群和都市圈建設加快。"
        )
        # Use a sentence whose character bigrams overlap with the corpus above,
        # so it doesn't look like a run of singletons.  "經濟發展帶來社會進步"
        # re-uses bigrams from "中國的經濟發展" and "社會保障體系" in the corpus.
        one_off_sentence = "經濟發展帶來社會進步人民生活水平不斷提高"
        # Repeat corpus 4× so bigrams gain statistical power.
        normal_text = ((varied_corpus + "\n") * 4).rstrip("\n") + "\n" + one_off_sentence

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>{normal_text}</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        garbled_findings = [
            item for item in findings if "garbled CJK" in item.get("token", "")
        ]
        self.assertEqual(
            [], garbled_findings,
            f"One normal one-off sentence should not trigger garbled detection, got: {garbled_findings}",
        )

    def test_scan_epub_for_ocr_noise_garbled_detection_ignores_short_text(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "scan_epub_for_ocr_noise"))

        # Fewer than window_size (12) CJK chars → cannot even form one window.
        short_garbled = "九戎所均勞"

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>{short_garbled}</p></body></html>",
                )

            findings = mod.scan_epub_for_ocr_noise(epub_path)

        garbled_findings = [
            item for item in findings if "garbled CJK" in item.get("token", "")
        ]
        self.assertEqual(
            [], garbled_findings,
            f"Short text should not trigger garbled detection, got: {garbled_findings}",
        )

    def test_validate_epub_no_ocr_noise_reports_truncated_finding_count(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "validate_epub_no_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                for index in range(9):
                    zf.writestr(
                        f"EPUB/chapter-{index}.xhtml",
                        r"<html><body><p>$ \underset{\cdot}{颂} $圣</p></body></html>",
                    )

            with self.assertRaises(RuntimeError) as cm:
                mod.validate_epub_no_ocr_noise(epub_path, strict=True)

        self.assertIn("showing 8 of 9", str(cm.exception))

    def test_validate_epub_no_ocr_noise_warns_without_failing_by_default(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "validate_epub_no_ocr_noise"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    r"<html><body><p>$ \underset{\cdot}{颂} $圣</p></body></html>",
                )

            with mock.patch("builtins.print") as mock_print:
                findings = mod.validate_epub_no_ocr_noise(epub_path)

        self.assertTrue(findings)
        self.assertTrue(any("[!]" in str(call.args[0]) for call in mock_print.call_args_list))

    def test_resolve_epub_fragment_href_keeps_same_file_fragment_in_source_file(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "_resolve_epub_fragment_href"))

        resolved = mod._resolve_epub_fragment_href("EPUB/nav.xhtml", "#toc-header")

        self.assertEqual(("EPUB/nav.xhtml", "toc-header"), resolved)

    def test_ensure_toc_targets_start_pages_marks_only_toc_heading_targets(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol><li>"
                        "<a href=\"chapter.xhtml#section-1\">Section</a>"
                        "</li></ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><head><link href=\"style/nav.css\" rel=\"stylesheet\" "
                        "type=\"text/css\"/></head><body>"
                        "<h2 id=\"intro\">Intro</h2>"
                        "<p>Before.</p>"
                        "<h2 id=\"section-1\">Section</h2>"
                        "<p>After.</p>"
                        "</body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                infos = zf.infolist()
                css = zf.read("EPUB/style/nav.css").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertEqual("mimetype", infos[0].filename)
        self.assertEqual(ZIP_STORED, infos[0].compress_type)
        self.assertIn(".toc-page-start", css)
        self.assertIn("break-before: page", css)
        self.assertIn('<h2 id="section-1" class="toc-page-start">Section</h2>', chapter)
        self.assertIn('<h2 id="intro">Intro</h2>', chapter)

    def test_ensure_toc_targets_start_pages_preserves_existing_heading_class(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/toc.ncx",
                    '<ncx><navMap><navPoint><content src="chapter.xhtml#section-1"/>'
                    '</navPoint></navMap></ncx>',
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><h2 class=\"chapter-title\" id=\"section-1\">"
                        "Section</h2></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertIn(
            '<h2 class="chapter-title toc-page-start" id="section-1">Section</h2>',
            chapter,
        )

    def test_ensure_toc_targets_start_pages_moves_chapter_link_to_number_heading(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol><li>"
                        "<a href=\"chapter.xhtml#title\">第五章 對外關係</a>"
                        "</li></ol></nav></body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/toc.ncx",
                    (
                        "<ncx><navMap><navPoint><navLabel><text>第五章 對外關係"
                        "</text></navLabel><content src=\"chapter.xhtml#title\"/>"
                        "</navPoint></navMap></ncx>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><p>Before.</p>"
                        "<h2 id=\"chapter-number\">第五章</h2>"
                        "<h2 id=\"title\" class=\"toc-page-start\">對外關係</h2>"
                        "<p>After.</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                ncx = zf.read("EPUB/toc.ncx").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertIn('href="chapter.xhtml#chapter-number"', nav)
        self.assertIn('src="chapter.xhtml#chapter-number"', ncx)
        self.assertIn('<h2 id="chapter-number" class="toc-page-start">第五章</h2>', chapter)
        self.assertIn('<h2 id="title">對外關係</h2>', chapter)

    def test_ensure_toc_targets_start_pages_moves_part_link_to_part_line(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"chapter.xhtml#title\">第一編 傳統制度的延續，1600-1800年</a></li>"
                        "<li><a href=\"chapter.xhtml#title\">第二章 清帝國的興盛</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><p>第一編</p><p>傳統制度的延續</p>"
                        "<p>1600-1800年</p><h2 id=\"title\">清帝國的興盛</h2>"
                        "<p>After.</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        part_href = re.search(r'href="chapter.xhtml#([^"]+)">第一編', nav)
        chapter_href = re.search(r'href="chapter.xhtml#([^"]+)">第二章', nav)
        self.assertIsNotNone(part_href)
        self.assertIsNotNone(chapter_href)
        self.assertNotEqual("title", part_href.group(1))
        self.assertNotEqual("title", chapter_href.group(1))
        self.assertIn(
            f'<p id="{part_href.group(1)}" class="toc-page-start">第一編</p>',
            chapter,
        )
        self.assertIn(
            f'<h2 id="{chapter_href.group(1)}" class="toc-page-start">第二章</h2>',
            chapter,
        )
        self.assertIn('<h2 id="title">清帝國的興盛</h2>', chapter)

    def test_ensure_toc_targets_start_pages_inserts_missing_part_heading(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"chapter.xhtml#title\">第三編 外國帝國主義加劇時期的自強運動 1861-1895年</a></li>"
                        "<li><a href=\"chapter.xhtml#title\">第十一章 清朝中興與自強運動</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><h1 id=\"title\">清朝中興與自強運動</h1>"
                        "<p>After.</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        part_href = re.search(r'href="chapter.xhtml#([^"]+)">第三編', nav)
        chapter_href = re.search(r'href="chapter.xhtml#([^"]+)">第十一章', nav)
        self.assertIsNotNone(part_href)
        self.assertIsNotNone(chapter_href)
        self.assertNotEqual("title", part_href.group(1))
        self.assertNotEqual("title", chapter_href.group(1))
        self.assertLess(chapter.index("第三編"), chapter.index("第十一章"))
        self.assertIn(
            f'<p id="{part_href.group(1)}" class="toc-page-start">第三編</p>',
            chapter,
        )
        self.assertIn("<p>外國帝國主義加劇時期的自強運動 1861-1895年</p>", chapter)
        self.assertIn(
            f'<h1 id="{chapter_href.group(1)}" class="toc-page-start">第十一章</h1>',
            chapter,
        )

    def test_ensure_toc_targets_start_pages_moves_part_heading_before_chapter_number(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"chapter.xhtml#part\">第三編 外國帝國主義加劇時期的自強運動 1861-1895年</a></li>"
                        "<li><a href=\"chapter.xhtml#chapter\">第十一章 清朝中興與自強運動</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><h1 id=\"chapter\" class=\"toc-page-start\">第十一章</h1>"
                        "<p id=\"part\" class=\"toc-page-start\">第三編</p>"
                        "<p>外國帝國主義加劇時期的自強運動 1861-1895年</p>"
                        "<h1 id=\"title\">清朝中興與自強運動</h1></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertLess(chapter.index("第三編"), chapter.index("第十一章"))
        self.assertLess(chapter.index("第十一章"), chapter.index("清朝中興與自強運動"))

    def test_ensure_toc_targets_start_pages_removes_next_heading_preview(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"prev\" href=\"prev.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"next\" href=\"next.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine>"
                        "<itemref idref=\"prev\"/><itemref idref=\"next\"/>"
                        "</spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"next.xhtml#part\">第五編 主義與抗戰，1917-1945年</a></li>"
                        "<li><a href=\"next.xhtml#chapter\">第二十一章 思想革命</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr("EPUB/style/nav.css", "body { margin: 1em; }\n")
                zf.writestr(
                    "EPUB/prev.xhtml",
                    (
                        "<html><body><p>上一章正文。</p>"
                        "<p>第五編</p><p>主義與抗戰</p><p>1917-1945年</p>"
                        "<h2 id=\"preview\">第二十一章</h2></body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/next.xhtml",
                    (
                        "<html><body><p id=\"part\">第五編</p>"
                        "<p>主義與抗戰，1917-1945年</p>"
                        "<h1 id=\"chapter\">第二十一章</h1>"
                        "<p>思想革命正文。</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                prev = zf.read("EPUB/prev.xhtml").decode("utf-8")
                next_chapter = zf.read("EPUB/next.xhtml").decode("utf-8")

        self.assertIn("上一章正文", prev)
        self.assertNotIn("第五編", prev)
        self.assertNotIn("第二十一章", prev)
        self.assertIn("第五編", next_chapter)
        self.assertIn("第二十一章", next_chapter)

    def test_ensure_toc_targets_start_pages_removes_truncated_part_preview(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"prev\" href=\"prev.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"next\" href=\"next.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine>"
                        "<itemref idref=\"prev\"/><itemref idref=\"next\"/>"
                        "</spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/prev.xhtml",
                    (
                        "<html><body><p>上一章正文。</p>"
                        "<p>第三編</p><p>的自强運動</p><p>1861-1895年</p>"
                        "</body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/next.xhtml",
                    (
                        "<html><body><p id=\"part\">第三編</p>"
                        "<p>外國帝國主義加劇時期的自強運動 1861-1895年</p>"
                        "<h1 id=\"chapter\">第十一章</h1>"
                        "<p>清朝中興與自強運動正文。</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                prev = zf.read("EPUB/prev.xhtml").decode("utf-8")
                next_chapter = zf.read("EPUB/next.xhtml").decode("utf-8")

        self.assertIn("上一章正文", prev)
        self.assertNotIn("第三編", prev)
        self.assertNotIn("的自强運動", prev)
        self.assertIn("第三編", next_chapter)
        self.assertIn("外國帝國主義加劇時期的自強運動", next_chapter)

    def test_ensure_toc_targets_start_pages_removes_orphaned_part_year_preview(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"prev\" href=\"prev.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"next\" href=\"next.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine>"
                        "<itemref idref=\"prev\"/><itemref idref=\"next\"/>"
                        "</spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/prev.xhtml",
                    (
                        "<html><body><p>上一章正文。</p>"
                        "<p>第三編</p><p>的自强運動</p><p>1861-1895年</p>"
                        "</body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/next.xhtml",
                    (
                        "<html><body><h1>清朝中興與自強運動</h1>"
                        "<p>正文。</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                prev = zf.read("EPUB/prev.xhtml").decode("utf-8")
                next_chapter = zf.read("EPUB/next.xhtml").decode("utf-8")

        self.assertIn("上一章正文", prev)
        self.assertNotIn("第三編", prev)
        self.assertNotIn("1861-1895年", prev)
        self.assertIn("清朝中興與自強運動", next_chapter)

    def test_ensure_toc_targets_start_pages_removes_short_orphaned_part_preview(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"prev\" href=\"prev.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"next\" href=\"next.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine>"
                        "<itemref idref=\"prev\"/><itemref idref=\"next\"/>"
                        "</spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/prev.xhtml",
                    (
                        "<html><body><p>上一章正文。</p>"
                        "<p>第七編</p><p>毛後中國：追求一個新秩序</p>"
                        "</body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/next.xhtml",
                    "<html><body><h1>第三十二章</h1><p>正文。</p></body></html>",
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                prev = zf.read("EPUB/prev.xhtml").decode("utf-8")
                next_chapter = zf.read("EPUB/next.xhtml").decode("utf-8")

        self.assertIn("上一章正文", prev)
        self.assertNotIn("第七編", prev)
        self.assertNotIn("毛後中國", prev)
        self.assertIn("第三十二章", next_chapter)

    def test_ensure_toc_targets_start_pages_removes_orphaned_chapter_marker_preview(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"prev\" href=\"prev.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"next\" href=\"next.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine>"
                        "<itemref idref=\"prev\"/><itemref idref=\"next\"/>"
                        "</spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/prev.xhtml",
                    "<html><body><p>上一章正文。</p><h2>第十二章</h2></body></html>",
                )
                zf.writestr(
                    "EPUB/next.xhtml",
                    "<html><body><h1>對外關係與宮廷政治</h1><p>正文。</p></body></html>",
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                prev = zf.read("EPUB/prev.xhtml").decode("utf-8")
                next_chapter = zf.read("EPUB/next.xhtml").decode("utf-8")

        self.assertIn("上一章正文", prev)
        self.assertNotIn("第十二章", prev)
        self.assertIn("對外關係與宮廷政治", next_chapter)

    def test_ensure_toc_targets_start_pages_removes_in_file_previous_chapter_bibliography_title(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"chapter\" href=\"chapter.xhtml\" "
                        "media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine><itemref idref=\"chapter\"/></spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><p>第十二章正文。</p>"
                        "<section class=\"page-footnotes\"><p>脚注。</p></section>"
                        "<h2 id=\"bib-title\">第十二章 對外關係與宮廷政治，1861–1880年</h2>"
                        "<p>Warner, Marina, The Dragon Empress.</p>"
                        "<h2 id=\"chapter13\">第十三章</h2>"
                        "<h2>外國侵佔臺灣、新疆與安南</h2>"
                        "<p>第十三章正文。</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertNotIn("第十二章 對外關係與宮廷政治", chapter)
        self.assertIn("Warner, Marina", chapter)
        self.assertIn("第十三章", chapter)

    def test_ensure_toc_targets_start_pages_retargets_numbered_link_without_fragment(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"cover\" href=\"Content_0.xhtml\" "
                        "media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"chapter\" href=\"chapter.xhtml\" "
                        "media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine>"
                        "<itemref idref=\"cover\"/><itemref idref=\"chapter\"/>"
                        "</spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/nav.xhtml",
                    (
                        "<html><body><nav><ol>"
                        "<li><a href=\"Content_0.xhtml\">第二十章 革命、共和與軍閥割據</a></li>"
                        "</ol></nav></body></html>"
                    ),
                )
                zf.writestr(
                    "EPUB/Content_0.xhtml",
                    "<html><body><p>目录页 第二十章 革命、共和與軍閥割據</p></body></html>",
                )
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><body><p>上一章结尾。</p>"
                        "<h2 id=\"chapter20\">第二十章</h2>"
                        "<h2>革命、共和與軍閥割據</h2>"
                        "<p>正文。</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertIn('href="chapter.xhtml#chapter20"', nav)
        self.assertRegex(chapter, r'<h2[^>]*id="chapter20"[^>]*toc-page-start')

    def test_ensure_toc_targets_start_pages_removes_known_garbled_leading_heading(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "ensure_toc_targets_start_pages"))

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "book.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/content.opf",
                    (
                        "<package><manifest>"
                        "<item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\"/>"
                        "<item id=\"chapter\" href=\"chapter.xhtml\" "
                        "media-type=\"application/xhtml+xml\"/>"
                        "</manifest><spine><itemref idref=\"chapter\"/></spine></package>"
                    ),
                )
                zf.writestr(
                    "EPUB/nav.xhtml",
                    "<html><body><nav><ol><li><a href=\"chapter.xhtml\">扙艶倣麠邉薬棩盩</a></li></ol></nav></body></html>",
                )
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    (
                        "<html><head><title>扙艶倣麠邉薬棩盩</title></head>"
                        "<body><h1 id=\"bad\">扙艶倣麠邉薬棩盩</h1>"
                        "<h2 id=\"chapter20\">第二十章</h2>"
                        "<h2>革命、共和與軍閥割據</h2>"
                        "<p>正文。</p></body></html>"
                    ),
                )

            mod.ensure_toc_targets_start_pages(epub_path)

            with ZipFile(epub_path) as zf:
                nav = zf.read("EPUB/nav.xhtml").decode("utf-8")
                chapter = zf.read("EPUB/chapter.xhtml").decode("utf-8")

        self.assertNotIn("扙艶倣麠邉薬棩盩", nav)
        self.assertNotIn("扙艶倣麠邉薬棩盩", chapter)
        self.assertIn("第二十章 革命、共和與軍閥割據", nav)
        self.assertIn("<title>第二十章 革命、共和與軍閥割據</title>", chapter)
        self.assertRegex(chapter, r'<h2[^>]*id="chapter20"[^>]*toc-page-start')

    def test_write_validated_epub_validates_once_after_toc_patching(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "write_validated_epub"))

        events = []

        def fake_write_epub(output_file, _book, _options):
            events.append("write")
            with ZipFile(output_file, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    "<html><body><p>Clean OCR chapter content.</p></body></html>",
                )

        def fake_ensure_toc_targets_start_pages(_output_file):
            events.append("toc")
            return {"targets": 0, "xhtml_files": [], "css_files": []}

        def fake_validate_epub_no_ocr_noise(_output_file, strict=False):
            events.append(f"validate:{strict}")
            return []

        fake_epub = SimpleNamespace(write_epub=fake_write_epub)

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "book.epub"
            with mock.patch.object(mod.epub_validate, "epub", fake_epub, create=True), \
                    mock.patch.object(
                        mod.epub_validate,
                        "ensure_toc_targets_start_pages",
                        fake_ensure_toc_targets_start_pages,
                    ), \
                    mock.patch.object(
                        mod.epub_validate,
                        "validate_epub_no_ocr_noise",
                        fake_validate_epub_no_ocr_noise,
                    ):
                mod.write_validated_epub(
                    object(),
                    str(output_path),
                    strict_ocr_validation=False,
                )

            self.assertTrue(output_path.exists())

        self.assertEqual(["write", "toc", "validate:False"], events)

    def test_create_epub_does_not_leave_output_file_when_validation_fails(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "create_epub"))

        class FakeBook:
            def set_identifier(self, _identifier):
                pass

            def set_title(self, _title):
                pass

            def set_language(self, _language):
                pass

            def add_author(self, _author):
                pass

            def add_item(self, _item):
                pass

            def set_cover(self, _filename, _data):
                pass

        class FakeItem:
            def __init__(self, *args, **kwargs):
                pass

            def add_item(self, _item):
                pass

        def fake_write_epub(output_file, _book, _options):
            Path(output_file).write_text("noisy epub", encoding="utf-8")

        fake_epub = SimpleNamespace(
            EpubBook=FakeBook,
            EpubItem=FakeItem,
            EpubImage=FakeItem,
            EpubHtml=FakeItem,
            EpubNcx=FakeItem,
            EpubNav=FakeItem,
            write_epub=fake_write_epub,
        )

        def fail_validation(_output_file, *args, **kwargs):
            raise RuntimeError("validation failed")

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "book.epub"
            with mock.patch.object(mod.epub_builder, "epub", fake_epub, create=True), \
                    mock.patch.object(mod.epub_validate, "epub", fake_epub, create=True), \
                    mock.patch.object(mod.epub_validate, "validate_epub_no_ocr_noise", fail_validation):
                with self.assertRaises(RuntimeError):
                    mod.create_epub(
                        "Title",
                        [],
                        str(output_path),
                        str(Path(td) / "images"),
                        strict_ocr_validation=True,
                    )

            self.assertFalse(output_path.exists())

    def test_create_epub_manual_toc_match_alias_uses_display_title(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "create_epub"))

        captured = {}

        def capture_book(book, _output_file, **_kwargs):
            captured["book"] = book

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "前置正文。\n(第二章) 南京年代\n章節正文開始。",
                                "images": {},
                            }
                        }
                    ]
                }
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                mod.epub_builder,
                "write_validated_epub",
                capture_book,
            ):
                mod.create_epub(
                    "Test Book",
                    results,
                    str(Path(td) / "book.epub"),
                    str(Path(td) / "images"),
                    confirmed_headings=[
                        {
                            "page": 1,
                            "title": "第三章 南京年代",
                            "match": "(第二章) 南京年代",
                        }
                    ],
                )

        chapters = list(captured["book"].toc)
        self.assertEqual(["Content", "第三章 南京年代"],
                         [chapter.title for chapter in chapters])
        chapter_html = chapters[1].content
        if isinstance(chapter_html, bytes):
            chapter_html = chapter_html.decode("utf-8")
        self.assertIn("第三章 南京年代", chapter_html)
        self.assertNotIn("(第二章) 南京年代", chapter_html)

    def test_create_epub_manual_toc_consumes_split_title_suffix(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "create_epub"))

        captured = {}

        def capture_book(book, _output_file, **_kwargs):
            captured["book"] = book

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "第八章\n妄想勝利\n正文開始。",
                                "images": {},
                            }
                        }
                    ]
                }
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                mod.epub_builder,
                "write_validated_epub",
                capture_book,
            ):
                mod.create_epub(
                    "Test Book",
                    results,
                    str(Path(td) / "book.epub"),
                    str(Path(td) / "images"),
                    confirmed_headings=[
                        {"page": 1, "title": "第八章 妄想勝利"}
                    ],
                )

        chapter_html = list(captured["book"].toc)[0].content
        if isinstance(chapter_html, bytes):
            chapter_html = chapter_html.decode("utf-8")
        self.assertIn("第八章 妄想勝利", chapter_html)
        self.assertNotIn("<p>妄想勝利</p>", chapter_html)
        self.assertIn("正文開始。", chapter_html)

    def test_create_epub_manual_toc_can_split_at_page_start(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "create_epub"))

        captured = {}

        def capture_book(book, _output_file, **_kwargs):
            captured["book"] = book

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "革命\n第一部\n正文開始。",
                                "images": {},
                            }
                        }
                    ]
                }
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                mod.epub_builder,
                "write_validated_epub",
                capture_book,
            ):
                mod.create_epub(
                    "Test Book",
                    results,
                    str(Path(td) / "book.epub"),
                    str(Path(td) / "images"),
                    confirmed_headings=[
                        {
                            "page": 1,
                            "title": "第一部 革命",
                            "page_start": True,
                        }
                    ],
                )

        chapters = list(captured["book"].toc)
        self.assertEqual(["第一部 革命"], [chapter.title for chapter in chapters])
        chapter_html = chapters[0].content
        if isinstance(chapter_html, bytes):
            chapter_html = chapter_html.decode("utf-8")
        self.assertIn("第一部 革命", chapter_html)
        self.assertIn("正文開始。", chapter_html)

    def test_create_epub_manual_toc_alias_cleans_subtitle_without_resplitting(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "create_epub"))

        captured = {}

        def capture_book(book, _output_file, **_kwargs):
            captured["book"] = book

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "〔第十三章〕\n尼克森及晚年\n正文開始。",
                                "images": {},
                            }
                        }
                    ]
                }
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                mod.epub_builder,
                "write_validated_epub",
                capture_book,
            ):
                mod.create_epub(
                    "Test Book",
                    results,
                    str(Path(td) / "book.epub"),
                    str(Path(td) / "images"),
                    confirmed_headings=[
                        {
                            "page": 1,
                            "title": "第十三章 尼克森和晚年歲月",
                            "aliases": ["尼克森及晚年"],
                        }
                    ],
                )

        chapter_html = list(captured["book"].toc)[0].content
        if isinstance(chapter_html, bytes):
            chapter_html = chapter_html.decode("utf-8")
        self.assertIn("第十三章 尼克森和晚年歲月", chapter_html)
        self.assertNotIn("尼克森及晚年", chapter_html)
        self.assertIn("正文開始。", chapter_html)

    def test_strip_missing_image_references_removes_markdown_and_html_images(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod.epub_builder, "_strip_missing_image_references"))

        markdown = (
            "Before.\n"
            "![](imgs/missing-a.jpg)\n"
            '<div style="text-align: center;"><img src="imgs/missing-b.jpg" '
            'alt="Image" width="99%" /></div>\n'
            "<img alt=\"Image\" src='imgs/missing-c.jpg' width=\"50%\" />\n"
            '<img src="imgs/kept.jpg" alt="Keep" />\n'
            "After."
        )

        cleaned = mod.epub_builder._strip_missing_image_references(
            markdown,
            {
                "imgs/missing-a.jpg",
                "imgs/missing-b.jpg",
                "imgs/missing-c.jpg",
            },
        )

        self.assertIn("Before.", cleaned)
        self.assertIn("After.", cleaned)
        self.assertIn('src="imgs/kept.jpg"', cleaned)
        self.assertNotIn("missing-a.jpg", cleaned)
        self.assertNotIn("missing-b.jpg", cleaned)
        self.assertNotIn("missing-c.jpg", cleaned)
        self.assertNotIn('<div style="text-align: center;"></div>', cleaned)

    def test_apply_page_image_fallbacks_renders_sparse_family_tree_page(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "apply_page_image_fallbacks"))

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "## 家 系 圖",
                                "images": {},
                            }
                        }
                    ]
                }
            }
        ]

        class FakePixmap:
            def save(self, path):
                Path(path).write_bytes(b"fake png")

        class FakePage:
            def get_pixmap(self, matrix=None, clip=None, alpha=False):
                return FakePixmap()

        class FakeDoc:
            page_count = 1

            def __getitem__(self, index):
                if index != 0:
                    raise IndexError(index)
                return FakePage()

            def close(self):
                pass

        fake_fitz = SimpleNamespace(
            open=lambda _path: FakeDoc(),
            Matrix=lambda _x, _y: object(),
        )

        with tempfile.TemporaryDirectory() as td:
            image_dir = Path(td) / "images"
            with mock.patch.object(mod.page_image_fallback, "fitz", fake_fitz):
                count = mod.apply_page_image_fallbacks(
                    "source.pdf",
                    results,
                    str(image_dir),
                )

            rel_path = "imgs/page_fallback_0001.png"
            self.assertEqual(1, count)
            markdown = results[0]["result"]["layoutParsingResults"][0]["markdown"]
            self.assertEqual(
                "",
                markdown["images"][rel_path],
            )
            self.assertIn(
                f'<img src="{rel_path}" alt="家 系 圖" width="100%" />',
                markdown["text"],
            )
            self.assertEqual(b"fake png", (image_dir / rel_path).read_bytes())

    def test_apply_page_image_fallbacks_renders_empty_ocr_page_with_visible_marks(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "apply_page_image_fallbacks"))

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": "",
                                "images": {},
                            }
                        }
                    ]
                }
            }
        ]

        class FakePixmap:
            width = 10
            height = 10
            n = 3
            samples = bytes([255, 255, 255] * 80 + [0, 0, 0] * 20)

            def save(self, path):
                Path(path).write_bytes(b"rendered manuscript page")

        class FakePage:
            def get_pixmap(self, matrix=None, clip=None, alpha=False):
                return FakePixmap()

        class FakeDoc:
            page_count = 1

            def __getitem__(self, index):
                if index != 0:
                    raise IndexError(index)
                return FakePage()

            def close(self):
                pass

        fake_fitz = SimpleNamespace(
            open=lambda _path: FakeDoc(),
            Matrix=lambda _x, _y: object(),
        )

        with tempfile.TemporaryDirectory() as td:
            image_dir = Path(td) / "images"
            with mock.patch.object(mod.page_image_fallback, "fitz", fake_fitz):
                count = mod.apply_page_image_fallbacks(
                    "source.pdf",
                    results,
                    str(image_dir),
                )

            rel_path = "imgs/page_fallback_0001.png"
            self.assertEqual(1, count)
            markdown = results[0]["result"]["layoutParsingResults"][0]["markdown"]
            self.assertEqual("", markdown["images"][rel_path])
            self.assertIn(
                f'<img src="{rel_path}" alt="Page image" width="100%" />',
                markdown["text"],
            )
            self.assertEqual(
                b"rendered manuscript page",
                (image_dir / rel_path).read_bytes(),
            )

    def test_apply_page_image_fallbacks_reuses_single_pdf_open_for_multiple_pages(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "apply_page_image_fallbacks"))

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "", "images": {}}},
                        {"markdown": {"text": "", "images": {}}},
                    ]
                }
            }
        ]

        class FakePixmap:
            width = 10
            height = 10
            n = 3
            samples = bytes([255, 255, 255] * 80 + [0, 0, 0] * 20)

            def save(self, path):
                Path(path).write_bytes(b"rendered page")

        class FakePage:
            rect = SimpleNamespace(width=100, height=120)

            def get_pixmap(self, matrix=None, clip=None, alpha=False):
                return FakePixmap()

        class FakeDoc:
            page_count = 2

            def __init__(self):
                self.close_calls = 0

            def __getitem__(self, index):
                if index not in (0, 1):
                    raise IndexError(index)
                return FakePage()

            def close(self):
                self.close_calls += 1

        fake_doc = FakeDoc()
        open_calls = []

        def fake_open(path):
            open_calls.append(path)
            return fake_doc

        fake_fitz = SimpleNamespace(
            open=fake_open,
            Matrix=lambda _x, _y: object(),
        )

        with tempfile.TemporaryDirectory() as td:
            image_dir = Path(td) / "images"
            with mock.patch.object(mod.page_image_fallback, "fitz", fake_fitz):
                count = mod.apply_page_image_fallbacks(
                    "source.pdf",
                    results,
                    str(image_dir),
                )

            self.assertEqual(2, count)
            self.assertEqual(["source.pdf"], open_calls)
            self.assertEqual(1, fake_doc.close_calls)
            self.assertTrue((image_dir / "imgs/page_fallback_0001.png").exists())
            self.assertTrue((image_dir / "imgs/page_fallback_0002.png").exists())

    def test_apply_page_image_fallbacks_renders_page_when_image_asset_missing(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "apply_page_image_fallbacks"))

        missing_rel_path = "imgs/missing-photo.jpg"
        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {
                                "text": (
                                    '<div style="text-align: center;">'
                                    f'<img src="{missing_rel_path}" alt="Image" width="99%" />'
                                    "</div>"
                                ),
                                "images": {missing_rel_path: "https://example.test/image.jpg"},
                            }
                        }
                    ]
                }
            }
        ]

        class FakePixmap:
            def save(self, path):
                Path(path).write_bytes(b"fallback for missing image")

        class FakePage:
            def get_pixmap(self, matrix=None, clip=None, alpha=False):
                return FakePixmap()

        class FakeDoc:
            page_count = 1

            def __getitem__(self, index):
                if index != 0:
                    raise IndexError(index)
                return FakePage()

            def close(self):
                pass

        fake_fitz = SimpleNamespace(
            open=lambda _path: FakeDoc(),
            Matrix=lambda _x, _y: object(),
        )

        with tempfile.TemporaryDirectory() as td:
            image_dir = Path(td) / "images"
            with mock.patch.object(mod.page_image_fallback, "fitz", fake_fitz):
                count = mod.apply_page_image_fallbacks(
                    "source.pdf",
                    results,
                    str(image_dir),
                )

            rel_path = "imgs/page_fallback_0001.png"
            self.assertEqual(1, count)
            markdown = results[0]["result"]["layoutParsingResults"][0]["markdown"]
            self.assertEqual("", markdown["images"][rel_path])
            self.assertIn(
                f'<img src="{rel_path}" alt="Page image" width="100%" />',
                markdown["text"],
            )
            self.assertEqual(
                b"fallback for missing image",
                (image_dir / rel_path).read_bytes(),
            )

    def test_repair_page_order_by_printed_numbers_swaps_adjacent_inversion(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "repair_page_order_by_printed_numbers"))

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "page 257 body"}},
                        {"markdown": {"text": "page 256 body"}},
                        {"markdown": {"text": "page 259 body"}},
                        {"markdown": {"text": "page 258 body"}},
                    ]
                }
            }
        ]

        class FakePage:
            def __init__(self, text):
                self._text = text

            def get_text(self):
                return self._text

        class FakeDoc:
            def __init__(self):
                self.pages = [
                    FakePage("• 257 •"),
                    FakePage("• 256 •"),
                    FakePage("• 259 •"),
                    FakePage("• 258 •"),
                ]
                self.page_count = len(self.pages)

            def __getitem__(self, index):
                return self.pages[index]

            def close(self):
                pass

        fake_fitz = SimpleNamespace(open=lambda _path: FakeDoc())

        with mock.patch.object(mod.page_order_repair, "fitz", fake_fitz):
            swaps = mod.repair_page_order_by_printed_numbers("source.pdf", results)

        pages = results[0]["result"]["layoutParsingResults"]
        self.assertEqual(2, swaps)
        self.assertEqual(
            ["page 256 body", "page 257 body", "page 258 body", "page 259 body"],
            [page["markdown"]["text"] for page in pages],
        )

    def test_repair_page_order_by_printed_numbers_infers_systematic_pair_inversion(self):
        mod = _load_script_module()
        self.assertTrue(hasattr(mod, "repair_page_order_by_printed_numbers"))

        results = [
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "front matter"}},
                        {"markdown": {"text": "page 1 body"}},
                        {"markdown": {"text": "blank verso"}},
                        {"markdown": {"text": "page 3 body"}},
                        {"markdown": {"text": "page 2 body"}},
                        {"markdown": {"text": "page 5 body"}},
                        {"markdown": {"text": "page 4 body"}},
                        {"markdown": {"text": "page 7 body"}},
                        {"markdown": {"text": "page 6 body"}},
                    ]
                }
            }
        ]

        class FakePage:
            def __init__(self, text):
                self._text = text

            def get_text(self):
                return self._text

        class FakeDoc:
            def __init__(self):
                self.pages = [
                    FakePage(""),
                    FakePage("• 1 •"),
                    FakePage(""),
                    FakePage(""),
                    FakePage(""),
                    FakePage("• 5 •"),
                    FakePage("• 4 •"),
                    FakePage("• 7 •"),
                    FakePage("• 6 •"),
                ]
                self.page_count = len(self.pages)

            def __getitem__(self, index):
                return self.pages[index]

            def close(self):
                pass

        fake_fitz = SimpleNamespace(open=lambda _path: FakeDoc())

        with mock.patch.object(mod.page_order_repair, "fitz", fake_fitz):
            swaps = mod.repair_page_order_by_printed_numbers("source.pdf", results)

        pages = results[0]["result"]["layoutParsingResults"]
        self.assertEqual(3, swaps)
        self.assertEqual(
            [
                "front matter",
                "page 1 body",
                "blank verso",
                "page 2 body",
                "page 3 body",
                "page 4 body",
                "page 5 body",
                "page 6 body",
                "page 7 body",
            ],
            [page["markdown"]["text"] for page in pages],
        )

    def test_invalid_numeric_env_values_fall_back_to_defaults(self):
        env_overrides = {
            "PADDLE_CHUNK_SIZE": "abc",
            "PADDLE_API_TIMEOUT_SECONDS": "def",
            "PADDLE_PAGE_MARGIN_PT": "ghi",
            "PADDLE_BOTTOM_PADDING_PERCENT": "jkl",
            "EPUB_COVER_MAX_EDGE": "ghi",
            "EPUB_COVER_JPEG_QUALITY": "jkl",
        }
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in env_overrides
        }
        clean_env.update(env_overrides)

        with mock.patch.dict(os.environ, clean_env, clear=True):
            mod = _load_fresh_script_module()

        self.assertEqual(5, mod.CHUNK_SIZE)
        self.assertEqual(600, mod.API_TIMEOUT_SECONDS)
        self.assertEqual(36, mod.PADDLE_PAGE_MARGIN_PT)
        self.assertEqual(0, mod.PADDLE_BOTTOM_PADDING_PERCENT)
        self.assertEqual(2000, mod.DEFAULT_COVER_MAX_EDGE)
        self.assertEqual(82, mod.DEFAULT_COVER_JPEG_QUALITY)

    def test_non_positive_numeric_env_values_fall_back_to_defaults(self):
        env_overrides = {
            "PADDLE_CHUNK_SIZE": "0",
            "PADDLE_API_TIMEOUT_SECONDS": "-1",
            "EPUB_COVER_MAX_EDGE": "0",
            "EPUB_COVER_JPEG_QUALITY": "-10",
        }
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in env_overrides
        }
        clean_env.update(env_overrides)

        with mock.patch.dict(os.environ, clean_env, clear=True):
            mod = _load_fresh_script_module()

        self.assertEqual(5, mod.CHUNK_SIZE)
        self.assertEqual(600, mod.API_TIMEOUT_SECONDS)
        self.assertEqual(2000, mod.DEFAULT_COVER_MAX_EDGE)
        self.assertEqual(82, mod.DEFAULT_COVER_JPEG_QUALITY)

    def test_split_pdf_adds_uniform_guard_band_margin_for_edge_ocr(self):
        mod = _load_script_module()
        fitz = mod.paddle_api.fitz
        if fitz is None:
            self.skipTest("PyMuPDF is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            source_pdf = os.path.join(temp_dir, "edge.pdf")
            doc = fitz.open()
            doc.new_page(width=100, height=200)
            doc.save(source_pdf)
            doc.close()

            chunk_paths = mod.paddle_api.split_pdf(source_pdf, chunk_size=1)
            try:
                chunk_doc = fitz.open(chunk_paths[0])
                mediabox = chunk_doc[0].mediabox
                chunk_doc.close()
            finally:
                if chunk_paths:
                    shutil.rmtree(os.path.dirname(chunk_paths[0]), ignore_errors=True)

        # Page stamped onto a larger canvas with 36pt guard-band on all 4 sides:
        # width = 100 + 2×36 = 172, height = 200 + 2×36 = 272
        self.assertAlmostEqual(0, mediabox.x0)
        self.assertAlmostEqual(0, mediabox.y0)
        self.assertAlmostEqual(172, mediabox.x1)
        self.assertAlmostEqual(272, mediabox.y1)


class TestGarbledCjkDetection(unittest.TestCase):
    """Unit tests for repeated-character and window-based garbled CJK detection."""

    def test_find_repeated_char_spans_detects_8plus_repeats(self):
        """Verify that consecutive repeated characters ≥8 times are detected."""
        mod = _load_fresh_script_module()
        from paddle_pipeline.ocr_noise import _find_repeated_char_spans

        chars = list("三三三三三三三三")  # 8 chars, should be detected
        spans = _find_repeated_char_spans(chars)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0], "三三三三三三三三")

    def test_find_repeated_char_spans_ignores_7_repeats(self):
        """Verify that consecutive repeated characters <8 times are ignored."""
        mod = _load_fresh_script_module()
        from paddle_pipeline.ocr_noise import _find_repeated_char_spans

        chars = list("三三三三三三三")  # 7 chars, should be ignored
        spans = _find_repeated_char_spans(chars)

        self.assertEqual(len(spans), 0)

    def test_find_repeated_char_spans_handles_empty_list(self):
        """Verify graceful handling of empty input."""
        mod = _load_fresh_script_module()
        from paddle_pipeline.ocr_noise import _find_repeated_char_spans

        spans = _find_repeated_char_spans([])

        self.assertEqual([], spans)

    def test_find_repeated_char_spans_handles_mixed_runs(self):
        """Verify detection across multiple repeated-character runs."""
        mod = _load_fresh_script_module()
        from paddle_pipeline.ocr_noise import _find_repeated_char_spans

        # Two runs: 8 repeats + 10 repeats
        chars = list("一一一一一一一一二二二二二二二二二二")
        spans = _find_repeated_char_spans(chars)

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], "一一一一一一一一")
        self.assertEqual(spans[1], "二二二二二二二二二二")

    def test_find_garbled_cjk_in_epub_returns_actual_spans(self):
        """Verify that find_garbled_cjk_in_epub returns actual span text, not summaries."""
        mod = _load_fresh_script_module()

        # Create a simple EPUB with obvious repeated characters
        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "garbled.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                # Long text corpus + a repeated-char hallucination
                corpus = (
                    "今天天氣很好，我和朋友一起去公園散步。" * 50
                )
                content = f"<html><body><p>{corpus}三三三三三三三三</p></body></html>"
                zf.writestr("EPUB/chapter.xhtml", content)

            findings = mod.find_garbled_cjk_in_epub(
                epub_path,
                mod.EPUB_STRUCTURAL_FILES,
            )

        # Verify structure: should have "file" and "spans" keys, not "token" or "count"
        self.assertGreater(len(findings), 0, "Should detect garbled text")
        finding = findings[0]
        self.assertIn("file", finding)
        self.assertIn("spans", finding)
        self.assertNotIn("token", finding)
        self.assertNotIn("count", finding)

        # Verify spans are actual text
        self.assertIsInstance(finding["spans"], list)
        self.assertGreater(len(finding["spans"]), 0)
        for span in finding["spans"]:
            self.assertIsInstance(span, str)

    def test_find_garbled_cjk_in_epub_returns_empty_for_clean_text(self):
        """Verify no false positives on normal Chinese text."""
        mod = _load_fresh_script_module()

        # Normal Chinese passage repeated to build bigram model
        normal = ("今天天氣很好，我和朋友一起去公園散步。" * 50)

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "clean.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                zf.writestr(
                    "EPUB/chapter.xhtml",
                    f"<html><body><p>{normal}</p></body></html>",
                )

            findings = mod.find_garbled_cjk_in_epub(
                epub_path,
                mod.EPUB_STRUCTURAL_FILES,
            )

        self.assertEqual([], findings, "Normal Chinese should not be detected as garbled")

    def test_find_garbled_cjk_in_epub_empty_epub_returns_empty(self):
        """Verify graceful handling of empty EPUB."""
        mod = _load_fresh_script_module()

        with tempfile.TemporaryDirectory() as td:
            epub_path = Path(td) / "empty.epub"
            with ZipFile(epub_path, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)

            findings = mod.find_garbled_cjk_in_epub(
                epub_path,
                mod.EPUB_STRUCTURAL_FILES,
            )

        self.assertEqual([], findings)

