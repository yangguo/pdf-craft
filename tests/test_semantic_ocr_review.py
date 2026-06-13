"""Tests for paddle_pipeline.semantic_ocr_review."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NORMAL_CHINESE = (
    "香港問題是中國近代史上的重要問題。中英双方經過多輪談判，"
    "逐步形成了有關香港前途和制度安排的基本方針。回顧當時情況，"
    "很多人既有期待，也有疑慮，需要耐心解釋和溝通。"
)

_SEMANTIC_GARBLE = (
    "誰誰歸白一工任合室什其林人然已將則一語至不遂金方是任見八百仁素今月新用至不"
    "根據中國大陸二○○○年出版的一篇未載明來源出處的文章"
)

_TIER3_GARBLE = (
    "安事變凸顯出壽、罰之罰均固人關系，口是聲刀就食詳書刀公"
)


def _write_epub(epub_path: Path, body_html: str) -> None:
    """Create a minimal valid EPUB containing *body_html*."""
    with ZipFile(epub_path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            (
                '<?xml version="1.0"?>'
                '<container version="1.0" '
                'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                "<rootfiles>"
                '<rootfile full-path="EPUB/book.opf" '
                'media-type="application/oebps-package+xml"/>'
                "</rootfiles>"
                "</container>"
            ),
        )
        archive.writestr(
            "EPUB/book.opf",
            (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                "<manifest>"
                '<item id="chap" href="chapter.xhtml" media-type="application/xhtml+xml"/>'
                "</manifest>"
                '<spine><itemref idref="chap"/></spine>'
                "</package>"
            ),
        )
        archive.writestr("EPUB/chapter.xhtml", body_html, compress_type=ZIP_DEFLATED)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

class TestSemanticCandidateGeneration(unittest.TestCase):
    def test_detects_semantic_garble(self):
        """Semantic garble should appear in high-score candidates."""
        from paddle_pipeline.semantic_ocr_review import generate_semantic_candidates

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE}</p>"
            f"<p>{_SEMANTIC_GARBLE}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)

            candidates = generate_semantic_candidates(
                str(epub), limit=50, min_score=0.45, min_cjk=14
            )

        excerpts = [c["excerpt"] for c in candidates]
        # The garbled portion should appear in at least one candidate excerpt.
        self.assertTrue(
            any("誰誰歸白" in e for e in excerpts),
            f"Candidates: {excerpts}",
        )

    def test_normal_chinese_not_top_ranked(self):
        """Large blocks of normal Chinese should yield few or no candidates."""
        from paddle_pipeline.semantic_ocr_review import generate_semantic_candidates

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE * 60}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_semantic_candidates(
                str(epub), limit=30, min_score=0.55, min_cjk=14
            )

        # Normal text should generate very few candidates.
        self.assertLessEqual(len(candidates), 5, candidates)

    def test_every_candidate_has_expected_fields(self):
        """Each candidate must have the documented fields."""
        from paddle_pipeline.semantic_ocr_review import generate_semantic_candidates

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE}</p>"
            f"<p>{_SEMANTIC_GARBLE}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_semantic_candidates(
                str(epub), limit=30, min_score=0.45, min_cjk=14
            )

        for c in candidates:
            self.assertIn("file", c)
            self.assertIn("excerpt", c)
            self.assertIn("context_before", c)
            self.assertIn("context_after", c)
            self.assertIn("score", c)
            self.assertIn("features", c)
            self.assertIn("cjk_len", c)
            self.assertIn("source", c)
            self.assertEqual(c["source"], "semantic_candidate")

    def test_semantic_candidate_keeps_punctuation_for_llm_review(self):
        """LLM review should see punctuation that the CJK scorer ignores."""
        from paddle_pipeline.semantic_ocr_review import generate_semantic_candidates

        title_list = (
            "譯作極豐，有《蔣經國傳》、《裕仁天皇》、《轉向：從尼克森到柯林頓美中關係揭密》、"
            "《季辛吉大外交》（合譯）、《大棋盤》、《將門虎子》、《買通白宮》、《李潔明回憶錄》。"
        )
        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE * 10}</p>"
            f"<p>{title_list}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)

            candidates = generate_semantic_candidates(
                str(epub), limit=50, min_score=0.20, min_cjk=14
            )

        candidate = next(
            c for c in candidates
            if "季辛吉大外交" in c["excerpt"]
        )
        self.assertIn("llm_excerpt", candidate)
        self.assertIn("《季辛吉大外交》（合譯）", candidate["llm_excerpt"])
        self.assertIn("《大棋盤》", candidate["llm_excerpt"])

    def test_empty_epub_returns_empty_list(self):
        """An EPUB with no CJK text should return an empty list."""
        from paddle_pipeline.semantic_ocr_review import generate_semantic_candidates

        body = "<html><body><p>Only English text here.</p></body></html>"
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_semantic_candidates(
                str(epub), limit=30, min_score=0.55, min_cjk=14
            )

        self.assertEqual(len(candidates), 0)

    def test_repetition_garble_scored_high(self):
        """Repeated-character garble (e.g. 三三三三…) should score very high."""
        from paddle_pipeline.semantic_ocr_review import generate_semantic_candidates

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE}</p>"
            "<p>前三三三三三三三三三三三三三三三三三三三三三三三三三。後續正常中文。</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_semantic_candidates(
                str(epub), limit=30, min_score=0.55, min_cjk=14
            )

        excerpts = [c["excerpt"] for c in candidates]
        self.assertTrue(
            any("三三三三三" in e for e in excerpts),
            f"Candidates: {excerpts}",
        )


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

class TestDotenvLoading(unittest.TestCase):
    def test_loads_openai_vars(self):
        from paddle_pipeline.semantic_ocr_review import _load_dotenv

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("OPENAI_API_KEY=sk-test-123\n")
            f.write("OPENAI_BASE_URL=https://api.openai.com/v1\n")
            f.write("OPENAI_MODEL=gpt-4o-mini\n")
            f.write("# comment\n")
            f.write("OTHER_VAR=ignored\n")
            env_path = f.name

        try:
            env = _load_dotenv(env_path)
            self.assertEqual(env.get("OPENAI_API_KEY"), "sk-test-123")
            self.assertEqual(
                env.get("OPENAI_BASE_URL"), "https://api.openai.com/v1"
            )
            self.assertEqual(env.get("OPENAI_MODEL"), "gpt-4o-mini")
            # _load_dotenv loads every key=value, not just OpenAI ones;
            # filtering happens downstream in _openai_config_from_args.
            self.assertEqual(env.get("OTHER_VAR"), "ignored")
        finally:
            os.unlink(env_path)

    def test_missing_file_returns_empty(self):
        from paddle_pipeline.semantic_ocr_review import _load_dotenv

        env = _load_dotenv("nonexistent_abc123.env")
        self.assertEqual(env, {})

    def test_strips_quotes(self):
        from paddle_pipeline.semantic_ocr_review import _load_dotenv

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write('OPENAI_API_KEY="sk-double-quoted"\n')
            f.write("OPENAI_MODEL='gpt-single-quoted'\n")
            env_path = f.name

        try:
            env = _load_dotenv(env_path)
            self.assertEqual(env["OPENAI_API_KEY"], "sk-double-quoted")
            self.assertEqual(env["OPENAI_MODEL"], "gpt-single-quoted")
        finally:
            os.unlink(env_path)


# ---------------------------------------------------------------------------
# LLM review (mocked)
# ---------------------------------------------------------------------------

class TestLLMReview(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.abspath("."))

    def _make_candidate(self, **overrides):
        c = {
            "file": "EPUB/chapter.xhtml",
            "excerpt": "誰誰歸白一工任合室什其林人然已將則一語",
            "context_before": "前文內容。",
            "context_after": "後文內容。",
            "score": 0.85,
            "features": {},
            "cjk_len": 18,
            "source": "semantic_candidate",
        }
        c.update(overrides)
        return c

    def test_build_llm_prompt_includes_context(self):
        from paddle_pipeline.semantic_ocr_review import _build_llm_prompt

        candidate = self._make_candidate()
        messages = _build_llm_prompt(candidate)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("OCR", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(candidate["excerpt"], messages[1]["content"])
        self.assertIn(candidate["context_before"], messages[1]["content"])

    def test_build_llm_prompt_uses_punctuation_preserving_excerpt(self):
        from paddle_pipeline.semantic_ocr_review import _build_llm_prompt

        candidate = self._make_candidate(
            excerpt="季辛吉大外交合譯大棋盤將門虎子買通白宮",
            llm_excerpt="《季辛吉大外交》（合譯）、《大棋盤》、《將門虎子》、《買通白宮》",
        )
        messages = _build_llm_prompt(candidate)

        self.assertIn(candidate["llm_excerpt"], messages[1]["content"])
        self.assertNotIn(
            "候選片段：\n" + candidate["excerpt"],
            messages[1]["content"],
        )

    def test_review_successful_response(self):
        """Mock a successful LLM response."""
        from paddle_pipeline.semantic_ocr_review import _review_candidate_with_llm

        candidate = self._make_candidate()
        config = {
            "api_key": "sk-test",
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
        }

        mock_response = json.dumps(
            {
                "is_garbled": True,
                "confidence": 0.95,
                "category": "semantic_garble",
                "suspicious_span": "誰誰歸白一工任合室",
                "reason": "這段文字不符合中文語法。",
                "needs_source_ocr": True,
            }
        )

        with mock.patch(
            "paddle_pipeline.semantic_ocr_review._call_openai_chat",
            return_value={"is_garbled": True, "confidence": 0.95, "category": "semantic_garble", "suspicious_span": "誰誰歸白一工任合室", "reason": "這段文字不符合中文語法。", "needs_source_ocr": True},
        ):
            result = _review_candidate_with_llm(candidate, config)

        self.assertIn("llm_review", result)
        review = result["llm_review"]
        self.assertTrue(review["is_garbled"])
        self.assertGreater(review["confidence"], 0.9)

    def test_review_http_error_records_llm_error(self):
        from paddle_pipeline.semantic_ocr_review import _review_candidate_with_llm

        candidate = self._make_candidate()
        config = {
            "api_key": "sk-test",
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
        }

        with mock.patch(
            "paddle_pipeline.semantic_ocr_review._call_openai_chat",
            return_value={"error": "connection refused"},
        ):
            result = _review_candidate_with_llm(candidate, config)

        self.assertIn("llm_error", result)
        self.assertNotIn("llm_review", result)

    def test_strict_mode_raises_on_error(self):
        from paddle_pipeline.semantic_ocr_review import review_candidates_with_llm

        candidates = [self._make_candidate()]
        config = {
            "api_key": "sk-test",
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
        }

        with mock.patch(
            "paddle_pipeline.semantic_ocr_review._call_openai_chat",
            return_value={"error": "timeout"},
        ):
            with self.assertRaises(RuntimeError):
                review_candidates_with_llm(candidates, config, strict=True)

    def test_review_candidates_with_llm_augments_in_place(self):
        from paddle_pipeline.semantic_ocr_review import review_candidates_with_llm

        candidates = [self._make_candidate(), self._make_candidate()]
        config = {
            "api_key": "sk-test",
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
        }

        review_mock = {
            "is_garbled": False,
            "confidence": 0.3,
            "category": "normal_text",
            "suspicious_span": "",
            "reason": "Looks like normal text.",
            "needs_source_ocr": False,
        }

        with mock.patch(
            "paddle_pipeline.semantic_ocr_review._call_openai_chat",
            return_value=review_mock,
        ):
            result = review_candidates_with_llm(candidates, config)

        self.assertEqual(len(result), 2)
        for c in result:
            self.assertIn("llm_review", c)
            self.assertFalse(c["llm_review"]["is_garbled"])


# ---------------------------------------------------------------------------
# Sentence-level candidate generation
# ---------------------------------------------------------------------------

class TestSentenceCandidates(unittest.TestCase):
    def test_iter_cjk_sentences_splits_by_punctuation(self):
        from paddle_pipeline.semantic_ocr_review import _iter_cjk_sentences

        text = "這是第一句話。這是第二句話！這是第三句話？"
        sentences = list(_iter_cjk_sentences(text, min_cjk=4))
        self.assertGreaterEqual(len(sentences), 2)

    def test_iter_cjk_sentences_skips_short(self):
        from paddle_pipeline.semantic_ocr_review import _iter_cjk_sentences

        text = "短。這是一句足夠長的中文句子所以不應該被跳過。"
        sentences = list(_iter_cjk_sentences(text, min_cjk=8))
        # "短。" is too short, only the long sentence should appear
        excerpts = [s[0] for s in sentences]
        self.assertTrue(any("足夠長" in e for e in excerpts), excerpts)

    def test_iter_cjk_sentences_skips_english_only(self):
        from paddle_pipeline.semantic_ocr_review import _iter_cjk_sentences

        text = "Only English text here. No CJK at all."
        sentences = list(_iter_cjk_sentences(text, min_cjk=4))
        self.assertEqual(len(sentences), 0)

    def test_generate_sentence_candidates_catches_tier3_garble(self):
        """Tier-3 semantic garble (common bigrams, nonsense sentence) should
        appear in sentence candidates even though it scores low statistically."""
        from paddle_pipeline.semantic_ocr_review import generate_sentence_candidates

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE}</p>"
            f"<p>{_TIER3_GARBLE}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_sentence_candidates(
                str(epub), limit=30, min_cjk=6
            )

        excerpts = [c["excerpt"] for c in candidates]
        self.assertTrue(
            any("安事變" in e for e in excerpts),
            f"Tier-3 garble should be a sentence candidate: {excerpts}",
        )

    def test_generate_sentence_candidates_has_correct_fields(self):
        from paddle_pipeline.semantic_ocr_review import generate_sentence_candidates

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE}</p>"
            f"<p>{_SEMANTIC_GARBLE}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_sentence_candidates(
                str(epub), limit=30, min_cjk=6
            )

        self.assertGreater(len(candidates), 0)
        for c in candidates:
            self.assertEqual(c["source"], "sentence_candidate")
            self.assertIn("file", c)
            self.assertIn("excerpt", c)
            self.assertIn("context_before", c)
            self.assertIn("context_after", c)
            self.assertIn("cjk_len", c)

    def test_round_robin_covers_multiple_files(self):
        """Round-robin interleaving ensures all files contribute candidates."""
        from paddle_pipeline.semantic_ocr_review import generate_sentence_candidates

        # Two XHTML files in one EPUB
        body1 = (
            "<html><body>"
            "<p>第一段文字，描述了一些事情。第二段文字，也是正常中文。</p>"
            "<p>第三段文字，包含更多中文内容。第四段文字，正常描述。</p>"
            "</body></html>"
        )
        body2 = (
            "<html><body>"
            "<p>這是另一個文件的內容。這裡有另一句中文。</p>"
            f"<p>{_TIER3_GARBLE}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            # Create multi-file EPUB
            from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
            with ZipFile(epub, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
                archive.writestr(
                    "META-INF/container.xml",
                    '<?xml version="1.0"?>'
                    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                    '<rootfiles>'
                    '<rootfile full-path="EPUB/book.opf" media-type="application/oebps-package+xml"/>'
                    '</rootfiles>'
                    '</container>',
                )
                archive.writestr(
                    "EPUB/book.opf",
                    '<?xml version="1.0" encoding="utf-8"?>'
                    '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                    '<manifest>'
                    '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
                    '<item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>'
                    '</manifest>'
                    '<spine><itemref idref="c1"/><itemref idref="c2"/></spine>'
                    '</package>',
                )
                archive.writestr("EPUB/ch1.xhtml", body1, compress_type=ZIP_DEFLATED)
                archive.writestr("EPUB/ch2.xhtml", body2, compress_type=ZIP_DEFLATED)

            candidates = generate_sentence_candidates(str(epub), limit=20, min_cjk=6)
            files = {c["file"] for c in candidates}
            self.assertGreaterEqual(len(files), 2, f"Round-robin should cover both files, got: {files}")

    def test_looks_like_punctuation_list_filters_book_lists(self):
        from paddle_pipeline.semantic_ocr_review import _looks_like_punctuation_list

        self.assertTrue(_looks_like_punctuation_list(
            "《書名一》《書名二》《書名三》《書名四》《書名五》"
        ))
        self.assertFalse(_looks_like_punctuation_list(
            "這是一段正常的中文文字，包含一些標點符號。"
        ))

    def test_empty_epub_sentence_candidates(self):
        from paddle_pipeline.semantic_ocr_review import generate_sentence_candidates

        body = "<html><body><p>No CJK here.</p></body></html>"
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            _write_epub(epub, body)
            candidates = generate_sentence_candidates(
                str(epub), limit=30, min_cjk=6
            )
        self.assertEqual(len(candidates), 0)

class TestCLI(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.abspath("."))

    def test_json_output_is_valid(self):
        from paddle_pipeline.semantic_ocr_review import main

        body = (
            "<html><body>"
            f"<p>{_NORMAL_CHINESE}</p>"
            f"<p>{_SEMANTIC_GARBLE}</p>"
            "</body></html>"
        )
        with tempfile.TemporaryDirectory() as td:
            epub = Path(td) / "test.epub"
            out = Path(td) / "report.json"
            _write_epub(epub, body)

            main([str(epub), "--json", str(out)])

            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsInstance(data, list)

            # At least one candidate should be generated for the garbled text.
            self.assertGreater(len(data), 0)


if __name__ == "__main__":
    unittest.main()
