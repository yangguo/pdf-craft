import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


class _Response:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content

    def json(self):
        return self._payload


def _result_zip():
    body = (
        "這是一段正常的中文正文內容，用來模擬 MinerU 回傳的 Markdown 結果。"
        "文字需要足夠長，才不會被 ZIP 解析流程視為空白或頁眉殘片。"
        "這裡繼續加入更多句子，確認解析後能產生 layoutParsingResults。"
    )
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("chapter.md", body)
    return data.getvalue()


class TestMineruApiRetries(unittest.TestCase):
    def test_submit_payload_uses_traditional_chinese_language_pack_by_default(self):
        import requests as real_requests

        import paddle_pipeline.mineru_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_payloads = []

            def post(self, *args, **kwargs):
                self.post_payloads.append(kwargs.get("json", {}))
                return _Response(
                    payload={
                        "code": 0,
                        "data": {
                            "batch_id": "batch-1",
                            "file_urls": ["https://upload.example.test/chunk.pdf"],
                        },
                    }
                )

            def put(self, *args, **kwargs):
                return _Response(status_code=204)

            def get(self, url, *args, **kwargs):
                if "extract-results/batch/batch-1" in url:
                    return _Response(
                        payload={
                            "code": 0,
                            "data": {
                                "extract_result": [
                                    {
                                        "state": "done",
                                        "full_zip_url": "https://download.example.test/result.zip",
                                    }
                                ]
                            },
                        }
                    )
                return _Response(content=_result_zip())

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        fd, chunk_path = tempfile.mkstemp(dir=repo_root)
        try:
            os.write(fd, b"%PDF-1.4\n")
            os.close(fd)

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

        self.assertIsNotNone(result)
        self.assertEqual("chinese_cht", fake_requests.post_payloads[0]["language"])

    def test_malformed_submit_payload_retries_then_succeeds(self):
        import requests as real_requests

        import paddle_pipeline.mineru_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.put_calls = 0
                self.batch_poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                if self.post_calls == 1:
                    return _Response(payload={"code": 0, "data": {}})
                return _Response(
                    payload={
                        "code": 0,
                        "data": {
                            "batch_id": "batch-1",
                            "file_urls": ["https://upload.example.test/chunk.pdf"],
                        },
                    }
                )

            def put(self, *args, **kwargs):
                self.put_calls += 1
                return _Response(status_code=204)

            def get(self, url, *args, **kwargs):
                if "extract-results/batch/batch-1" in url:
                    self.batch_poll_calls += 1
                    return _Response(
                        payload={
                            "code": 0,
                            "data": {
                                "extract_result": [
                                    {
                                        "state": "done",
                                        "full_zip_url": "https://download.example.test/result.zip",
                                    }
                                ]
                            },
                        }
                    )
                return _Response(content=_result_zip())

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        fd, chunk_path = tempfile.mkstemp(dir=repo_root)
        try:
            os.write(fd, b"%PDF-1.4\n")
            os.close(fd)

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

        self.assertIsNotNone(result)
        self.assertEqual(fake_requests.post_calls, 2)
        self.assertEqual(fake_requests.put_calls, 1)
        self.assertEqual(fake_requests.batch_poll_calls, 1)

    def test_malformed_poll_payload_retries_without_resubmit(self):
        import requests as real_requests

        import paddle_pipeline.mineru_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.put_calls = 0
                self.batch_poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response(
                    payload={
                        "code": 0,
                        "data": {
                            "batch_id": "batch-1",
                            "file_urls": ["https://upload.example.test/chunk.pdf"],
                        },
                    }
                )

            def put(self, *args, **kwargs):
                self.put_calls += 1
                return _Response(status_code=204)

            def get(self, url, *args, **kwargs):
                if "extract-results/batch/batch-1" in url:
                    self.batch_poll_calls += 1
                    if self.batch_poll_calls == 1:
                        return _Response(payload={"code": 0, "data": {}})
                    return _Response(
                        payload={
                            "code": 0,
                            "data": {
                                "extract_result": [
                                    {
                                        "state": "done",
                                        "full_zip_url": "https://download.example.test/result.zip",
                                    }
                                ]
                            },
                        }
                    )
                return _Response(content=_result_zip())

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        fd, chunk_path = tempfile.mkstemp(dir=repo_root)
        try:
            os.write(fd, b"%PDF-1.4\n")
            os.close(fd)

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

        self.assertIsNotNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.put_calls, 1)
        self.assertEqual(fake_requests.batch_poll_calls, 2)

    def test_zip_download_retry_after_completed_batch_does_not_reupload(self):
        import requests as real_requests

        import paddle_pipeline.mineru_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.put_calls = 0
                self.batch_poll_calls = 0
                self.zip_download_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response(
                    payload={
                        "code": 0,
                        "data": {
                            "batch_id": "batch-1",
                            "file_urls": ["https://upload.example.test/chunk.pdf"],
                        },
                    }
                )

            def put(self, *args, **kwargs):
                self.put_calls += 1
                return _Response(status_code=204)

            def get(self, url, *args, **kwargs):
                if "extract-results/batch/batch-1" in url:
                    self.batch_poll_calls += 1
                    return _Response(
                        payload={
                            "code": 0,
                            "data": {
                                "extract_result": [
                                    {
                                        "state": "done",
                                        "full_zip_url": "https://download.example.test/result.zip",
                                    }
                                ]
                            },
                        }
                    )

                self.zip_download_calls += 1
                if self.zip_download_calls == 1:
                    return _Response(status_code=500)
                return _Response(content=_result_zip())

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        fd, chunk_path = tempfile.mkstemp(dir=repo_root)
        try:
            os.write(fd, b"%PDF-1.4\n")
            os.close(fd)

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

        self.assertIsNotNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.put_calls, 1)
        self.assertEqual(fake_requests.batch_poll_calls, 1)
        self.assertEqual(fake_requests.zip_download_calls, 2)
        pages = result["result"]["layoutParsingResults"]
        self.assertEqual(1, len(pages))
        self.assertIn("正常的中文正文", pages[0]["markdown"]["text"])

    def test_exhausted_zip_download_retries_do_not_reupload_or_resubmit(self):
        import requests as real_requests

        import paddle_pipeline.mineru_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.put_calls = 0
                self.batch_poll_calls = 0
                self.zip_download_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response(
                    payload={
                        "code": 0,
                        "data": {
                            "batch_id": "batch-1",
                            "file_urls": ["https://upload.example.test/chunk.pdf"],
                        },
                    }
                )

            def put(self, *args, **kwargs):
                self.put_calls += 1
                return _Response(status_code=204)

            def get(self, url, *args, **kwargs):
                if "extract-results/batch/batch-1" in url:
                    self.batch_poll_calls += 1
                    return _Response(
                        payload={
                            "code": 0,
                            "data": {
                                "extract_result": [
                                    {
                                        "state": "done",
                                        "full_zip_url": "https://download.example.test/result.zip",
                                    }
                                ]
                            },
                        }
                    )

                self.zip_download_calls += 1
                return _Response(status_code=500)

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        fd, chunk_path = tempfile.mkstemp(dir=repo_root)
        try:
            os.write(fd, b"%PDF-1.4\n")
            os.close(fd)

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

        self.assertIsNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.put_calls, 1)
        self.assertEqual(fake_requests.batch_poll_calls, 1)
        self.assertEqual(fake_requests.zip_download_calls, 5)


class TestReparseCheckpointSentinel(unittest.TestCase):
    def test_reparse_checkpoint_skips_download_when_already_reparsed(self):
        """reparse_checkpoint must return original checkpoint without downloading
        when _mineru_reparsed sentinel is already present."""
        import paddle_pipeline.mineru_api as api

        download_calls = []

        def fake_download(url, token):
            download_calls.append(url)
            return b"zipdata"

        checkpoint = {
            "result": {"layoutParsingResults": [{"markdown": {"text": "page", "images": {}}}]},
            "_mineru_zip_url": "https://example.test/result.zip",
            "_mineru_reparsed": True,
        }

        with mock.patch.object(api, "_download_zip_with_retry", fake_download):
            result = api.reparse_checkpoint(checkpoint, token="tok")

        self.assertEqual([], download_calls)
        self.assertIs(checkpoint, result)

    def test_reparse_checkpoint_sets_sentinel_after_successful_reparse(self):
        """reparse_checkpoint must set _mineru_reparsed=True in the returned
        dict so subsequent runs do not re-download the ZIP."""
        import io
        import zipfile

        import paddle_pipeline.mineru_api as api

        # Build a minimal valid result ZIP
        body = (
            "重新解析後的正文內容，長度超過八十個字元，"
            "確保 _parse_zip 不會將此頁視為空白頁面並丟棄。"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("page.md", body)
        zip_bytes = buf.getvalue()

        checkpoint = {
            "result": {"layoutParsingResults": [{"markdown": {"text": "old", "images": {}}}]},
            "_mineru_zip_url": "https://example.test/result.zip",
        }

        with mock.patch.object(api, "_download_zip_with_retry", return_value=zip_bytes):
            result = api.reparse_checkpoint(checkpoint, token="tok")

        self.assertTrue(result.get("_mineru_reparsed"),
                        "_mineru_reparsed sentinel not set after successful reparse")

    def test_parse_pdf_chunk_result_includes_reparsed_sentinel(self):
        """parse_pdf_chunk must stamp _mineru_reparsed=True on the returned
        checkpoint so that reparse_checkpoint skips the ZIP re-download on
        the very first resume (not only after reparse has run once)."""
        import requests as real_requests

        import paddle_pipeline.mineru_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def post(self, *args, **kwargs):
                return _Response(payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example.test/chunk.pdf"],
                    },
                })

            def put(self, *args, **kwargs):
                return _Response(status_code=204)

            def get(self, url, *args, **kwargs):
                if "extract-results/batch/batch-1" in url:
                    return _Response(payload={
                        "code": 0,
                        "data": {
                            "extract_result": [{
                                "state": "done",
                                "full_zip_url": "https://download.example.test/result.zip",
                            }]
                        },
                    })
                return _Response(content=_result_zip())

        repo_root = Path(__file__).resolve().parents[1]
        fd, chunk_path = tempfile.mkstemp(dir=repo_root)
        try:
            os.write(fd, b"%PDF-1.4\n")
            os.close(fd)

            with mock.patch.object(api, "requests", FakeRequests()):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

        self.assertIsNotNone(result)
        self.assertTrue(
            result.get("_mineru_reparsed"),
            "_mineru_reparsed sentinel missing from parse_pdf_chunk result; "
            "first resume will unnecessarily re-download the ZIP",
        )


class TestStripCheckpointDataUris(unittest.TestCase):
    def test_strip_clears_base64_values_but_preserves_keys(self):
        """strip_checkpoint_data_uris must replace data URI values with empty
        strings so checkpoint files stay small, while keeping image keys so
        _is_sparse_visual_page still recognises pages that have images."""
        import paddle_pipeline.mineru_api as api

        res = {
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {
                            "text": "page one",
                            "images": {
                                "img/fig1.png": "data:image/png;base64,iVBOR...",
                                "img/fig2.jpg": "data:image/jpeg;base64,/9j/4A...",
                            },
                        }
                    },
                    {
                        "markdown": {
                            "text": "page two",
                            "images": {},
                        }
                    },
                ]
            }
        }

        changed = api.strip_checkpoint_data_uris(res)

        self.assertTrue(changed)
        page_images = res["result"]["layoutParsingResults"][0]["markdown"]["images"]
        # Keys must still be present
        self.assertIn("img/fig1.png", page_images)
        self.assertIn("img/fig2.jpg", page_images)
        # Values must be cleared
        self.assertEqual("", page_images["img/fig1.png"])
        self.assertEqual("", page_images["img/fig2.jpg"])

    def test_strip_returns_false_when_no_data_uris_present(self):
        """strip_checkpoint_data_uris must return False when there is nothing
        to strip (avoids a spurious checkpoint re-save on resume)."""
        import paddle_pipeline.mineru_api as api

        res = {
            "result": {
                "layoutParsingResults": [
                    {"markdown": {"text": "page", "images": {"img/fig1.png": ""}}},
                ]
            }
        }

        changed = api.strip_checkpoint_data_uris(res)

        self.assertFalse(changed)

    def test_stripped_image_keys_prevent_sparse_visual_page_fallback(self):
        """A page whose images values were stripped to '' must still be treated
        as a page-with-images (non-sparse) by _is_sparse_visual_page."""
        from paddle_pipeline.page_image_fallback import _is_sparse_visual_page

        # This mimics a stripped MinerU checkpoint entry
        images = {"img/cover.png": ""}
        markdown_text = "# 序言"

        result = _is_sparse_visual_page(markdown_text, images)

        self.assertFalse(result,
                         "_is_sparse_visual_page should return False when images dict is non-empty")


if __name__ == "__main__":
    unittest.main()
