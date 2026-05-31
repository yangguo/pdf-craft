import io
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
        with tempfile.NamedTemporaryFile(dir=repo_root) as chunk:
            chunk.write(b"%PDF-1.4\n")
            chunk.flush()

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk.name, "token")

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
        with tempfile.NamedTemporaryFile(dir=repo_root) as chunk:
            chunk.write(b"%PDF-1.4\n")
            chunk.flush()

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk.name, "token")

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
        with tempfile.NamedTemporaryFile(dir=repo_root) as chunk:
            chunk.write(b"%PDF-1.4\n")
            chunk.flush()

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk.name, "token")

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
        with tempfile.NamedTemporaryFile(dir=repo_root) as chunk:
            chunk.write(b"%PDF-1.4\n")
            chunk.flush()

            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk.name, "token")

        self.assertIsNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.put_calls, 1)
        self.assertEqual(fake_requests.batch_poll_calls, 1)
        self.assertEqual(fake_requests.zip_download_calls, 5)


if __name__ == "__main__":
    unittest.main()
