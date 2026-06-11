import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


class _Response:
    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@contextmanager
def _temporary_chunk_file(repo_root: Path):
    """Create a closed temp PDF file so tests work on Windows and Unix."""
    with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
        chunk_path = Path(temp_dir) / "chunk.pdf"
        chunk_path.write_bytes(b"%PDF-1.4\n")
        yield str(chunk_path)


class TestPaddleApiRetries(unittest.TestCase):
    def test_exhausted_malformed_submit_payload_retries_return_none(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                if self.post_calls == 1:
                    return _Response({"data": {}})  # missing jobId
                return _Response({"data": {"jobId": "job-1"}})

            def get(self, url, *args, **kwargs):
                if url.endswith("/job-1"):
                    self.poll_calls += 1
                    return _Response(
                        {
                            "data": {
                                "state": "done",
                                "extractProgress": {"extractedPages": 1},
                                "resultUrl": {"jsonUrl": "https://example.test/result.jsonl"},
                            }
                        }
                    )
                return _Response(text=json.dumps({"result": {"layoutParsingResults": []}}))

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertEqual(fake_requests.post_calls, 2)
        self.assertEqual(fake_requests.poll_calls, 1)
        self.assertIsNotNone(result)

    def test_malformed_poll_payload_retries_without_resubmitting(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response({"data": {"jobId": "job-1"}})

            def get(self, url, *args, **kwargs):
                if url.endswith("/job-1"):
                    self.poll_calls += 1
                    if self.poll_calls == 1:
                        return _Response({"data": {}})  # missing state/result fields
                    return _Response(
                        {
                            "data": {
                                "state": "done",
                                "extractProgress": {"extractedPages": 1},
                                "resultUrl": {"jsonUrl": "https://example.test/result.jsonl"},
                            }
                        }
                    )
                return _Response(text=json.dumps({"result": {"layoutParsingResults": []}}))

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.poll_calls, 2)
        self.assertIsNotNone(result)

    def test_job_submission_transport_failure_does_not_retry(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                raise real_requests.exceptions.RequestException("submit timeout")

            def get(self, url, *args, **kwargs):
                self.poll_calls += 1
                return _Response()

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertIsNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.poll_calls, 0)

    def test_malformed_submit_payload_retries_before_giving_up(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.get_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response({"data": {}})

            def get(self, *args, **kwargs):
                self.get_calls += 1
                return _Response()

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertIsNone(result)
        self.assertEqual(fake_requests.post_calls, 5)
        self.assertEqual(fake_requests.get_calls, 0)

    def test_job_poll_retry_does_not_resubmit_job(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response({"data": {"jobId": "job-1"}})

            def get(self, url, *args, **kwargs):
                if url.endswith("/job-1"):
                    self.poll_calls += 1
                    if self.poll_calls == 1:
                        raise real_requests.exceptions.RequestException("temporary poll error")
                    return _Response(
                        {
                            "data": {
                                "state": "done",
                                "extractProgress": {"extractedPages": 1},
                                "resultUrl": {"jsonUrl": "https://example.test/result.jsonl"},
                            }
                        }
                    )

                result_line = {
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "ok", "images": {}}}
                        ]
                    }
                }
                return _Response(text=json.dumps(result_line))

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.poll_calls, 2)
        self.assertEqual(
            result,
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "ok", "images": {}}}
                    ]
                }
            },
        )

    def test_result_json_download_retry_does_not_resubmit_job(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.json_download_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response({"data": {"jobId": "job-1"}})

            def get(self, url, *args, **kwargs):
                if url.endswith("/job-1"):
                    return _Response(
                        {
                            "data": {
                                "state": "done",
                                "extractProgress": {"extractedPages": 1},
                                "resultUrl": {"jsonUrl": "https://example.test/result.jsonl"},
                            }
                        }
                    )

                self.json_download_calls += 1
                if self.json_download_calls == 1:
                    raise real_requests.exceptions.RequestException("temporary JSON error")

                result_line = {
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": "ok", "images": {}}}
                        ]
                    }
                }
                return _Response(text=json.dumps(result_line))

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.json_download_calls, 2)
        self.assertEqual(
            result,
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "ok", "images": {}}}
                    ]
                }
            },
        )

    def test_exhausted_result_json_download_retries_do_not_resubmit_job(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.json_download_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response({"data": {"jobId": "job-1"}})

            def get(self, url, *args, **kwargs):
                if url.endswith("/job-1"):
                    return _Response(
                        {
                            "data": {
                                "state": "done",
                                "extractProgress": {"extractedPages": 1},
                                "resultUrl": {"jsonUrl": "https://example.test/result.jsonl"},
                            }
                        }
                    )

                self.json_download_calls += 1
                raise real_requests.exceptions.RequestException("persistent JSON error")

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertIsNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.json_download_calls, 5)

    def test_exhausted_job_poll_retries_do_not_resubmit_job(self):
        import requests as real_requests

        import paddle_pipeline.paddle_api as api

        if api.requests is None:
            self.skipTest("requests dependency not installed")

        class FakeRequests:
            exceptions = real_requests.exceptions

            def __init__(self):
                self.post_calls = 0
                self.poll_calls = 0

            def post(self, *args, **kwargs):
                self.post_calls += 1
                return _Response({"data": {"jobId": "job-1"}})

            def get(self, url, *args, **kwargs):
                self.poll_calls += 1
                raise real_requests.exceptions.RequestException("persistent poll error")

        fake_requests = FakeRequests()
        repo_root = Path(__file__).resolve().parents[1]
        with _temporary_chunk_file(repo_root) as chunk_path:
            with mock.patch.object(api, "requests", fake_requests):
                with mock.patch.object(api.time, "sleep", return_value=None):
                    result = api.parse_pdf_chunk(chunk_path, "token")

        self.assertIsNone(result)
        self.assertEqual(fake_requests.post_calls, 1)
        self.assertEqual(fake_requests.poll_calls, 5)


if __name__ == "__main__":
    unittest.main()
