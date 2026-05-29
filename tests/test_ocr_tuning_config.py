import os
import sys
import unittest
from importlib import reload
from pathlib import Path


def _load_fresh_modules():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for name in list(sys.modules):
        if name == "paddle_pipeline" or name.startswith("paddle_pipeline."):
            del sys.modules[name]
    import paddle_pipeline.config as config
    import paddle_pipeline.paddle_api as paddle_api
    return config, paddle_api


class TestOcrTuningConfig(unittest.TestCase):
    def test_paddle_optional_payload_default_does_not_include_unset_optional_flags(self):
        os.environ.pop("PADDLE_USE_LAYOUT_DETECTION", None)
        config, paddle_api = _load_fresh_modules()
        reload(config)
        reload(paddle_api)
        payload = paddle_api.build_paddle_optional_payload()
        self.assertIn("useDocOrientationClassify", payload)
        self.assertIn("useDocUnwarping", payload)
        self.assertIn("layoutThreshold", payload)
        self.assertNotIn("useLayoutDetection", payload)

    def test_paddle_optional_payload_overrides(self):
        _, paddle_api = _load_fresh_modules()
        payload = paddle_api.build_paddle_optional_payload(
            {"temperature": 0.05, "useLayoutDetection": False}
        )
        self.assertEqual(payload["temperature"], 0.05)
        self.assertEqual(payload["useLayoutDetection"], False)

