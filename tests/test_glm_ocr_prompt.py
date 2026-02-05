import sys
import types
import unittest
from pathlib import Path

_pdf_craft_stub = types.ModuleType("pdf_craft")
_pdf_craft_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "pdf_craft")]
sys.modules.setdefault("pdf_craft", _pdf_craft_stub)

_epub_stub = types.ModuleType("epub_generator")


class _DummyEpub:  # pragma: no cover - import shim
    pass


_epub_stub.BookMeta = _DummyEpub
_epub_stub.LaTeXRender = _DummyEpub
_epub_stub.TableRender = _DummyEpub
sys.modules.setdefault("epub_generator", _epub_stub)

_openai_stub = types.ModuleType("openai")
_openai_stub.OpenAI = object
sys.modules.setdefault("openai", _openai_stub)

_openai_types_stub = types.ModuleType("openai.types")
_openai_types_chat_stub = types.ModuleType("openai.types.chat")


class _DummyChatCompletionMessageParam:  # pragma: no cover - import shim
    pass


_openai_types_chat_stub.ChatCompletionMessageParam = _DummyChatCompletionMessageParam
sys.modules.setdefault("openai.types", _openai_types_stub)
sys.modules.setdefault("openai.types.chat", _openai_types_chat_stub)

_tiktoken_stub = types.ModuleType("tiktoken")


class _DummyEncoding:  # pragma: no cover - import shim
    def encode(self, text: str, *args, **kwargs):
        return [0]


def _dummy_get_encoding(name: str):  # pragma: no cover - import shim
    return _DummyEncoding()


_tiktoken_stub.Encoding = _DummyEncoding
_tiktoken_stub.get_encoding = _dummy_get_encoding
sys.modules.setdefault("tiktoken", _tiktoken_stub)

_pil_stub = types.ModuleType("PIL")
_pil_image_stub = types.ModuleType("PIL.Image")


class _DummyImage:  # pragma: no cover - import shim
    pass


_pil_image_stub.Image = _DummyImage
sys.modules.setdefault("PIL", _pil_stub)
sys.modules.setdefault("PIL.Image", _pil_image_stub)

from pdf_craft.pdf.page_extractor import PageExtractorNode  # noqa: E402


class _DummyTokenizer:
    def __init__(self, image_token: str | None = None) -> None:
        self.image_token = image_token


class _DummyProcessor:
    def __init__(self, image_token: str | None = None, tokenizer: _DummyTokenizer | None = None) -> None:
        self.image_token = image_token
        self.tokenizer = tokenizer


class TestGlmOcrPrompt(unittest.TestCase):
    def test_uses_processor_image_token(self):
        processor = _DummyProcessor(image_token="<img>")
        prompt = PageExtractorNode._build_glm_ocr_prompt(processor)
        self.assertTrue(prompt.startswith("<img>"))

    def test_uses_tokenizer_image_token(self):
        processor = _DummyProcessor(image_token=None, tokenizer=_DummyTokenizer("<tok>"))
        prompt = PageExtractorNode._build_glm_ocr_prompt(processor)
        self.assertTrue(prompt.startswith("<tok>"))

    def test_falls_back_to_default_image_token(self):
        processor = _DummyProcessor(image_token=None, tokenizer=_DummyTokenizer(None))
        prompt = PageExtractorNode._build_glm_ocr_prompt(processor)
        self.assertTrue(prompt.startswith("<|image|>"))
