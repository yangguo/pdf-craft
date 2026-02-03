import re
import tempfile
from pathlib import Path
from typing import Iterable

from PIL.Image import Image

from ..common import ASSET_TAGS, AssetHub, remove_surrogates
from ..error import OCRError
from ..metering import AbortedCheck, check_aborted
from .ngrams import has_repetitive_ngrams
from .types import DeepSeekOCRSize, DeepSeekOCRVersion, Page, PageLayout


class _GLMOCRFallbackProcessor:
    def __init__(self, tokenizer, image_processor) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, images, text, return_tensors="pt"):
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        text_inputs = self.tokenizer(text, return_tensors=return_tensors)
        merged = dict(image_inputs)
        merged.update(text_inputs)
        return merged

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class PageExtractorNode:
    def __init__(
        self,
        model_path: Path | None = None,
        model_name: str | None = None,
        ocr_version: DeepSeekOCRVersion = "v1",
        local_only: bool = False,
        enable_devices_numbers: Iterable[int] | None = None,
    ) -> None:
        self._model_path: Path | None = model_path
        self._model_name: str | None = model_name
        self._ocr_version: DeepSeekOCRVersion = ocr_version
        self._local_only: bool = local_only
        self._enable_devices_numbers: Iterable[int] | None = enable_devices_numbers
        self._page_extractor = None
        self._v2_model = None
        self._glm_ocr_model = None

    def _get_page_extractor(self):
        if not self._page_extractor:
            # 尽可能推迟 doc-page-extractor 的加载时间
            from doc_page_extractor import create_page_extractor

            kwargs = {
                "model_path": self._model_path,
                "local_only": self._local_only,
                "enable_devices_numbers": self._enable_devices_numbers,
            }
            if self._model_name:
                try:
                    self._page_extractor = create_page_extractor(
                        model_name=self._model_name,
                        **kwargs,
                    )
                except TypeError:
                    # Backward compatibility with older doc-page-extractor
                    self._page_extractor = create_page_extractor(**kwargs)
            else:
                self._page_extractor = create_page_extractor(**kwargs)
        return self._page_extractor

    def _predownload_v2(self, revision: str | None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception as error:  # pragma: no cover - import guard
            raise OCRError(
                "DeepSeek-OCR-2 requires huggingface_hub to predownload models."
            ) from error

        model_name = self._model_name or "deepseek-ai/DeepSeek-OCR-2"
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=str(self._model_path) if self._model_path else None,
            local_files_only=self._local_only,
        )

    def _get_v2_model(self):
        if self._v2_model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:  # pragma: no cover - import guard
                raise OCRError(
                    "DeepSeek-OCR-2 requires transformers and torch to be installed. "
                    "Install with: pip install 'transformers>=4.46' torch"
                ) from error

            if not torch.cuda.is_available():
                raise OCRError(
                    "CUDA is not available for DeepSeek-OCR-2. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )

            model_name = self._model_name or "deepseek-ai/DeepSeek-OCR-2"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(self._model_path) if self._model_path else None,
                local_files_only=self._local_only,
            )

            # Try flash_attention_2 first, fall back to sdpa or eager if unavailable
            attn_impl = self._get_best_attention_implementation()
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                attn_implementation=attn_impl,
                cache_dir=str(self._model_path) if self._model_path else None,
                local_files_only=self._local_only,
            )
            model = model.eval().cuda()
            try:
                model = model.to(torch.bfloat16)
            except Exception:
                model = model.to(torch.float16)

            self._v2_model = (tokenizer, model)
        return self._v2_model

    def _get_best_attention_implementation(self) -> str:
        """Determine the best available attention implementation."""
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass

        try:
            import torch

            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                return "sdpa"
        except ImportError:
            pass

        return "eager"

    def _predownload_glm_ocr(self, revision: str | None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception as error:  # pragma: no cover - import guard
            raise OCRError(
                "GLM-OCR requires huggingface_hub to predownload models."
            ) from error

        model_name = self._model_name or "zai-org/GLM-OCR"
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=str(self._model_path) if self._model_path else None,
            local_files_only=self._local_only,
        )

    def _load_glm_ocr_processor(self, model_name: str):
        cache_dir = str(self._model_path) if self._model_path else None
        local_only = self._local_only
        try:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
        except Exception:
            pass

        try:
            from transformers import AutoImageProcessor, AutoTokenizer
        except ImportError:
            from transformers import AutoTokenizer

            AutoImageProcessor = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )

            if AutoImageProcessor is not None:
                try:
                    image_processor = AutoImageProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        local_files_only=local_only,
                    )
                except Exception:
                    image_processor = None
            else:
                image_processor = None

            if image_processor is None:
                try:
                    from transformers import AutoFeatureExtractor
                except ImportError as feature_error:
                    raise OCRError(
                        "GLM-OCR requires a processor or image processor. "
                        "Failed to load any compatible processor."
                    ) from feature_error

                image_processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=local_only,
                )

            return _GLMOCRFallbackProcessor(tokenizer, image_processor)
        except Exception as fallback_error:  # pragma: no cover - safety net
            raise OCRError(
                "Failed to load GLM-OCR processor. "
                "If you see 'Unrecognized processing class', "
                "upgrade transformers or use a model repo with processor files."
            ) from fallback_error

    def _get_glm_ocr_model(self):
        if self._glm_ocr_model is None:
            try:
                import torch
                from transformers import AutoModelForImageTextToText
            except ImportError as error:  # pragma: no cover - import guard
                raise OCRError(
                    "GLM-OCR requires transformers and torch to be installed. "
                    "Install with: pip install 'transformers>=4.46' torch"
                ) from error

            if not torch.cuda.is_available():
                raise OCRError(
                    "CUDA is not available for GLM-OCR. "
                    "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support."
                )

            model_name = self._model_name or "zai-org/GLM-OCR"
            processor = self._load_glm_ocr_processor(model_name)

            # Try flash_attention_2 first, fall back to sdpa or eager if unavailable
            attn_impl = self._get_best_attention_implementation()
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                attn_implementation=attn_impl,
                cache_dir=str(self._model_path) if self._model_path else None,
                local_files_only=self._local_only,
            )
            model = model.eval().cuda()
            try:
                model = model.to(torch.bfloat16)
            except Exception:
                model = model.to(torch.float16)

            self._glm_ocr_model = (processor, model)
        return self._glm_ocr_model

    def download_models(self, revision: str | None) -> None:
        if self._ocr_version == "v2":
            self._predownload_v2(revision)
            return
        if self._ocr_version == "glm-ocr":
            self._predownload_glm_ocr(revision)
            return
        self._get_page_extractor().download_models(revision)

    def load_models(self) -> None:
        if self._ocr_version == "v2":
            self._get_v2_model()
            return
        if self._ocr_version == "glm-ocr":
            self._get_glm_ocr_model()
            return
        self._get_page_extractor().load_models()

    def image2page(
        self,
        image: Image,
        page_index: int,
        asset_hub: AssetHub,
        ocr_size: DeepSeekOCRSize,
        includes_footnotes: bool,
        includes_raw_image: bool,
        plot_path: Path | None,
        max_tokens: int | None,
        max_output_tokens: int | None,
        device_number: int | None,
        aborted: AbortedCheck,
    ) -> Page:
        if self._ocr_version == "v2":
            return self._image2page_v2(
                image=image,
                page_index=page_index,
                asset_hub=asset_hub,
                includes_raw_image=includes_raw_image,
                aborted=aborted,
                device_number=device_number,
            )

        if self._ocr_version == "glm-ocr":
            return self._image2page_glm_ocr(
                image=image,
                page_index=page_index,
                asset_hub=asset_hub,
                includes_raw_image=includes_raw_image,
                aborted=aborted,
                device_number=device_number,
            )

        from doc_page_extractor import ExtractionContext, plot

        body_layouts: list[PageLayout] = []
        footnotes_layouts: list[PageLayout] = []
        raw_image: Image | None = None

        if includes_raw_image:
            raw_image = image
            image = image.copy()

        with tempfile.TemporaryDirectory() as temp_dir_path:
            context = ExtractionContext(
                check_aborted=aborted,
                max_tokens=max_tokens,
                max_output_tokens=max_output_tokens,
                output_dir_path=temp_dir_path,
            )
            step_index: int = 1
            generator = self._get_page_extractor().extract(
                image=image,
                size=ocr_size,
                stages=2 if includes_footnotes else 1,
                context=context,
                device_number=device_number,
            )
            while True:
                try:
                    image, layouts = next(generator)
                except StopIteration:
                    break
                except Exception as error:
                    raise OCRError(
                        f"Failed to extract page {page_index} layout at stage {step_index}.",
                        page_index=page_index,
                        step_index=step_index,
                    ) from error

                for layout in layouts:
                    ref = self._normalize_text(layout.ref)
                    text = self._normalize_text(layout.text)
                    det = self._normalize_layout_det(image.size, layout.det)

                    if det is None:
                        continue

                    # 检测短模式重复（如 "1.1.1.1."）
                    if has_repetitive_ngrams(
                        text, min_ngram=2, max_ngram=5, repeat_threshold=16
                    ):
                        continue

                    # 检测长模式重复（保守策略）
                    if has_repetitive_ngrams(
                        text, min_ngram=6, max_ngram=20, repeat_threshold=8
                    ):
                        continue

                    hash: str | None = None
                    if ref in ASSET_TAGS:
                        hash = asset_hub.clip(image, det)

                    if step_index == 1:
                        order = len(body_layouts)
                    elif step_index == 2 and ref not in ASSET_TAGS:
                        order = len(footnotes_layouts)
                    else:
                        continue

                    page_layout = PageLayout(
                        ref=ref,
                        det=det,
                        text=text,
                        hash=hash,
                        order=order,
                    )
                    if step_index == 1:
                        body_layouts.append(page_layout)
                    elif step_index == 2 and ref not in ASSET_TAGS:
                        footnotes_layouts.append(page_layout)

                check_aborted(aborted)
                if plot_path is not None:
                    plot_file_path = (
                        plot_path / f"page_{page_index}_stage_{step_index}.png"
                    )
                    image = plot(image.copy(), layouts)
                    image.save(plot_file_path, format="PNG")
                    check_aborted(aborted)

                step_index += 1

            return Page(
                index=page_index,
                image=raw_image,
                body_layouts=body_layouts,
                footnotes_layouts=footnotes_layouts,
                input_tokens=context.input_tokens,
                output_tokens=context.output_tokens,
            )

    def _image2page_v2(
        self,
        image: Image,
        page_index: int,
        asset_hub: AssetHub,
        includes_raw_image: bool,
        aborted: AbortedCheck,
        device_number: int | None,
    ) -> Page:
        import os
        import tempfile

        import torch

        if device_number is not None:
            torch.cuda.set_device(device_number)

        tokenizer, model = self._get_v2_model()
        raw_image: Image | None = None
        if includes_raw_image:
            raw_image = image

        with tempfile.TemporaryDirectory() as temp_dir_path:
            image_file = os.path.join(temp_dir_path, f"page_{page_index}.png")
            output_path = os.path.join(temp_dir_path, "output")
            os.makedirs(output_path, exist_ok=True)
            image.save(image_file, format="PNG")

            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            try:
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=image_file,
                    output_path=output_path,
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=True,
                )
            except Exception as error:
                raise OCRError(
                    f"DeepSeek-OCR-2 inference failed for page {page_index}: {error}",
                    page_index=page_index,
                    step_index=1,
                ) from error

            check_aborted(aborted)

            text = None
            if isinstance(result, str):
                text = result
            elif isinstance(result, dict):
                text = (
                    result.get("markdown")
                    or result.get("text")
                    or result.get("result")
                    or result.get("output")
                )

            if text is None:
                output_dir = Path(output_path)
                if output_dir.exists():
                    candidates = list(output_dir.glob("*.md")) + list(
                        output_dir.glob("*.txt")
                    )
                    if candidates:
                        text = candidates[0].read_text(encoding="utf-8")

            if text is None:
                raise OCRError(
                    f"DeepSeek-OCR-2 did not return text for page {page_index}.",
                    page_index=page_index,
                    step_index=1,
                )

            # Estimate token counts using the tokenizer
            input_tokens, output_tokens = self._estimate_v2_tokens(
                tokenizer, prompt, text
            )

            # Process the markdown text to extract layouts
            body_layouts = self._parse_v2_markdown(
                text=text,
                image=image,
                asset_hub=asset_hub,
            )

            return Page(
                index=page_index,
                image=raw_image,
                body_layouts=body_layouts,
                footnotes_layouts=[],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    def _image2page_glm_ocr(
        self,
        image: Image,
        page_index: int,
        asset_hub: AssetHub,
        includes_raw_image: bool,
        aborted: AbortedCheck,
        device_number: int | None,
    ) -> Page:
        import torch

        if device_number is not None:
            torch.cuda.set_device(device_number)

        processor, model = self._get_glm_ocr_model()
        raw_image: Image | None = None
        if includes_raw_image:
            raw_image = image

        # Convert PIL Image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prepare prompt for GLM-OCR
        prompt = "Text Recognition:"

        try:
            # Prepare inputs using the processor
            inputs = processor(images=image, text=prompt, return_tensors="pt")

            # Move inputs to GPU
            for key in inputs:
                inputs[key] = inputs[key].cuda()

            # Generate output
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=4096)

            # Decode the result
            if hasattr(processor, "decode"):
                result = processor.decode(generated_ids[0], skip_special_tokens=True)
            else:
                result = processor.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
        except Exception as error:
            raise OCRError(
                f"GLM-OCR inference failed for page {page_index}: {error}",
                page_index=page_index,
                step_index=1,
            ) from error

        check_aborted(aborted)

        text = result if isinstance(result, str) else str(result)

        if not text:
            raise OCRError(
                f"GLM-OCR did not return text for page {page_index}.",
                page_index=page_index,
                step_index=1,
            )

        # Estimate token counts using the processor's tokenizer
        input_tokens, output_tokens = self._estimate_glm_ocr_tokens(
            processor, prompt, text
        )

        # Process the markdown text to extract layouts
        # GLM-OCR output format is similar to v2, so reuse the parsing logic
        body_layouts = self._parse_v2_markdown(
            text=text,
            image=image,
            asset_hub=asset_hub,
        )

        return Page(
            index=page_index,
            image=raw_image,
            body_layouts=body_layouts,
            footnotes_layouts=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _estimate_glm_ocr_tokens(
        self,
        processor,
        prompt: str,
        output_text: str,
    ) -> tuple[int, int]:
        """Estimate input and output token counts for GLM-OCR model."""
        try:
            # Input tokens: prompt + image tokens (estimated)
            tokenizer = processor.tokenizer
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            # GLM-OCR uses variable image tokens based on image size
            # Estimate ~1000 tokens for a typical document page image
            estimated_image_tokens = 1000
            input_tokens = prompt_tokens + estimated_image_tokens

            # Output tokens: the generated text
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

            return input_tokens, output_tokens
        except Exception:
            # If tokenization fails, return 0
            return 0, 0

    def _estimate_v2_tokens(
        self,
        tokenizer,
        prompt: str,
        output_text: str,
    ) -> tuple[int, int]:
        """Estimate input and output token counts for v2 model."""
        try:
            # Input tokens: prompt + image tokens (estimated)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            # DeepSeek-OCR-2 uses variable image tokens based on image size
            # Estimate ~1000 tokens for a typical document page image
            estimated_image_tokens = 1000
            input_tokens = prompt_tokens + estimated_image_tokens

            # Output tokens: the generated markdown
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

            return input_tokens, output_tokens
        except Exception:
            # If tokenization fails, return 0
            return 0, 0

    def _parse_v2_markdown(
        self,
        text: str,
        image: Image,
        asset_hub: AssetHub,
    ) -> list[PageLayout]:
        """Parse v2 markdown output into PageLayout objects.

        DeepSeek-OCR-2 outputs markdown that may contain image references.
        This method extracts and processes them.
        """
        import re

        width, height = image.size
        layouts: list[PageLayout] = []

        # Pattern to match image placeholders in markdown: ![...](...)
        # DeepSeek-OCR-2 may output images as base64 or file references
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

        last_end = 0
        order = 0

        for match in image_pattern.finditer(text):
            # Add text before this image
            text_before = text[last_end : match.start()].strip()
            if text_before:
                layouts.append(
                    PageLayout(
                        ref="text",
                        det=(0, 0, width, height),
                        text=self._normalize_text(text_before),
                        order=order,
                        hash=None,
                    )
                )
                order += 1

            # Process the image reference
            alt_text = match.group(1)
            img_src = match.group(2)

            # Check if this is a region reference (e.g., coordinates in the image)
            # For now, we keep the markdown image syntax as-is in the text
            # The image from the page is already available via asset_hub if needed
            layouts.append(
                PageLayout(
                    ref="image",
                    det=(0, 0, width, height),
                    text=f"![{alt_text}]({img_src})",
                    order=order,
                    hash=None,
                )
            )
            order += 1
            last_end = match.end()

        # Add remaining text after last image
        text_after = text[last_end:].strip()
        if text_after:
            layouts.append(
                PageLayout(
                    ref="text",
                    det=(0, 0, width, height),
                    text=self._normalize_text(text_after),
                    order=order,
                    hash=None,
                )
            )
        elif not layouts:
            # No images found, treat entire text as single layout
            layouts.append(
                PageLayout(
                    ref="text",
                    det=(0, 0, width, height),
                    text=self._normalize_text(text),
                    order=0,
                    hash=None,
                )
            )

        return layouts

    def _normalize_text(self, text: str | None) -> str:
        if text is None:
            return ""
        text = remove_surrogates(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_layout_det(
        self,
        size: tuple[int, int],
        det: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        width, height = size
        left, top, right, bottom = det
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(0, min(right, width))
        bottom = max(0, min(bottom, height))

        if left >= right or top >= bottom:
            return None
        return left, top, right, bottom
