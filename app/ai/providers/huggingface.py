"""Lazy-loaded Hugging Face providers using modern transformers pipelines."""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.ai.exceptions import InferenceError, ModelLoadError, ProviderUnavailableError
from app.ai.schemas import (
    EmbeddingResult,
    EmotionResult,
    EmotionScore,
    ValueResult,
    ValueScore,
)

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect whether CUDA GPU is available, defaulting to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _resolve_device(device_pref: str | None) -> str:
    """Resolve device preference ('auto', 'cpu', 'cuda') safely."""
    if not device_pref or device_pref.strip().lower() == "auto":
        return _detect_device()
    norm = device_pref.strip().lower()
    if norm in ("cuda", "gpu"):
        return "cuda"
    return "cpu"


def _handle_inference_error(exc: Exception, model_name: str) -> None:
    """Classify and raise descriptive runtime errors including OOM conditions."""
    exc_type = type(exc).__name__
    if "OutOfMemoryError" in exc_type or isinstance(exc, MemoryError):
        msg = (
            f"Out of memory error during inference on '{model_name}'. "
            "Consider reducing batch size, using a smaller model, or switching device to CPU."
        )
        logger.error(msg)
        raise InferenceError(msg) from exc

    msg = f"Inference failed on model '{model_name}': {exc_type}"
    logger.error("Inference failed on model '%s'", model_name)
    raise InferenceError(msg) from exc


def _handle_load_error(exc: Exception, model_name: str, model_type: str = "model") -> None:
    """Classify and raise descriptive loading errors: network, incompatible model, OOM."""
    exc_type = type(exc).__name__
    exc_str = str(exc).lower()

    if "outofmemoryerror" in exc_str or "out of memory" in exc_str or isinstance(exc, MemoryError):
        msg = (
            f"Out of memory error while loading {model_type} '{model_name}'. "
            "Consider switching device to CPU or using a smaller model checkpoint."
        )
    elif (
        any(
            kw in exc_str
            for kw in (
                "connection",
                "offline",
                "network",
                "timeout",
                "dns",
                "could not resolve",
                "404 client error",
                "repository not found",
                "entry not found",
            )
        )
        or "connectionerror" in exc_type.lower()
        or "timeout" in exc_type.lower()
    ):
        msg = (
            f"Network or remote connectivity failure loading {model_type} '{model_name}'. "
            "Verify internet access, Hugging Face Hub availability, or model identifier."
        )
    elif any(
        kw in exc_str
        for kw in ("config.json", "incompatible", "unsupported", "format", "weights not found")
    ):
        msg = f"Incompatible architecture or missing weights configuration for {model_type} '{model_name}': {exc_type}."
    else:
        msg = f"Failed to load {model_type} '{model_name}': {exc_type}"

    logger.error("Failed to load %s '%s': %s", model_type, model_name, exc_type)
    raise ModelLoadError(msg) from exc


class HuggingFaceLLMProvider:
    """Hugging Face text generation provider with thread-safe lazy pipeline loading."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        hf_token: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._hf_token = hf_token or None
        self._device = _resolve_device(device)
        self._pipeline: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_pipeline(self) -> Any:
        """Lazy load the transformers pipeline with double-checked thread safety."""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                from transformers import pipeline
            except ImportError as exc:
                msg = (
                    "transformers and torch are required for HuggingFaceLLMProvider. "
                    "Install with: pip install -e '.[ai]'"
                )
                logger.error(
                    "transformers/torch not installed; HuggingFaceLLMProvider unavailable."
                )
                raise ProviderUnavailableError(msg) from exc

            try:
                device_id = 0 if self._device == "cuda" else -1
                logger.info("Loading LLM pipeline '%s' on %s...", self._model_name, self._device)
                self._pipeline = pipeline(
                    "text2text-generation",
                    model=self._model_name,
                    device=device_id,
                    token=self._hf_token,
                )
                return self._pipeline
            except Exception as exc:
                _handle_load_error(exc, self._model_name, "LLM model")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        pipe = self._ensure_pipeline()
        try:
            output = pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
            )
            if output and isinstance(output, list) and "generated_text" in output[0]:
                return str(output[0]["generated_text"]).strip()
            return str(output).strip()
        except Exception as exc:
            _handle_inference_error(exc, self._model_name)
            raise  # Unreachable, but satisfies linters


class HuggingFaceEmbeddingProvider:
    """Hugging Face dense vector embedding provider with thread-safe lazy loading."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._device = _resolve_device(device)
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_model(self) -> Any:
        """Lazy load the sentence transformer model with double-checked thread safety."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                msg = (
                    "sentence-transformers is required for HuggingFaceEmbeddingProvider. "
                    "Install with: pip install -e '.[ai]'"
                )
                logger.error("sentence-transformers not installed; provider unavailable.")
                raise ProviderUnavailableError(msg) from exc

            try:
                logger.info("Loading embedding model '%s' on %s...", self._model_name, self._device)
                self._model = SentenceTransformer(self._model_name, device=self._device)
                return self._model
            except Exception as exc:
                _handle_load_error(exc, self._model_name, "embedding model")

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=[], dimension=self._dimension, model_name=self._model_name
            )

        model = self._ensure_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            vectors = [row.tolist() for row in embeddings]
            return EmbeddingResult(
                vectors=vectors,
                dimension=len(vectors[0]) if vectors else self._dimension,
                model_name=self._model_name,
            )
        except Exception as exc:
            _handle_inference_error(exc, self._model_name)
            raise


class HuggingFaceEmotionClassifier:
    """Zero-shot emotion classification provider with thread-safe lazy pipeline loading."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = _resolve_device(device)
        self._pipeline: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                from transformers import pipeline
            except ImportError as exc:
                msg = (
                    "transformers and torch are required for HuggingFaceEmotionClassifier. "
                    "Install with: pip install -e '.[ai]'"
                )
                logger.error(
                    "transformers/torch not installed; HuggingFaceEmotionClassifier unavailable."
                )
                raise ProviderUnavailableError(msg) from exc

            try:
                device_id = 0 if self._device == "cuda" else -1
                logger.info(
                    "Loading zero-shot emotion classifier '%s' on %s...",
                    self._model_name,
                    self._device,
                )
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self._model_name,
                    device=device_id,
                )
                return self._pipeline
            except Exception as exc:
                _handle_load_error(exc, self._model_name, "emotion classifier")

    def classify(
        self,
        text: str,
        candidate_labels: list[str] | None = None,
    ) -> EmotionResult:
        from app.ai.providers.mock import DEFAULT_EMOTIONS

        labels = candidate_labels or DEFAULT_EMOTIONS
        pipe = self._ensure_pipeline()

        try:
            result = pipe(text, candidate_labels=labels, multi_label=False)
            res_labels = result["labels"]
            res_scores = result["scores"]

            scores = [
                EmotionScore(label=lbl, score=round(float(sc), 4))
                for lbl, sc in zip(res_labels, res_scores, strict=False)
            ]
            top = res_labels[0] if res_labels else labels[0]
            confidence = float(res_scores[0]) if res_scores else 0.5

            return EmotionResult(
                top_emotion=top,
                emotions=scores,
                confidence=confidence,
                model_name=self._model_name,
            )
        except Exception as exc:
            _handle_inference_error(exc, self._model_name)
            raise


class HuggingFaceValueClassifier:
    """Zero-shot core value classification provider with thread-safe lazy pipeline loading."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = _resolve_device(device)
        self._pipeline: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                from transformers import pipeline
            except ImportError as exc:
                msg = (
                    "transformers and torch are required for HuggingFaceValueClassifier. "
                    "Install with: pip install -e '.[ai]'"
                )
                logger.error(
                    "transformers/torch not installed; HuggingFaceValueClassifier unavailable."
                )
                raise ProviderUnavailableError(msg) from exc

            try:
                device_id = 0 if self._device == "cuda" else -1
                logger.info(
                    "Loading zero-shot value classifier '%s' on %s...",
                    self._model_name,
                    self._device,
                )
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self._model_name,
                    device=device_id,
                )
                return self._pipeline
            except Exception as exc:
                _handle_load_error(exc, self._model_name, "value classifier")

    def classify(
        self,
        text: str,
        candidate_labels: list[str] | None = None,
    ) -> ValueResult:
        from app.ai.providers.mock import DEFAULT_VALUES

        labels = candidate_labels or DEFAULT_VALUES
        pipe = self._ensure_pipeline()

        try:
            result = pipe(text, candidate_labels=labels, multi_label=False)
            res_labels = result["labels"]
            res_scores = result["scores"]

            scores = [
                ValueScore(label=lbl, score=round(float(sc), 4))
                for lbl, sc in zip(res_labels, res_scores, strict=False)
            ]
            top = res_labels[0] if res_labels else labels[0]
            confidence = float(res_scores[0]) if res_scores else 0.5

            return ValueResult(
                top_value=top,
                values=scores,
                confidence=confidence,
                model_name=self._model_name,
            )
        except Exception as exc:
            _handle_inference_error(exc, self._model_name)
            raise
