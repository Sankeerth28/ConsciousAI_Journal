"""Unit tests for Hugging Face providers, lazy loading, and error boundaries."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.ai.exceptions import InferenceError, ProviderUnavailableError
from app.ai.interfaces import (
    EmbeddingProvider,
    EmotionClassifier,
    LLMProvider,
    ValueClassifier,
)
from app.ai.providers.huggingface import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceEmotionClassifier,
    HuggingFaceLLMProvider,
    HuggingFaceValueClassifier,
    _detect_device,
    _resolve_device,
)


class TestHuggingFaceProtocolsAndLazyLoading:
    """Verify protocol compliance and guarantee zero loading on instantiation."""

    def test_huggingface_llm_satisfies_protocol(self):
        llm = HuggingFaceLLMProvider(model_name="google/flan-t5-large")
        assert isinstance(llm, LLMProvider)
        assert llm.model_name == "google/flan-t5-large"
        assert llm._pipeline is None
        assert hasattr(llm, "_lock")

    def test_huggingface_embedding_satisfies_protocol(self):
        emb = HuggingFaceEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert isinstance(emb, EmbeddingProvider)
        assert emb.dimension == 384
        assert emb._model is None
        assert hasattr(emb, "_lock")

    def test_huggingface_emotion_satisfies_protocol(self):
        emo = HuggingFaceEmotionClassifier(model_name="facebook/bart-large-mnli")
        assert isinstance(emo, EmotionClassifier)
        assert emo._pipeline is None
        assert hasattr(emo, "_lock")

    def test_huggingface_value_satisfies_protocol(self):
        val = HuggingFaceValueClassifier(model_name="facebook/bart-large-mnli")
        assert isinstance(val, ValueClassifier)
        assert val._pipeline is None
        assert hasattr(val, "_lock")


class TestDeviceResolutionAndLifecycle:
    """Test device selection configuration and resolution."""

    def test_detect_device_without_cuda(self):
        device = _detect_device()
        assert device in ("cpu", "cuda")

    def test_resolve_device_explicit(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("gpu") == "cuda"
        assert _resolve_device("auto") in ("cpu", "cuda")
        assert _resolve_device(None) in ("cpu", "cuda")

    def test_provider_device_configuration(self):
        llm = HuggingFaceLLMProvider(device="cpu")
        assert llm._device == "cpu"
        emb = HuggingFaceEmbeddingProvider(device="cpu")
        assert emb._device == "cpu"


class TestMissingDependenciesHandling:
    """Verify descriptive ProviderUnavailableError when optional packages are missing."""

    def test_llm_missing_transformers_raises_provider_unavailable(self):
        llm = HuggingFaceLLMProvider()
        with (
            patch.dict(sys.modules, {"transformers": None}),
            pytest.raises(ProviderUnavailableError, match="transformers and torch are required"),
        ):
            llm.generate("Hello")

    def test_embedding_missing_sentence_transformers_raises_provider_unavailable(self):
        emb = HuggingFaceEmbeddingProvider()
        with (
            patch.dict(sys.modules, {"sentence_transformers": None}),
            pytest.raises(ProviderUnavailableError, match="sentence-transformers is required"),
        ):
            emb.embed(["Hello"])

    def test_emotion_missing_transformers_raises_provider_unavailable(self):
        emo = HuggingFaceEmotionClassifier()
        with (
            patch.dict(sys.modules, {"transformers": None}),
            pytest.raises(ProviderUnavailableError, match="transformers and torch are required"),
        ):
            emo.classify("Hello")

    def test_value_missing_transformers_raises_provider_unavailable(self):
        val = HuggingFaceValueClassifier()
        with (
            patch.dict(sys.modules, {"transformers": None}),
            pytest.raises(ProviderUnavailableError, match="transformers and torch are required"),
        ):
            val.classify("Hello")


class TestHuggingFacePipelineMockInference:
    """Test pipeline response parsing and error handling without downloading weights."""

    def test_llm_pipeline_output_parsing(self):
        llm = HuggingFaceLLMProvider()
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "Validation sentence. What is on your mind?"}]
        llm._pipeline = mock_pipe

        result = llm.generate("Reflective prompt")
        assert result == "Validation sentence. What is on your mind?"
        mock_pipe.assert_called_once()

    def test_llm_out_of_memory_handling(self):
        llm = HuggingFaceLLMProvider()
        mock_pipe = MagicMock()
        mock_pipe.side_effect = MemoryError("CUDA out of memory")
        llm._pipeline = mock_pipe

        with pytest.raises(InferenceError, match="Out of memory error"):
            llm.generate("Reflective prompt")

    def test_emotion_pipeline_output_parsing(self):
        emo = HuggingFaceEmotionClassifier()
        mock_pipe = MagicMock()
        mock_pipe.return_value = {
            "labels": ["hopeful", "calm", "sad"],
            "scores": [0.82, 0.12, 0.06],
        }
        emo._pipeline = mock_pipe

        res = emo.classify("I feel optimistic about tomorrow.")
        assert res.top_emotion == "hopeful"
        assert res.confidence == 0.82
        assert len(res.emotions) == 3
        assert res.emotions[0].label == "hopeful"

    def test_value_pipeline_output_parsing(self):
        val = HuggingFaceValueClassifier()
        mock_pipe = MagicMock()
        mock_pipe.return_value = {
            "labels": ["honesty", "courage"],
            "scores": [0.78, 0.22],
        }
        val._pipeline = mock_pipe

        res = val.classify("I told the hard truth.")
        assert res.top_value == "honesty"
        assert res.confidence == 0.78
        assert len(res.values) == 2
