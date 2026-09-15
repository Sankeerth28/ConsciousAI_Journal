"""Explicit opt-in smoke test for genuine Hugging Face model loading and inference.

DO NOT RUN IN NORMAL CI OR OFFLINE DEV ENVIRONMENTS.
This test will genuinely download model checkpoints over the network and execute
inference on the local machine. It requires:
1. `pip install -e '.[ai]'`
2. Network access to huggingface.co
3. Environment variable `RUN_REAL_HF_SMOKE_TESTS=1`
"""

import os

import pytest

from app.ai.providers.huggingface import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceEmotionClassifier,
    HuggingFaceLLMProvider,
)

# Explicit gating: will be skipped in normal CI / default pytest invocations
pytestmark = [
    pytest.mark.real_ai,
    pytest.mark.skipif(
        os.environ.get("RUN_REAL_HF_SMOKE_TESTS") != "1",
        reason="Real Hugging Face smoke test skipped. Set RUN_REAL_HF_SMOKE_TESTS=1 to run genuine model download & inference.",
    ),
]


class TestRealHuggingFaceSmoke:
    """Smoke test that executes genuine model downloads and real forward passes when opted-in."""

    def test_real_emotion_classifier_inference(self):
        """Genuine forward-pass on a lightweight or zero-shot classifier."""
        classifier = HuggingFaceEmotionClassifier(
            model_name="typeform/distilbert-base-uncased-mnli"
        )
        result = classifier.classify("I feel so excited and grateful for this sunny day!")

        assert result is not None
        assert result.top_emotion in result.candidate_labels
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.emotions) > 0

    def test_real_embedding_provider_inference(self):
        """Genuine forward-pass generating real dense embeddings."""
        provider = HuggingFaceEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            device="cpu",
        )
        result = provider.embed(["This is a test journal reflection."])

        assert result is not None
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == 384
        assert result.dimension == 384

    def test_real_llm_provider_inference(self):
        """Genuine text generation using a lightweight seq2seq model."""
        llm = HuggingFaceLLMProvider(
            model_name="google/flan-t5-small",
            device="cpu",
        )
        prompt = "Answer in one sentence: What is the purpose of a reflective journal?"
        output = llm.generate(prompt, max_tokens=64, temperature=0.1)

        assert output is not None
        assert isinstance(output, str)
        assert len(output.strip()) > 5
