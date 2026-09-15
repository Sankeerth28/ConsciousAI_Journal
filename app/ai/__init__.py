"""AI package exports — interfaces, schemas, services, pipeline, and exceptions."""

from __future__ import annotations

from app.ai.embeddings import EmbeddingService
from app.ai.emotion_classifier import EmotionClassificationService
from app.ai.exceptions import (
    AIError,
    InferenceError,
    ModelLoadError,
    ProviderUnavailableError,
    SafetyViolationError,
)
from app.ai.interfaces import (
    EmbeddingProvider,
    EmotionClassifier,
    LLMProvider,
    ValueClassifier,
    VectorStore,
)
from app.ai.pipeline import JournalReflectionPipeline
from app.ai.providers.mock import MockVectorStore
from app.ai.providers.vector_store import LocalVectorStore
from app.ai.reflection_engine import ReflectionEngine
from app.ai.safety import SafetyInterceptor
from app.ai.schemas import (
    EmbeddingResult,
    EmotionResult,
    EmotionScore,
    MemoryRetrievalResult,
    MemorySearchResult,
    OutputSafetyCheckResult,
    PersonaEnum,
    ReflectionPipelineResult,
    ReflectionResult,
    SafetyCheckResult,
    ValueResult,
    ValueScore,
    VectorRecord,
    VectorSearchResult,
)
from app.ai.value_classifier import ValueClassificationService

__all__ = [
    "AIError",
    "EmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingService",
    "EmotionClassificationService",
    "EmotionClassifier",
    "EmotionResult",
    "EmotionScore",
    "InferenceError",
    "JournalReflectionPipeline",
    "LLMProvider",
    "LocalVectorStore",
    "MemoryRetrievalResult",
    "MemorySearchResult",
    "MockVectorStore",
    "ModelLoadError",
    "OutputSafetyCheckResult",
    "PersonaEnum",
    "ProviderUnavailableError",
    "ReflectionEngine",
    "ReflectionPipelineResult",
    "ReflectionResult",
    "SafetyCheckResult",
    "SafetyInterceptor",
    "SafetyViolationError",
    "ValueClassificationService",
    "ValueClassifier",
    "ValueResult",
    "ValueScore",
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
]
