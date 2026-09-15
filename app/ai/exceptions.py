"""AI-specific exception classes."""

from __future__ import annotations


class AIError(Exception):
    """Base exception for all AI-related errors."""


class ProviderUnavailableError(AIError):
    """Raised when an AI provider or its optional dependencies are not installed or reachable."""


class ModelLoadError(AIError):
    """Raised when an AI model fails to initialize or load weights."""


class InferenceError(AIError):
    """Raised when a model inference call fails during execution."""


class SafetyViolationError(AIError):
    """Raised when input triggers a critical safety violation."""
