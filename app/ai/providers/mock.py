"""Deterministic mock AI providers for offline testing and development."""

from __future__ import annotations

import hashlib
import math

from app.ai.schemas import (
    EmbeddingResult,
    EmotionResult,
    EmotionScore,
    ValueResult,
    ValueScore,
    VectorRecord,
    VectorSearchResult,
)

DEFAULT_EMOTIONS = ["happy", "sad", "angry", "hopeful", "guilty", "calm", "anxious", "confused"]
DEFAULT_VALUES = ["honesty", "trust", "communication", "regret", "growth", "forgiveness", "courage"]


class MockLLMProvider:
    """Mock LLM provider generating deterministic reflective responses."""

    def __init__(self, model_name: str = "mock-flan-t5") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Return a reflective response based on persona and detected emotions in prompt."""
        lines = prompt.splitlines()
        header = lines[0].lower() if lines else ""
        lower = prompt.lower()

        # Isolate persona from header or explicit 'persona:' tag (avoid matching guidelines)
        is_coach = "coach" in header or "action-oriented" in header or "persona: coach" in lower or ("coach" in lower and "licensed therapist" not in lower)
        is_neutral = "neutral" in header or "factual" in header or "concise" in header or "persona: neutral" in lower
        is_therapist = "therapist" in header or "contemplative" in header or "persona: therapist" in lower

        # Detect emotion mentioned in theme clause
        detected_emotion = None
        for emo in ["hopeful", "happy", "sad", "angry", "guilty", "calm", "anxious", "confused"]:
            if f"emotion: {emo}" in lower or f"detected emotion: {emo}" in lower:
                detected_emotion = emo
                break

        if is_coach:
            if detected_emotion in ("hopeful", "happy"):
                return "Building on this positive momentum shows dedication to your path. What is one concrete action you want to take next to keep progressing?"
            if detected_emotion in ("anxious", "confused", "sad"):
                return "Navigating competing priorities can be challenging, but recognizing your limits shows strong agency. What is one single, tangible action you can focus on first?"
            return "Taking proactive steps shows dedication to your path. What is one tangible action you want to take next?"

        if is_neutral:
            if detected_emotion:
                return f"The entry records clear observations and notes feelings of being {detected_emotion}. What specific aspects of this experience are most relevant to consider moving forward?"
            return "The entry reflects observations on your current situation. What aspects of this remain to be considered?"

        if is_therapist:
            if detected_emotion in ("hopeful", "calm"):
                return "Noticing what brings you a sense of hope offers valuable insight into what nourishes you. What feelings arise when you reflect on this progress?"
            return "It seems there are several layered emotions beneath this experience. What feels most important to acknowledge right now?"

        # Default supportive persona
        if detected_emotion == "hopeful":
            return "Celebrating your small victories and feeling hopeful is a wonderful reminder of your resilience. What part of today's progress feels most meaningful to you?"
        if detected_emotion in ("anxious", "confused"):
            return "It is completely natural to feel overwhelmed when holding so many expectations at once. What would offering yourself a moment of rest and patience look like today?"
        if detected_emotion == "happy":
            return "Experiencing joy and fulfillment is deeply energizing. What contributed most to that uplifting feeling today?"
        if detected_emotion == "sad":
            return "Giving yourself space to process these tender emotions takes genuine courage. What kind of care or reassurance does your mind need right now?"

        return "Acknowledging these feelings is a meaningful step toward self-awareness. What part of this experience feels most significant to you?"


class MockEmbeddingProvider:
    """Mock embedding provider returning deterministic 384-dimensional unit vectors."""

    def __init__(self, model_name: str = "mock-minilm-l6", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate deterministic unit vectors derived from SHA-256 hashes of the texts."""
        vectors: list[list[float]] = []

        for text in texts:
            # Hash text to generate a seed
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw_floats: list[float] = []

            # Repeat hash bytes to populate vector dimension
            for i in range(self._dimension):
                byte_val = digest[i % len(digest)]
                raw_floats.append((byte_val / 128.0) - 1.0)  # Value between -1.0 and 1.0

            # Normalize to unit length
            norm = math.sqrt(sum(x * x for x in raw_floats)) or 1.0
            unit_vector = [round(x / norm, 6) for x in raw_floats]
            vectors.append(unit_vector)

        return EmbeddingResult(
            vectors=vectors,
            dimension=self._dimension,
            model_name=self._model_name,
        )


class MockEmotionClassifier:
    """Mock emotion classifier using deterministic keyword rules."""

    def __init__(self, model_name: str = "mock-bart-emotion") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def classify(
        self,
        text: str,
        candidate_labels: list[str] | None = None,
    ) -> EmotionResult:
        labels = candidate_labels or DEFAULT_EMOTIONS
        lower = text.lower()

        # Keyword mapping
        top = "calm"
        if any(w in lower for w in ["happy", "joy", "great", "glad", "excited"]):
            top = "happy"
        elif any(w in lower for w in ["sad", "down", "cry", "depressed", "unhappy"]):
            top = "sad"
        elif any(w in lower for w in ["angry", "mad", "furious", "annoyed", "rage"]):
            top = "angry"
        elif any(w in lower for w in ["hope", "optimistic", "future", "looking forward"]):
            top = "hopeful"
        elif any(w in lower for w in ["guilt", "sorry", "fault", "ashamed"]):
            top = "guilty"
        elif any(w in lower for w in ["anxious", "nervous", "worry", "panic", "stress"]):
            top = "anxious"
        elif any(w in lower for w in ["confused", "lost", "uncertain", "puzzled"]):
            top = "confused"

        if top not in labels:
            top = labels[0] if labels else "calm"

        # Distribute scores
        scores: list[EmotionScore] = []
        top_score = 0.75
        other_score = (1.0 - top_score) / max(len(labels) - 1, 1)

        for lbl in labels:
            sc = top_score if lbl == top else other_score
            scores.append(EmotionScore(label=lbl, score=round(sc, 4)))

        # Sort descending by score
        scores.sort(key=lambda s: s.score, reverse=True)

        return EmotionResult(
            top_emotion=top,
            emotions=scores,
            confidence=top_score,
            model_name=self._model_name,
        )


class MockValueClassifier:
    """Mock core value classifier using deterministic keyword rules."""

    def __init__(self, model_name: str = "mock-bart-value") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def classify(
        self,
        text: str,
        candidate_labels: list[str] | None = None,
    ) -> ValueResult:
        labels = candidate_labels or DEFAULT_VALUES
        lower = text.lower()

        top = "growth"
        if any(w in lower for w in ["honest", "truth", "sincere", "frank"]):
            top = "honesty"
        elif any(w in lower for w in ["trust", "rely", "depend", "faithful"]):
            top = "trust"
        elif any(w in lower for w in ["talk", "communicat", "listen", "share", "discuss"]):
            top = "communication"
        elif any(w in lower for w in ["regret", "wish i", "mistake", "should have"]):
            top = "regret"
        elif any(w in lower for w in ["forgive", "let go", "pardon"]):
            top = "forgiveness"
        elif any(w in lower for w in ["brave", "courage", "bold", "fearless"]):
            top = "courage"

        if top not in labels:
            top = labels[0] if labels else "growth"

        scores: list[ValueScore] = []
        top_score = 0.70
        other_score = (1.0 - top_score) / max(len(labels) - 1, 1)

        for lbl in labels:
            sc = top_score if lbl == top else other_score
            scores.append(ValueScore(label=lbl, score=round(sc, 4)))

        scores.sort(key=lambda s: s.score, reverse=True)

        return ValueResult(
            top_value=top,
            values=scores,
            confidence=top_score,
            model_name=self._model_name,
        )


class MockVectorStore:
    """Deterministic, zero-dependency mock vector store for testing."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            msg = f"Vector dimension must be greater than 0, got {dimension}"
            raise ValueError(msg)
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            if len(record.vector) != self._dimension:
                msg = (
                    f"Vector dimension mismatch: "
                    f"expected {self._dimension}, got {len(record.vector)}"
                )
                raise ValueError(msg)
            if any(math.isnan(x) or math.isinf(x) for x in record.vector):
                msg = f"Vector contains non-finite numeric value for record '{record.id}'"
                raise ValueError(msg)
            self._records[record.id] = record

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        if not query_vector:
            msg = "Query vector cannot be empty."
            raise ValueError(msg)

        if len(query_vector) != self._dimension:
            msg = (
                f"Query vector dimension mismatch: "
                f"expected {self._dimension}, got {len(query_vector)}"
            )
            raise ValueError(msg)

        if any(math.isnan(x) or math.isinf(x) for x in query_vector):
            msg = "Query vector contains non-finite numeric value."
            raise ValueError(msg)

        if top_k <= 0 or not self._records:
            return []

        if not any(query_vector):
            if min_score <= 0.0:
                sorted_zero = sorted(self._records.values(), key=lambda r: str(r.id))
                return [
                    VectorSearchResult(id=rec.id, score=0.0, metadata=dict(rec.metadata))
                    for rec in sorted_zero[:top_k]
                ]
            return []

        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0.0:
            return []

        scored: list[tuple[VectorRecord, float]] = []
        for rec in self._records.values():
            rec_norm = math.sqrt(sum(x * x for x in rec.vector))
            if rec_norm == 0.0:
                score = 0.0
            else:
                dot = sum(a * b for a, b in zip(query_vector, rec.vector, strict=False))
                raw_cos = max(-1.0, min(1.0, dot / (q_norm * rec_norm)))
                score = round((1.0 + raw_cos) / 2.0, 6)

            if score >= min_score:
                scored.append((rec, score))

        scored.sort(key=lambda item: (-item[1], str(item[0].id)))
        top = scored[:top_k]

        return [
            VectorSearchResult(
                id=rec.id,
                score=sc,
                metadata=dict(rec.metadata),
            )
            for rec, sc in top
        ]

    def delete(self, ids: list[str]) -> bool:
        if not ids:
            return False
        deleted_any = False
        for item_id in ids:
            if self._records.pop(str(item_id), None) is not None:
                deleted_any = True
        return deleted_any

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)
