"""Vector store implementations for semantic similarity search."""

from __future__ import annotations

import logging
import threading

from app.ai.schemas import VectorRecord, VectorSearchResult

logger = logging.getLogger(__name__)


def _compute_cosine_similarity(
    vec1: list[float],
    vec2: list[float],
) -> float:
    """Compute normalized cosine similarity in range [0.0, 1.0].

    Semantics:
        Cosine similarity raw value ranges from -1.0 to +1.0.
        Normalized score = (1.0 + raw_cosine) / 2.0
        Returns 0.0 if either vector has zero magnitude.
    """
    import math

    if len(vec1) != len(vec2) or not vec1 or not vec2:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    raw_cosine = dot_product / (norm1 * norm2)
    # Clamp for numerical floating-point safety
    raw_cosine = max(-1.0, min(1.0, raw_cosine))
    normalized_score = (1.0 + raw_cosine) / 2.0
    return round(float(normalized_score), 6)


class LocalVectorStore:
    """In-memory vector store with NumPy/Python cosine similarity search.

    Features:
    - Thread-safe using threading.Lock()
    - Dimension enforcement: rejects vectors that do not match declared dimension
    - Replaces existing records on duplicate ID
    - Zero mandatory external dependencies (lazy numpy import with pure math fallback)
    - Normalized cosine similarity scores in [0.0, 1.0]
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            msg = f"Vector dimension must be greater than 0, got {dimension}"
            raise ValueError(msg)
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        """Declared vector dimension for this store."""
        return self._dimension

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace vector records in the index.

        Raises:
            ValueError: If any vector's length does not match declared dimension or contains non-finite values.
        """
        if not records:
            return

        import math

        with self._lock:
            for record in records:
                if len(record.vector) != self._dimension:
                    msg = (
                        f"Vector dimension mismatch for record '{record.id}': "
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
        """Search the index for nearest vectors using normalized cosine similarity.

        Args:
            query_vector: Query vector matching declared dimension.
            top_k: Maximum number of results to return (>= 1).
            min_score: Minimum normalized similarity score threshold in [0.0, 1.0].

        Returns:
            List of VectorSearchResult ordered by score descending (ties broken by ID).

        Raises:
            ValueError: If query_vector dimension does not match declared dimension or is invalid.
        """
        if not query_vector:
            msg = "Query vector cannot be empty."
            raise ValueError(msg)

        if len(query_vector) != self._dimension:
            msg = (
                f"Query vector dimension mismatch: "
                f"expected {self._dimension}, got {len(query_vector)}"
            )
            raise ValueError(msg)

        import math

        if any(math.isnan(x) or math.isinf(x) for x in query_vector):
            msg = "Query vector contains non-finite numeric value."
            raise ValueError(msg)

        if top_k <= 0:
            return []

        with self._lock:
            if not self._records:
                return []
            records_snapshot = list(self._records.values())

        # If zero-magnitude query, return candidates with score 0.0 if min_score <= 0.0
        if not any(query_vector):
            if min_score <= 0.0:
                sorted_zero = sorted(records_snapshot, key=lambda r: str(r.id))
                return [
                    VectorSearchResult(id=rec.id, score=0.0, metadata=dict(rec.metadata))
                    for rec in sorted_zero[:top_k]
                ]
            return []

        # Attempt vectorized numpy computation if numpy is available
        try:
            import numpy as np

            q = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0.0:
                normalized_scores = np.zeros(len(records_snapshot), dtype=np.float32)
            else:
                matrix = np.array([r.vector for r in records_snapshot], dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1)
                # Avoid division by zero
                safe_norms = np.where(norms == 0.0, 1e-9, norms)
                dot = np.dot(matrix, q)
                raw_cosine = dot / (safe_norms * q_norm)
                raw_cosine = np.clip(raw_cosine, -1.0, 1.0)
                normalized_scores = (1.0 + raw_cosine) / 2.0
                # Zero out rows with zero norm
                normalized_scores = np.where(norms == 0.0, 0.0, normalized_scores)

            scored_candidates: list[tuple[VectorRecord, float]] = []
            for record, score_val in zip(
                records_snapshot, normalized_scores.tolist(), strict=False
            ):
                score = round(float(score_val), 6)
                if score >= min_score:
                    scored_candidates.append((record, score))

        except ImportError:
            # Graceful pure-python math fallback
            scored_candidates = []
            for record in records_snapshot:
                score = _compute_cosine_similarity(query_vector, record.vector)
                if score >= min_score:
                    scored_candidates.append((record, score))

        # Sort descending by score; ties broken deterministically by ID ascending
        scored_candidates.sort(key=lambda x: (-x[1], str(x[0].id)))
        top_results = scored_candidates[:top_k]

        return [
            VectorSearchResult(
                id=rec.id,
                score=score,
                metadata=dict(rec.metadata),
            )
            for rec, score in top_results
        ]

    def delete(self, ids: list[str]) -> bool:
        """Remove vector records by their IDs. Returns True if any record was removed."""
        if not ids:
            return False
        deleted_any = False
        with self._lock:
            for item_id in ids:
                if self._records.pop(str(item_id), None) is not None:
                    deleted_any = True
        return deleted_any

    def clear(self) -> None:
        """Clear all records from the index."""
        with self._lock:
            self._records.clear()

    def count(self) -> int:
        """Return the number of vectors stored in the index."""
        with self._lock:
            return len(self._records)
