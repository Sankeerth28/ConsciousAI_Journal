"""Unit tests for VectorStore schemas, LocalVectorStore, and MockVectorStore."""

from __future__ import annotations

import concurrent.futures

import pytest

from app.ai.providers.mock import MockVectorStore
from app.ai.providers.vector_store import LocalVectorStore
from app.ai.schemas import (
    MemoryRetrievalResult,
    MemorySearchResult,
    VectorRecord,
    VectorSearchResult,
)


class TestVectorSchemas:
    """Tests for vector data transfer schemas and validation."""

    def test_vector_record_valid(self):
        record = VectorRecord(id="mem_1", vector=[0.1, 0.2, 0.3], metadata={"user": "u1"})
        assert record.id == "mem_1"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.metadata["user"] == "u1"

    def test_vector_record_rejects_empty_vector(self):
        with pytest.raises(ValueError, match=r"VectorRecord\.vector cannot be empty"):
            VectorRecord(id="mem_2", vector=[])

    def test_vector_search_result_score_bounds(self):
        result = VectorSearchResult(id="mem_1", score=0.85)
        assert 0.0 <= result.score <= 1.0

        with pytest.raises(ValueError):
            VectorSearchResult(id="mem_1", score=1.5)

        with pytest.raises(ValueError):
            VectorSearchResult(id="mem_1", score=-0.1)

    def test_memory_search_result_fields(self):
        res = MemorySearchResult(
            memory_id=42,
            content="Historical note",
            score=0.92,
            memory_type="reflection",
            importance=0.7,
        )
        assert res.memory_id == 42
        assert res.content == "Historical note"
        assert res.score == 0.92

    def test_memory_retrieval_result_defaults(self):
        res = MemoryRetrievalResult(query_text_length=10)
        assert res.is_empty is True
        assert res.matches == []
        assert res.context_string is None


class TestMockVectorStore:
    """Tests for MockVectorStore deterministic behavior."""

    def test_mock_vector_store_lifecycle(self):
        store = MockVectorStore(dimension=2)
        assert store.count() == 0

        # Upsert
        r1 = VectorRecord(id="1", vector=[1.0, 0.0])
        r2 = VectorRecord(id="2", vector=[0.0, 1.0])
        store.upsert([r1, r2])
        assert store.count() == 2

        # Duplicate overwrite
        r1_updated = VectorRecord(id="1", vector=[0.5, 0.5])
        store.upsert([r1_updated])
        assert store.count() == 2

        # Search
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert all(0.0 <= r.score <= 1.0 for r in results)

        # Delete
        store.delete(["1"])
        assert store.count() == 1
        assert store.search([1.0, 0.0], top_k=5)[0].id == "2"

        # Clear
        store.clear()
        assert store.count() == 0


class TestLocalVectorStore:
    """Tests for LocalVectorStore cosine similarity, edge cases, and thread safety."""

    def test_upsert_and_count(self):
        store = LocalVectorStore(dimension=3)
        assert store.count() == 0

        records = [
            VectorRecord(id="1", vector=[1.0, 0.0, 0.0]),
            VectorRecord(id="2", vector=[0.0, 1.0, 0.0]),
        ]
        store.upsert(records)
        assert store.count() == 2

    def test_dimension_validation_on_upsert(self):
        store = LocalVectorStore(dimension=3)
        bad_record = VectorRecord(id="bad", vector=[1.0, 0.0])  # Dim 2 instead of 3
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            store.upsert([bad_record])

    def test_dimension_validation_on_search(self):
        store = LocalVectorStore(dimension=3)
        store.upsert([VectorRecord(id="1", vector=[1.0, 0.0, 0.0])])
        with pytest.raises(ValueError, match="Query vector dimension mismatch"):
            store.search([1.0, 0.0])

    def test_cosine_score_semantics_normalized(self):
        """Verify scores are mapped to [0.0, 1.0].

        Identical vectors: ~1.0
        Orthogonal vectors: ~0.5
        Opposite vectors: ~0.0
        """
        store = LocalVectorStore(dimension=2)
        store.upsert(
            [
                VectorRecord(id="identical", vector=[1.0, 0.0]),
                VectorRecord(id="orthogonal", vector=[0.0, 1.0]),
                VectorRecord(id="opposite", vector=[-1.0, 0.0]),
            ]
        )

        results = store.search(query_vector=[1.0, 0.0], top_k=3, min_score=0.0)
        assert len(results) == 3

        scores = {r.id: r.score for r in results}
        assert pytest.approx(scores["identical"], rel=1e-3) == 1.0
        assert pytest.approx(scores["orthogonal"], rel=1e-3) == 0.5
        assert pytest.approx(scores["opposite"], rel=1e-3) == 0.0

    def test_duplicate_ids_overwrite_cleanly(self):
        store = LocalVectorStore(dimension=2)
        store.upsert([VectorRecord(id="mem1", vector=[1.0, 0.0], metadata={"v": 1})])
        assert store.count() == 1

        # Overwrite with new vector and metadata
        store.upsert([VectorRecord(id="mem1", vector=[0.0, 1.0], metadata={"v": 2})])
        assert store.count() == 1

        res = store.search([0.0, 1.0], top_k=1)
        assert len(res) == 1
        assert res[0].id == "mem1"
        assert res[0].metadata["v"] == 2
        assert pytest.approx(res[0].score, rel=1e-3) == 1.0

    def test_zero_vector_handling_safe(self):
        """Zero vectors have norm 0.0; cosine similarity must handle zero-division safely."""
        store = LocalVectorStore(dimension=2)
        store.upsert([VectorRecord(id="zero", vector=[0.0, 0.0])])

        # Querying with a non-zero vector against a stored zero vector returns score 0.0
        results = store.search([1.0, 0.0], top_k=1, min_score=0.0)
        assert len(results) == 1
        assert results[0].id == "zero"
        assert results[0].score == 0.0

        # Querying with a zero vector returns score 0.0
        zero_query_results = store.search([0.0, 0.0], top_k=1, min_score=0.0)
        assert len(zero_query_results) == 1
        assert zero_query_results[0].score == 0.0

    def test_empty_search_query_raises_value_error(self):
        store = LocalVectorStore(dimension=2)
        with pytest.raises(ValueError, match="Query vector cannot be empty"):
            store.search([])

    def test_delete_and_clear(self):
        store = LocalVectorStore(dimension=2)
        store.upsert(
            [
                VectorRecord(id="1", vector=[1.0, 0.0]),
                VectorRecord(id="2", vector=[0.0, 1.0]),
            ]
        )
        assert store.count() == 2

        # Delete non-existent ID returns False
        assert store.delete(["999"]) is False
        assert store.count() == 2

        # Delete existing ID
        assert store.delete(["1"]) is True
        assert store.count() == 1

        # Clear
        store.clear()
        assert store.count() == 0

    def test_min_score_filtering(self):
        store = LocalVectorStore(dimension=2)
        store.upsert(
            [
                VectorRecord(id="high", vector=[1.0, 0.0]),
                VectorRecord(id="low", vector=[-1.0, 0.0]),
            ]
        )
        # Search with min_score=0.6: only 'high' (~1.0) should match
        results = store.search([1.0, 0.0], min_score=0.6)
        assert len(results) == 1
        assert results[0].id == "high"

    def test_concurrent_access_thread_safe(self):
        """Verify LocalVectorStore behaves safely under multi-threaded read/write."""
        store = LocalVectorStore(dimension=4)

        def worker_write(idx: int):
            store.upsert(
                [
                    VectorRecord(
                        id=f"worker_{idx}",
                        vector=[float(idx), 1.0, 0.5, 0.2],
                    )
                ]
            )

        def worker_read():
            return store.search([1.0, 1.0, 0.5, 0.2], top_k=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            write_futures = [executor.submit(worker_write, i) for i in range(20)]
            read_futures = [executor.submit(worker_read) for _ in range(20)]
            concurrent.futures.wait(write_futures + read_futures)

        assert store.count() == 20

    def test_deterministic_ordering_for_ties(self):
        """When multiple records have identical scores, results are sorted deterministically by ID."""
        store = LocalVectorStore(dimension=2)
        store.upsert(
            [
                VectorRecord(id="item_z", vector=[1.0, 0.0]),
                VectorRecord(id="item_a", vector=[1.0, 0.0]),
                VectorRecord(id="item_m", vector=[1.0, 0.0]),
            ]
        )
        results = store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3
        assert [r.id for r in results] == ["item_a", "item_m", "item_z"]

    def test_non_finite_vector_values_rejected(self):
        """Non-finite floats (NaN, Inf) must be rejected to prevent silent vector pollution."""
        store = LocalVectorStore(dimension=2)

        # NaN in record vector via schema
        with pytest.raises(ValueError, match="contains non-finite numeric value"):
            VectorRecord(id="nan_rec", vector=[float("nan"), 1.0])

        # Inf in query vector
        with pytest.raises(ValueError, match="Query vector contains non-finite"):
            store.search([float("inf"), 1.0])
