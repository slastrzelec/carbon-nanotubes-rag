"""
Unit tests for api.py.

These tests mock rag_query() and the Retriever, so they run fast, free, and
without a real OpenAI API key or a built FAISS index — they test the API layer
itself (validation, status codes, response shape), not RAG quality (that's
covered separately by src/evaluation.py).

Run with: pytest tests/test_api.py -v
"""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Patches Retriever() before importing api.py, so the module-level
    `retriever = Retriever()` call in api.py doesn't try to load a real
    embedding model or FAISS index from disk.
    """
    with patch("src.retrieval.Retriever") as MockRetriever:
        MockRetriever.return_value = MagicMock(chunks_meta=[
            {"filename": "paper_a.pdf", "text": "chunk 1"},
            {"filename": "paper_a.pdf", "text": "chunk 2"},
            {"filename": "paper_b.pdf", "text": "chunk 3"},
        ])
        import api
        from src.ratelimit import QueryLimiter

        # Fresh, generous limiter per test so rate limiting doesn't leak between tests.
        api.limiter = QueryLimiter(per_minute=1000, daily_cap=100000)
        api.search_limiter = QueryLimiter(per_minute=1000, daily_cap=100000)
        yield TestClient(api.app)


class TestHealthEndpoint:
    def test_health_check_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDocumentsEndpoint:
    def test_lists_unique_filenames(self, client):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2  # paper_a.pdf and paper_b.pdf, deduplicated
        assert set(data["documents"]) == {"paper_a.pdf", "paper_b.pdf"}


class TestQueryEndpoint:
    def test_valid_question_returns_answer_and_sources(self, client):
        fake_answer = "The D/G ratio indicates defect density."
        fake_sources = [{"rank": 1, "filename": "paper_a.pdf", "text": "...", "score": 0.9}]

        with patch("api.rag_query", return_value=(fake_answer, fake_sources)):
            response = client.post("/query", json={"question": "What is D/G ratio?", "top_k": 5})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == fake_answer
        assert data["sources"] == fake_sources

    def test_empty_question_is_rejected(self, client):
        response = client.post("/query", json={"question": "", "top_k": 5})
        assert response.status_code == 422

    def test_whitespace_only_question_is_rejected(self, client):
        response = client.post("/query", json={"question": "   ", "top_k": 5})
        assert response.status_code == 422

    def test_top_k_above_limit_is_rejected(self, client):
        response = client.post("/query", json={"question": "valid question", "top_k": 50})
        assert response.status_code == 422

    def test_top_k_below_minimum_is_rejected(self, client):
        response = client.post("/query", json={"question": "valid question", "top_k": 0})
        assert response.status_code == 422

    def test_missing_question_field_is_rejected(self, client):
        response = client.post("/query", json={"top_k": 5})
        assert response.status_code == 422

    def test_default_top_k_is_used_when_omitted(self, client):
        with patch("api.rag_query", return_value=("answer", [])) as mock_rag_query:
            client.post("/query", json={"question": "valid question"})

        # confirm rag_query was called with the configured default, not some arbitrary value
        _, kwargs = mock_rag_query.call_args
        assert kwargs["top_k"] == 5

    def test_rag_pipeline_failure_returns_502(self, client):
        with patch("api.rag_query", side_effect=RuntimeError("OpenAI timeout")):
            response = client.post("/query", json={"question": "valid question", "top_k": 5})

        assert response.status_code == 502
        assert "OpenAI timeout" in response.json()["detail"]


class TestSearchEndpoint:
    FAKE_RESULTS = [{"rank": 1, "filename": "paper_a.pdf", "text": "chunk 1", "score": 0.81}]

    def test_returns_ranked_chunks_without_calling_the_llm(self, client):
        import api

        with patch.object(api.retriever, "retrieve", return_value=self.FAKE_RESULTS) as mock_retrieve, \
                patch("api.rag_query") as mock_rag_query:
            response = client.post("/search", json={"query": "  D band  ", "top_k": 3})

        assert response.status_code == 200
        assert response.json() == {"results": self.FAKE_RESULTS}
        mock_rag_query.assert_not_called()
        args, kwargs = mock_retrieve.call_args
        assert args[0] == "D band"  # stripped before retrieval
        assert kwargs["top_k"] == 3

    @pytest.mark.parametrize("payload", [
        {"query": "", "top_k": 5},
        {"query": "   ", "top_k": 5},
        {"query": "x" * 501, "top_k": 5},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": 11},
        {"top_k": 5},
    ])
    def test_invalid_input_is_rejected_before_retrieval(self, client, payload):
        import api

        with patch.object(api.retriever, "retrieve") as mock_retrieve:
            response = client.post("/search", json=payload)
        assert response.status_code == 422
        mock_retrieve.assert_not_called()

    def test_failure_returns_500_without_internal_details(self, client):
        import api

        with patch.object(api.retriever, "retrieve", side_effect=RuntimeError("index file /secret/path missing")):
            response = client.post("/search", json={"query": "valid", "top_k": 5})
        assert response.status_code == 500
        assert response.json() == {"detail": "Search failed."}

    def test_has_its_own_rate_limit_separate_from_query(self, client):
        import api
        from src.ratelimit import QueryLimiter

        api.search_limiter = QueryLimiter(per_minute=1, daily_cap=100)
        with patch.object(api.retriever, "retrieve", return_value=[]), \
                patch("api.rag_query", return_value=("ok", [])):
            search_codes = [client.post("/search", json={"query": "q"}).status_code for _ in range(2)]
            query_code = client.post("/query", json={"question": "q"}).status_code
        assert search_codes == [200, 429]
        assert query_code == 200  # the paid /query budget is untouched

    def test_query_text_is_not_logged(self, client, caplog):
        import logging
        import api

        with caplog.at_level(logging.INFO, logger="api"), \
                patch.object(api.retriever, "retrieve", return_value=[]):
            client.post("/search", json={"query": "UNIQUE-MARKER-123"})
        assert "UNIQUE-MARKER-123" not in caplog.text
        assert all("UNIQUE-MARKER-123" not in str(r.__dict__) for r in caplog.records)


class TestRequestLoggingMiddleware:
    def test_middleware_logs_request_details(self, client, caplog):
        """The logging middleware should fire for every request, regardless of endpoint,
        and record method, path, and status code."""
        import logging

        with caplog.at_level(logging.INFO, logger="api"):
            response = client.get("/health")

        assert response.status_code == 200
        log_records = [r for r in caplog.records if r.name == "api" and r.message == "Request handled"]
        assert len(log_records) == 1

        record = log_records[0]
        assert record.method == "GET"
        assert record.path == "/health"
        assert record.status_code == 200
        assert record.duration_ms >= 0

    def test_middleware_records_error_status_codes(self, client, caplog):
        """A validation failure (422) should still be picked up by the middleware,
        since it wraps every request regardless of the outcome."""
        import logging

        with caplog.at_level(logging.INFO, logger="api"):
            response = client.post("/query", json={"question": "", "top_k": 5})

        assert response.status_code == 422
        log_records = [r for r in caplog.records if r.name == "api" and r.message == "Request handled"]
        assert len(log_records) == 1
        assert log_records[0].status_code == 422


class TestQueryLimits:
    def test_overlong_question_is_rejected(self, client):
        response = client.post("/query", json={"question": "x" * 501, "top_k": 5})
        assert response.status_code == 422

    def test_question_at_max_length_is_accepted(self, client):
        with patch("api.rag_query", return_value=("ok", [])):
            response = client.post("/query", json={"question": "x" * 500, "top_k": 5})
        assert response.status_code == 200

    def test_per_minute_limit_returns_429(self, client):
        import api
        from src.ratelimit import QueryLimiter

        api.limiter = QueryLimiter(per_minute=2, daily_cap=100)
        with patch("api.rag_query", return_value=("ok", [])):
            codes = [client.post("/query", json={"question": "q", "top_k": 1}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]

    def test_daily_cap_returns_429(self, client):
        import api
        from src.ratelimit import QueryLimiter

        api.limiter = QueryLimiter(per_minute=100, daily_cap=1)
        with patch("api.rag_query", return_value=("ok", [])):
            codes = [client.post("/query", json={"question": "q", "top_k": 1}).status_code for _ in range(2)]
        assert codes == [200, 429]
