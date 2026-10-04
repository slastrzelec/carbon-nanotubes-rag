"""Unit tests for src/generation.rag_query (retriever and LLM are mocked)."""
from unittest.mock import MagicMock, patch

from src.generation import rag_query


def test_rag_query_passes_use_hybrid_to_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [{"rank": 1, "filename": "a.pdf", "text": "t", "score": 0.5}]
    with patch("src.generation.generate_answer", return_value="answer"):
        answer, chunks = rag_query(retriever, "q", top_k=3, selected_files=["a.pdf"], use_hybrid=False)

    retriever.retrieve.assert_called_once_with("q", top_k=3, selected_files=["a.pdf"], use_hybrid=False)
    assert answer == "answer"
    assert chunks[0]["filename"] == "a.pdf"


def test_rag_query_defaults_to_hybrid():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    with patch("src.generation.generate_answer", return_value="answer"):
        rag_query(retriever, "q")
    assert retriever.retrieve.call_args.kwargs["use_hybrid"] is True
