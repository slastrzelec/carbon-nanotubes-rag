"""
FastAPI backend for the RAG system (Phase 1 + Phase 2 logging).

Reuses the same src/ modules (Retriever, rag_query) that power the Streamlit app —
no duplicated logic between the two interfaces.

Run with:
    uvicorn api:app --reload
"""
import time

from dotenv import load_dotenv

load_dotenv()  # must run before importing src.generation, which creates an OpenAI client lazily but still needs the key available

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.ratelimit import QueryLimiter
from src.retrieval import Retriever
from src.generation import rag_query
from src import config
from src.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="RAG Raman Nanotubes API",
    description="REST API for semantic search and Q&A over scientific papers on Raman spectroscopy of carbon nanotubes.",
    version="0.1.0",
)

# Loaded once at startup, reused across requests — avoids reloading the embedding
# model and FAISS index on every call.
retriever = Retriever()

limiter = QueryLimiter(config.RATE_LIMIT_PER_MINUTE, config.DAILY_QUERY_CAP)


def client_id(request: Request) -> str:
    """Best-effort client identity. Behind a proxy the first X-Forwarded-For
    entry is used; it can be spoofed, which is why the global daily cap in
    QueryLimiter is the actual cost ceiling."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every incoming request: method, path, status code, and duration."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "Request handled",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_QUESTION_CHARS,
        description=f"The question to ask (1-{config.MAX_QUESTION_CHARS} characters).",
    )
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve (1-20).")


@app.get("/health")
def health_check():
    """Basic liveness check — confirms the server is up and responding."""
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest, http_request: Request):
    """Runs the RAG pipeline: retrieves relevant chunks and generates an answer."""
    refused = limiter.check(client_id(http_request))
    if refused == "rate":
        raise HTTPException(status_code=429, detail="Too many requests, please slow down.")
    if refused == "daily":
        raise HTTPException(status_code=429, detail="Daily query limit for this demo has been reached. Try again tomorrow.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty or whitespace-only.")

    start_time = time.perf_counter()
    try:
        answer, retrieved_chunks = rag_query(retriever, question, top_k=request.top_k)
    except Exception as e:
        logger.error(
            "RAG pipeline failed",
            extra={"question": question, "top_k": request.top_k, "error": str(e)},
        )
        # Covers OpenAI API errors (rate limits, timeouts, auth issues) and any
        # unexpected failure in the retrieval/generation pipeline — the caller
        # gets a clear 502 instead of a raw stack trace.
        raise HTTPException(status_code=502, detail=f"Failed to generate an answer: {e}")

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "Query processed",
        extra={
            "question": question,
            "top_k": request.top_k,
            "num_sources": len(retrieved_chunks),
            "duration_ms": duration_ms,
        },
    )

    return {"answer": answer, "sources": retrieved_chunks}


@app.get("/documents")
def list_documents():
    """Lists all unique source PDF filenames currently indexed."""
    filenames = sorted({chunk["filename"] for chunk in retriever.chunks_meta})
    return {"count": len(filenames), "documents": filenames}