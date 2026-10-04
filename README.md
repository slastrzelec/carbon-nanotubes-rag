# RAG – Raman Nanotubes QA

[![Tests](https://github.com/slastrzelec/carbon-nanotubes-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/slastrzelec/carbon-nanotubes-rag/actions/workflows/tests.yml)

A Retrieval-Augmented Generation (RAG) system for semantic search and Q&A over scientific
publications on Raman spectroscopy of carbon nanotubes. Started as a simple Streamlit demo,
rebuilt in six phases into a tested, containerized system with a REST API (see below).

**Live demo (Streamlit UI):** [carbon-nanotubes-rag.streamlit.app](https://carbon-nanotubes-rag.streamlit.app/)  
**Live API (Swagger docs):** [rag-raman-api.onrender.com/docs](https://rag-raman-api.onrender.com/docs)

## Overview

- Semantic + keyword hybrid search over 25 scientific PDFs on Raman spectroscopy of carbon nanotubes
- LLM-generated answers grounded in retrieved context, with source citations
- Evaluated for factual faithfulness using [RAGAs](https://github.com/explodinggradients/ragas)

## Tech Stack

- **Language / core:** Python
- **Retrieval:** FAISS (`IndexFlatIP`, normalized embeddings → cosine similarity), `sentence-transformers` (`all-MiniLM-L6-v2`), BM25 (hybrid search)
- **Generation:** OpenAI API (`gpt-4o-mini`)
- **UI:** Streamlit
- **Evaluation:** RAGAs (faithfulness)
- **Testing:** pytest

## Architecture

```
app.py                  → Streamlit UI (thin layer, imports from src/)
api.py                   → FastAPI backend (thin layer, imports from src/)
src/
  config.py              → paths, model names, chunking/retrieval parameters
  logger.py                → structured JSON logging configuration
  ingestion.py            → PDF → text → dedup → sentence-based chunking → FAISS index
  retrieval.py             → dense (FAISS/cosine) + BM25 hybrid search
  generation.py             → prompt construction + LLM call
  eval_dataset.py            → fixed set of evaluation questions
  evaluation.py               → RAGAs evaluation script
tests/
  test_ingestion.py            → unit tests for chunking, deduplication
  test_api.py                   → unit tests for FastAPI endpoints (mocked, no real API calls)
data/
  raw/                          → source PDFs (not tracked in git)
  processed/                     → FAISS index + chunk metadata
  evaluation/                     → RAGAs results (JSON)
```

Both `app.py` and `api.py` are thin interface layers over the same `src/` modules —
no duplicated retrieval or generation logic between the Streamlit UI and the REST API.

**Docker:**
```
Dockerfile.api          → container image for the FastAPI backend
Dockerfile.streamlit     → container image for the Streamlit UI
docker-compose.yml        → runs both containers together
.dockerignore               → excludes raw PDFs, secrets, dev artifacts from images
```

## Phase 0 — Retrieval Quality & Engineering Foundations

The original prototype used `IndexFlatL2` on non-normalized embeddings and naive
character-based chunking. Phase 0 focused on fixing retrieval correctness and adding
the engineering practices (tests, evaluation) that were identified as gaps in
technical interview feedback.

**What changed:**

- **Correct similarity metric** — switched to `IndexFlatIP` with L2-normalized vectors,
  giving mathematically correct cosine similarity instead of raw Euclidean distance on
  unnormalized vectors.
- **Sentence-based chunking** — replaced fixed 500-character cuts with sentence-aware
  chunking (~120 words per chunk, with overlap), avoiding mid-sentence truncation.
- **Source deduplication** — added content-hash-based deduplication of source PDFs.
  This caught real duplicates in the dataset (the same paper saved under two different
  filenames), which had previously caused identical, redundant results to appear
  in the top of every search.
- **Hybrid search** — combined dense (embedding) retrieval with BM25 keyword search,
  to catch matches that pure semantic search can miss (e.g. exact technical terms,
  abbreviations like "RBM", "SWNT").
- **Modular architecture** — split the original single-file Streamlit app into
  `ingestion` / `retrieval` / `generation` modules, independently testable and reusable
  (e.g. by a future FastAPI layer).
- **Unit tests** — 9 tests covering chunking and deduplication logic (`pytest tests/`).

## Phase 1 — REST API (FastAPI)

Added a FastAPI backend alongside the existing Streamlit app, so the RAG pipeline
can be consumed by any client (curl, another service, a future integration) — not
just through the Streamlit UI. Both interfaces share the exact same `src/` modules.

**What was added:**

- **Three endpoints:**
  - `GET /health` — liveness check
  - `POST /query` — runs the RAG pipeline (retrieval + generation), returns the
    answer and its sources
  - `GET /documents` — lists all indexed source PDFs
- **Request validation** (Pydantic) — questions must be non-empty and non-whitespace;
  `top_k` is bounded to a sane range (1–20). Invalid requests are rejected with a
  clear `422` response before any retrieval or LLM call happens.
- **Error handling** — failures in the retrieval/generation pipeline (e.g. an OpenAI
  API timeout or rate limit) are caught and returned as a `502` with a readable
  message, instead of a raw stack trace.
- **Automatic interactive docs** — available at `/docs` (Swagger UI), generated
  directly from the Pydantic models and endpoint definitions.
- **Unit tests** (`tests/test_api.py`) — 10 tests covering all three endpoints,
  validation edge cases (empty/whitespace questions, out-of-range `top_k`, missing
  fields), and pipeline failure handling. These mock `rag_query()` and the
  `Retriever`, so they run in seconds without a real OpenAI key or a built index —
  they test the API layer itself, not RAG answer quality (that's covered separately
  by the RAGAs evaluation in Phase 0).

Run the API locally:
```bash
uvicorn api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive documentation.

## Phase 2 — Structured Logging

Replaced plain-text logging with structured JSON logs, making the system's
behavior easy to parse, filter, and eventually feed into monitoring tools (e.g.
Grafana).

**What was added:**

- **`src/logger.py`** — a shared JSON formatter and `get_logger()` helper used
  consistently across the ingestion pipeline and the API, instead of ad-hoc
  `print()` calls or Python's default text logging.
- **Request logging middleware** (`api.py`) — automatically logs every incoming
  HTTP request (method, path, status code, duration), regardless of endpoint, with
  no extra code needed when new endpoints are added later.
- **Pipeline-level logging** — the `/query` endpoint logs the question, `top_k`,
  number of retrieved sources, and generation time on success; on failure, it logs
  the error before returning the `502` response. This makes it possible to see,
  for example, that a slow request is spending most of its time in the OpenAI call
  rather than in retrieval — useful both for debugging and for understanding
  where future optimization effort would pay off.
- **Unit tests** for the middleware, verifying it captures request details on both
  successful and failed (422) requests.

Example log line:
```json
{"timestamp": "2026-08-19T14:49:05.01Z", "level": "INFO", "logger": "api", "message": "Query processed", "question": "What is the D/G ratio?", "top_k": 5, "num_sources": 5, "duration_ms": 6982.88}
```

## Evaluation (RAGAs)

The system was evaluated on a fixed set of 14 domain questions (see
`src/eval_dataset.py`), covering core Raman spectroscopy concepts (D band, G band,
D/G ratio, RBM, chirality, defect characterization), measurement methodology, and one
deliberately out-of-scope question (as a hallucination sanity check).

**Metric: faithfulness** — does the generated answer stick to facts present in the
retrieved context, without introducing unsupported claims?

| Metric | Result |
|---|---|
| Mean faithfulness | **~0.80–0.82** (14 questions) |

**Observations:**

- The out-of-scope question ("What is the melting point of carbon nanotubes?")
  scored **1.0** — the system correctly stated the context didn't contain this
  information instead of fabricating an answer.
- Lower-scoring answers were manually inspected by reviewing the full retrieved
  context and generated answer for each question (saved alongside the faithfulness
  score in `data/evaluation/ragas_results.json`). In most low-scoring cases, retrieval
  was accurate — the retrieved chunks did contain the relevant facts — but the
  generated answer additionally drew on general domain knowledge not explicitly
  present in the retrieved passages (e.g. correct textbook facts about the G-band's
  relation to the E2g phonon mode). This is a case of the model enriching answers
  with accurate general knowledge, not hallucination, but it is worth noting as a
  known characteristic of the current prompt design.
- `context_precision` was intentionally left out of this evaluation round, since it
  requires reference (ground-truth) answers that this question set does not include.

Run the evaluation yourself:
```bash
python -m src.evaluation
```
Note: this makes real OpenAI API calls (both for answer generation and RAGAs'
LLM-as-judge scoring) and is not part of the automated test suite.

## Phase 3 — Docker

Containerized both the API and the Streamlit UI, so the system can run
identically on any machine with Docker installed — no manual Python/conda
environment setup, no dependency version conflicts.

**What was added:**

- **Two separate Dockerfiles** (`Dockerfile.api`, `Dockerfile.streamlit`) — one
  container per service, each with its own entrypoint and exposed port.
- **CPU-only PyTorch install** — `torch` (a `sentence-transformers` dependency)
  defaults to a CUDA-enabled build from PyPI, which is 800MB–2GB even though
  these containers have no GPU access. Installing the CPU-only build from
  PyTorch's own package index instead keeps image size and build time down
  significantly.
- **Layered builds** — dependencies are installed in their own Docker layer,
  before application code is copied in. Code changes don't trigger a full
  reinstall of torch and the rest of the dependencies on rebuild.
- **`docker-compose.yml`** — orchestrates both containers together with a
  single command, sharing the same `.env` file for the OpenAI API key.
- **`.dockerignore`** — keeps raw source PDFs, secrets, and dev artifacts
  (`.git`, `__pycache__`, notebooks) out of the built images.

Run both services with:
```bash
docker-compose up --build
```
Then open `http://127.0.0.1:8000/health` (API) and `http://127.0.0.1:8501`
(Streamlit UI).

## Phase 4 — CI/CD (GitHub Actions)

Added a GitHub Actions workflow that runs the full test suite automatically on
every push and pull request, so broken code is caught immediately rather than
discovered later (or not at all).

**What was added:**

- **`.github/workflows/tests.yml`** — installs dependencies (CPU-only torch,
  same approach as the Docker images) and runs `pytest tests/ -v` on every
  push to any branch and every PR to `main`.
- **Dependency caching** — pip packages are cached between runs, keeping
  workflow duration reasonable despite the heavy `torch`/`sentence-transformers`
  dependencies.
- **Status badge** — the badge at the top of this README reflects the current
  test status directly from GitHub Actions.

## Phase 5 — Deployment

Deployed the FastAPI backend publicly on [Render](https://render.com), using
the same `Dockerfile.api` built in Phase 3 — no separate deployment-specific
build process was needed.

**Live API:** [rag-raman-api.onrender.com](https://rag-raman-api.onrender.com)
([interactive docs](https://rag-raman-api.onrender.com/docs))

**Notes:**

- Running on Render's free tier (512 MB RAM, 0.1 CPU). The CPU-only PyTorch
  install from Phase 3 was essential here — a CUDA build would not fit in the
  available memory.
- Free tier instances spin down after 15 minutes of inactivity; the first
  request after idle time takes 30-60 seconds (cold start) while the
  container restarts and the embedding model reloads.
- The `OPENAI_API_KEY` is configured as an environment variable in Render's
  dashboard, not baked into the image.

## Running Locally

```bash
# install dependencies
pip install -r requirements.txt        # or requirements-dev.txt for tests/evaluation

# build the index from PDFs in data/raw/
python -m src.ingestion

# run the Streamlit app
streamlit run app.py

# run the REST API
uvicorn api:app --reload

# run tests
pytest tests/ -v
```

## Public demo protections

Every question triggers a paid OpenAI call, so the public deployments are
protected against abuse and runaway cost:

- Questions are limited to 500 characters (API validation and UI input).
- The API allows 10 requests per minute per client and 500 queries per day in
  total (HTTP 429 beyond that); the UI has a daily cap and a per-session cap.
- The pre-filled example question is answered once and cached, so page loads
  do not each trigger an OpenAI call.
- Limits are enforced in process memory (`src/ratelimit.py`): they reset on
  restart and are not shared between instances, which is enough for a single
  free-tier instance. The per-client key comes from `X-Forwarded-For` and can
  be spoofed, so the global daily cap is the actual cost ceiling.

## Limitations

- The evaluation covers 14 hand-written questions and one metric
  (faithfulness) scored by an LLM judge, so the ~0.80–0.82 figure is an
  indication, not a precise measurement. Retrieval quality (context precision)
  was not measured.
- Answers are grounded in a small corpus of about 25 papers; questions outside
  it get a refusal or a general-knowledge answer, not a literature search.
- The embedding model (`all-MiniLM-L6-v2`) is a small general-purpose model,
  not tuned for spectroscopy terminology.
- The deployment runs on free tiers: the first request after inactivity takes
  30–60 seconds while the services wake up.

## Roadmap

All planned phases are complete:

- [x] Phase 0 — retrieval correctness, hybrid search, testing, evaluation
- [x] Phase 1 — FastAPI backend, request/response validation, error handling, API tests
- [x] Phase 2 — structured JSON logging (request middleware, pipeline logs)
- [x] Phase 3 — Docker (API + Streamlit containers, docker-compose)
- [x] Phase 4 — CI/CD (GitHub Actions, automated tests on every push)
- [x] Phase 5 — deployment (live API on Render)

## Possible Future Work

Ideas for further extending the project, not currently planned:

- **Architecture documentation** — a data-flow diagram and a short write-up of
  key design decisions (why hybrid search over pure dense retrieval, why
  `IndexFlatIP` over alternatives like HNSW or a managed vector DB, why
  FastAPI runs alongside Streamlit rather than replacing it).
- **Expanded evaluation** — adding RAGAs' `context_precision` metric, which
  requires reference (ground-truth) answers for each evaluation question.
  This was intentionally left out of the Phase 0 evaluation (see above) in
  favor of shipping `faithfulness` results quickly; a full reference answer
  set would give a more complete picture of retrieval quality specifically,
  separate from generation quality.

## License

Code is released under the MIT License (see `LICENSE`).
