import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # MUSI być przed importem src.generation, bo tam klient OpenAI tworzy się przy imporcie

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from src.retrieval import Retriever
from src.generation import rag_query
from src import config
from src.ratelimit import QueryLimiter

# 🔹 Cache Retrievera - ciężkie zasoby (model, indeks) ładowane raz na sesję
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()


# 🔹 Cost guard: every question is a paid OpenAI call. The limiter is shared
# by all sessions of this process (global daily cap), plus a per-session cap.
@st.cache_resource
def get_limiter():
    return QueryLimiter(config.RATE_LIMIT_PER_MINUTE * 6, config.DAILY_QUERY_CAP)


MAX_QUESTIONS_PER_SESSION = 20


@st.cache_data(show_spinner=False)
def cached_default_answer(query, top_k, selected_files, use_hybrid):
    """The pre-filled example question is answered once and cached, so page
    loads (including bots) do not each trigger an OpenAI call."""
    return rag_query(
        retriever, query, top_k=top_k, selected_files=list(selected_files), use_hybrid=use_hybrid
    )

# 🔹 Highlight keywords
def highlight_keywords(text, keywords):
    for kw in keywords:
        text = re.sub(f"({re.escape(kw)})", r"**\1**", text, flags=re.IGNORECASE)
    return text

# 🔹 Extract top keywords from corpus
def extract_top_keywords(chunks, top_n=20):
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(corpus)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    term_scores = list(zip(terms, scores))
    term_scores.sort(key=lambda x: x[1], reverse=True)
    top_terms = [t[0] for t in term_scores[:top_n]]
    return top_terms

# 🔹 Streamlit UI
st.set_page_config(page_title="RAG Raman Nanotubes", page_icon="🧪", layout="wide")

# 🔹 Custom CSS — naukowy/techniczny styl: typografia, spacing, karty
st.markdown("""
<style>
    /* Typografia nagłówków — mono, wyraźny letter-spacing jak w dokumentacji technicznej */
    h1, h2, h3 {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        letter-spacing: -0.02em;
    }
    h1 {
        font-weight: 700 !important;
        font-size: 2.1rem !important;
        color: #F2F2F2 !important;
    }

    /* Caption pod tytułem — subtelny, techniczny */
    [data-testid="stCaptionContainer"] {
        font-family: monospace;
        color: #E8482C !important;
        opacity: 0.85;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.75rem !important;
    }

    /* Więcej oddechu między sekcjami */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Karty z fragmentami — subtelna ramka, mono dla nazw plików */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 6px !important;
        border: 1px solid #2A2F3A !important;
        transition: border-color 0.15s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #E8482C !important;
    }

    /* Sidebar — nieco ciemniejsze tło, wyraźniejsze separatory */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #2A2F3A;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9AA0AC !important;
    }

    /* Metric — mono, akcent czerwony na liczbie */
    [data-testid="stMetricValue"] {
        font-family: monospace !important;
        color: #E8482C !important;
    }

    /* Przyciski — ostrzejsze rogi, mono label */
    .stButton > button {
        font-family: monospace !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    /* Pole tekstowe zapytania — mono */
    .stTextInput input {
        font-family: monospace !important;
    }

    /* Relevance score i nazwy plików w kartach fragmentów — mono */
    [data-testid="stMarkdownContainer"] code {
        background-color: rgba(232, 72, 44, 0.12) !important;
        color: #E8482C !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🧪")
with col2:
    st.markdown("# RAG – Raman Nanotubes QA")

st.markdown("### 🔬 Semantic search engine for carbon nanotube research")
st.caption("v2 — hybrid search (dense + BM25), normalized cosine similarity")
st.divider()

# 🔹 Sidebar
st.sidebar.header("📚 About this project")
st.sidebar.markdown("""
**RAG (Retrieval-Augmented Generation)** on scientific PDFs about Raman spectroscopy of carbon nanotubes.

**Tech Stack:**
- Python, FAISS (IndexFlatIP, cosine similarity), Streamlit, OpenAI
- Hybrid search: dense embeddings + BM25
- Sentence-based chunking with deduplication
""")

with st.sidebar.expander("⚙️ This demo is one of two interfaces"):
    st.markdown("""
This Streamlit app is a UI on top of a full production-style system built around the same RAG pipeline:

- 🔌 **[Live REST API (Swagger docs)](https://rag-raman-api.onrender.com/docs)** — the same retrieval/generation logic, exposed as a proper API (`/query`, `/documents`, `/health`), with request validation and error handling.
- 💻 **[Full source code on GitHub](https://github.com/slastrzelec/carbon-nanotubes-rag)** — includes:
  - Unit tests (31 passing) and RAGAs evaluation (faithfulness scoring)
  - Structured JSON logging
  - Docker + docker-compose (this UI and the API each run as a separate container)
  - CI/CD via GitHub Actions (tests run automatically on every push)

The API deploys on a free-tier instance, so the first request after a period of inactivity may take 30–60 seconds (cold start) while it wakes up.
""")

st.sidebar.divider()

all_files = sorted({chunk["filename"] for chunk in retriever.chunks_meta})
selected_files = st.sidebar.multiselect("📄 Select PDFs for retrieval:", all_files, default=all_files[:5])

use_hybrid = st.sidebar.toggle("🔀 Use hybrid search (dense + BM25)", value=True)

# 🔹 Live keywords based on selected PDFs
filtered_chunks = [c for c in retriever.chunks_meta if c["filename"] in selected_files]
top_keywords = extract_top_keywords(filtered_chunks, top_n=20) if filtered_chunks else []
highlight_keywords_selected = st.sidebar.multiselect("🔑 Highlight keywords:", top_keywords, default=top_keywords[:5])

st.sidebar.divider()
st.sidebar.metric("PDFs Selected", len(selected_files))
st.sidebar.metric("Chunks Available", len(filtered_chunks))

# 🔹 Input
DEFAULT_QUERY = "What is the D/G ratio in Raman spectroscopy and carbon nanotubes?"
query = st.text_input(
    "❓ Ask your question (English recommended — source documents are in English):",
    value=DEFAULT_QUERY,
    placeholder="e.g., What is RBM in carbon nanotubes?",
    max_chars=config.MAX_QUESTION_CHARS,
)
top_k = st.slider("📊 Fragments to retrieve:", 1, 10, 5)

asked = st.button("🔍 Ask question", type="primary")
# Auto-run only the pre-filled example with the default settings (one cached
# call); any other combination requires pressing the button.
run_default = (
    not asked
    and query == DEFAULT_QUERY
    and selected_files == all_files[:5]
    and top_k == 5
    and use_hybrid
)

answer = None
if asked or run_default:
    if not query.strip():
        st.warning("Please enter a question.")
    elif run_default:
        with st.spinner("⏳ Searching and generating answer..."):
            answer, retrieved_chunks = cached_default_answer(
                query, top_k, tuple(selected_files), use_hybrid
            )
    else:
        st.session_state["questions_asked"] = st.session_state.get("questions_asked", 0) + 1
        if st.session_state["questions_asked"] > MAX_QUESTIONS_PER_SESSION:
            st.warning("Question limit for this session reached. Reload the page to continue.")
        elif get_limiter().check("ui"):
            st.warning("The demo's daily query limit has been reached. Please try again tomorrow.")
        else:
            with st.spinner("⏳ Searching and generating answer..."):
                answer, retrieved_chunks = rag_query(
                    retriever, query, top_k=top_k,
                    selected_files=selected_files, use_hybrid=use_hybrid,
                )

if answer is not None:
    st.success("✅ Answer generated!")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 💬 Answer")
        st.text_area("Answer", answer, height=300, label_visibility="collapsed")

    with col2:
        st.markdown("### 📄 Top Retrieved Fragments")
        for chunk in retrieved_chunks:
            with st.container(border=True):
                st.markdown(f"**#{chunk['rank']}** • `{chunk['filename']}`")
                st.caption(f"Relevance score: {chunk['score']:.3f}")
                st.markdown(highlight_keywords(chunk['text'][:500] + "...", highlight_keywords_selected))