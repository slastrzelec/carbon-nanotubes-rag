"""
Moduł generacji: buduje prompt z pobranego kontekstu i wywołuje LLM.
Wydzielony z app.py, żeby dało się go testować i używać zarówno w Streamlit, jak i w FastAPI (Faza 1).
"""
from openai import OpenAI

from src import config
from src.retrieval import Retriever

_client = None


def get_client() -> OpenAI:
    """
    Leniwa inicjalizacja klienta OpenAI — tworzony dopiero przy pierwszym użyciu,
    nie przy imporcie modułu. Dzięki temu load_dotenv() zdąży wczytać OPENAI_API_KEY
    niezależnie od kolejności importów w kodzie wywołującym (app.py, api.py, testy).
    """
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    return (
        f"You have access to the following scientific publication fragments:\n\n{context}\n\n"
        f"Please provide a detailed answer to the question: {query}"
    )


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    prompt = build_prompt(query, retrieved_chunks)
    response = get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content


def rag_query(
    retriever: Retriever,
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    selected_files: list[str] | None = None,
    use_hybrid: bool = True,
) -> tuple[str, list[dict]]:
    """Pełny przepływ RAG: retrieval + generacja. Główny punkt wejścia dla app.py i api.py."""
    retrieved_chunks = retriever.retrieve(
        query, top_k=top_k, selected_files=selected_files, use_hybrid=use_hybrid
    )
    answer = generate_answer(query, retrieved_chunks)
    return answer, retrieved_chunks