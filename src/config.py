"""
Centralna konfiguracja projektu RAG Raman Nanotubes.
Wszystkie ścieżki i parametry w jednym miejscu — Faza 0/1.
"""
import os

# 🔹 Ścieżki
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

INDEX_PATH = os.path.join(PROCESSED_DIR, "faiss_index.index")
CHUNKS_META_PATH = os.path.join(PROCESSED_DIR, "chunks_meta.json")
CHUNKS_RAW_PATH = os.path.join(PROCESSED_DIR, "chunks.json")

# 🔹 Model embeddingów
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# 🔹 Chunking (Faza 0: cięcie po zdaniach, nie po znakach)
CHUNK_SIZE_TOKENS = 120       # docelowa liczba "słów" w chunku (przybliżenie tokenów)
CHUNK_OVERLAP_TOKENS = 20     # nakładka między chunkami (w słowach)

# 🔹 Retrieval
DEFAULT_TOP_K = 5
HYBRID_ALPHA = 0.5  # waga dense vs BM25 w hybrid search (0=czysty BM25, 1=czysty dense)

# 🔹 Ochrona kosztów publicznego wdrożenia (każde pytanie = płatne wywołanie OpenAI)
MAX_QUESTION_CHARS = 500
RATE_LIMIT_PER_MINUTE = 10   # na klienta
DAILY_QUERY_CAP = 500        # łącznie, dla wszystkich klientów

# 🔹 LLM
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 500
LLM_SYSTEM_PROMPT = (
    "You are an expert in Raman spectroscopy and carbon nanotube nanostructures. "
    "Answer strictly based on the provided context. If the context does not contain "
    "the answer, say so explicitly instead of guessing."
)
