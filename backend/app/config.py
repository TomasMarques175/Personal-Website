from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STORAGE_DIR = PROJECT_ROOT / "storage"


class Settings:
    APP_NAME = "Tomás Portfolio RAG Chat"
    APP_VERSION = "0.2.0"

    DATA_DIR = DATA_DIR
    FRONTEND_DIR = FRONTEND_DIR
    STORAGE_DIR = STORAGE_DIR

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", str(STORAGE_DIR / "faiss_index"))
    CHUNKS_INVENTORY_PATH = STORAGE_DIR / "chunks_inventory.csv"

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K = int(os.getenv("TOP_K", "4"))

    # groq | huggingface | ollama | extractive
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000",
        ).split(",")
        if origin.strip()
    ]


settings = Settings()

# Persist Hugging Face downloads next to the FAISS index so Docker volumes reuse them.
os.environ.setdefault("HF_HOME", str(STORAGE_DIR / "hf_cache"))
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
