# Portfolio + RAG Chat

Personal site with an embedded chatbot. Visitors ask questions about Tomás’s education, projects and skills. Answers come from a **Retrieval-Augmented Generation (RAG)** pipeline over markdown in `data/`, not from an unconstrained LLM.

## Stack

| Layer | Choice | Why |
| --- | --- | --- |
| API | FastAPI | `POST /chat`, `GET /health`, static frontend |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` | Runs on **PyTorch**; small and local |
| Vectors | FAISS `IndexFlatL2` | Exact search, no extra service |
| Inventory | Pandas CSV of chunks | Inspect what was indexed |
| Generation | Groq (LangChain `ChatOpenAI`) | Fast hosted LLM; swap to Ollama later |
| Frontend | HTML / CSS / JS | Served by the same FastAPI process |
| Ops | Docker Compose | Reproducible CPU image |

LangChain is used for embeddings and the chat model. Chunking, FAISS and retrieval are explicit classes so you can explain each step in an interview.

## Layout

```
backend/app/           FastAPI + RAG
backend/scripts/ingest.py
data/site|cv|projects  knowledge base
frontend/              portfolio + widget
storage/               FAISS index, corpus.pkl, chunks_inventory.csv (gitignored)
docker/Dockerfile
```

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
copy .env.example .env   # then add GROQ_API_KEY if you want fluent answers
python -m backend.scripts.ingest
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000

Without `GROQ_API_KEY` the API still works: it returns the retrieved chunks (extractive fallback).

First start downloads the embedding model from Hugging Face (cached under `storage/hf_cache`).

## Docker

```bash
copy .env.example .env
docker compose up --build
```

Volumes: `./storage` (index + HF cache) and `./data` (so you can edit documents without rebuilding). After changing markdown, run ingest inside the container or restart so the app rebuilds a missing index. To force a rebuild, delete `storage/faiss_index` and restart.

## LLM providers

Set `LLM_PROVIDER` in `.env`:

- `groq` — needs `GROQ_API_KEY` (default model `llama-3.1-8b-instant`)
- `huggingface` — needs `HF_TOKEN`
- `ollama` — local OpenAI-compatible API (`OLLAMA_BASE_URL`, default model `llama3.2`)
- `extractive` — no generation

## Knowledge base

Write facts, not slogans. Fill in:

- `data/site/about.md`, `skills.md`, `faq.md`
- `data/cv/education.md`, `experience.md`
- `data/projects/*.md` — one file per project

Then `python -m backend.scripts.ingest` and check `storage/chunks_inventory.csv`.

## CV blurb

See [docs/cv-project-blurb.md](docs/cv-project-blurb.md).
