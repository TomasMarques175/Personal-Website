# Project: AI-powered personal portfolio assistant

- Status: this repository
- Problem: visitors to a personal site cannot easily explore a CV as a conversation
- Approach: Retrieval-Augmented Generation (RAG). Markdown documents are chunked, embedded with Hugging Face sentence-transformers (PyTorch), stored in FAISS, retrieved with L2 nearest neighbours, and passed to an LLM (Groq by default, Ollama optional)
- Backend: FastAPI (`POST /chat`)
- Frontend: static HTML/CSS/JS widget served by the same app
- Ops: Docker Compose, local vector index under `storage/`
- What this demonstrates: ingestion, embeddings, vector search, prompt design, API design, containerisation

Links: add GitHub URL when the repo is public.
