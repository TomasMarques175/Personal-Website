# Project blurb for a CV / LinkedIn

Use one of these blocks. Keep the stack honest: embeddings run on PyTorch via sentence-transformers; there is no custom trained model.

## Short (3–4 lines)

Personal portfolio with an embedded RAG chatbot (“Ask Tomás”). Markdown CV and project notes are chunked, embedded with Hugging Face sentence-transformers (PyTorch), indexed in FAISS, and retrieved into a FastAPI `/chat` endpoint. Generation uses a hosted LLM (Groq) through LangChain, with an extractive fallback and a documented path to a local Ollama model. Packaged with Docker Compose.

## Bullet version

- Designed a small RAG system: ingest → overlapping chunks → dense embeddings → FAISS L2 retrieval → grounded LLM answers
- Backend in FastAPI; knowledge base as versioned markdown; chunk inventory exported with Pandas
- Swappable generators (Groq API, Hugging Face Inference, Ollama) behind one interface so retrieval stays local
- Containerised CPU deployment; frontend widget served from the same app

## Interview talking points

- Why RAG instead of stuffing the whole CV into the prompt: you can grow documents without changing the API, and you can show sources.
- Why FAISS IndexFlatL2: corpus is tiny; approximate indexes would be premature.
- Why not a local LLM in the default Docker image: embeddings are cheap; generation is not. The code is provider-agnostic.
- Failure mode you handled: FAISS does not store text, so the corpus is pickled next to the index.

## Portuguese one-liner

Site portfólio com chatbot RAG (FastAPI, LangChain, sentence-transformers/PyTorch, FAISS, Docker) para responder a perguntas sobre formação, projetos e skills a partir de documentos indexados.
