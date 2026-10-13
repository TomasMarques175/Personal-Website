from __future__ import annotations

import logging
from typing import List, Protocol

from backend.app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the portfolio assistant for Tomás, a Junior ML/AI Engineer candidate.
Answer questions about his education, projects, skills, and experience.

Rules:
- Use only the provided context. Do not invent jobs, grades, companies, or dates.
- If the context is not enough, say you do not have that information and suggest asking about education, projects, or skills.
- Reply in the same language as the user's question (Portuguese or English).
- Be concise and concrete. Speak in the third person about Tomás, or as a helpful site assistant — not as if you were Tomás impersonating in a casual chat unless the user asks "who are you" about the bot.
- When useful, mention which source file the fact came from (for example data/projects/...).
"""


class LLMClient(Protocol):
    def generate(self, query: str, context_docs: List[dict]) -> str: ...


def _format_context(context_docs: List[dict]) -> str:
    parts = []
    for doc in context_docs:
        source = doc.get("metadata", {}).get("source", "unknown")
        parts.append(f"[source: {source}]\n{doc['text']}")
    return "\n\n".join(parts)


class ExtractiveLLM:
    """Offline fallback: no generation, only the retrieved chunks.

    Useful when there is no API key. The answer is honest but not fluent.
    """

    def generate(self, query: str, context_docs: List[dict]) -> str:
        if not context_docs:
            return (
                "Não encontrei contexto suficiente no portfólio para responder. "
                "Pergunta sobre formação, projetos, skills ou experiência."
            )
        context = _format_context(context_docs)
        return (
            "Ainda não há um LLM de geração configurado (define GROQ_API_KEY ou "
            "LLM_PROVIDER). Com base nos trechos mais próximos da tua pergunta:\n\n"
            f"{context[:2200]}"
        )


class GroqLLM:
    """Groq is OpenAI-compatible: LangChain ChatOpenAI with a different base_url."""

    def __init__(self, model_name: str | None = None):
        from langchain_openai import ChatOpenAI

        self.model_name = model_name or settings.LLM_MODEL or "llama-3.1-8b-instant"
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
        )

    def generate(self, query: str, context_docs: List[dict]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        if not context_docs:
            return ExtractiveLLM().generate(query, context_docs)

        human = (
            f"Context:\n{_format_context(context_docs)}\n\n"
            f"Question: {query}"
        )
        response = self.llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]
        )
        return str(response.content)


class HuggingFaceLLM:
    """Hosted generation via the Hugging Face Inference API."""

    def __init__(self, model_name: str | None = None):
        from huggingface_hub import InferenceClient

        self.model_name = model_name or settings.LLM_MODEL or "mistralai/Mistral-7B-Instruct-v0.3"
        self.client = InferenceClient(model=self.model_name, token=settings.HF_TOKEN)

    def generate(self, query: str, context_docs: List[dict]) -> str:
        if not context_docs:
            return ExtractiveLLM().generate(query, context_docs)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{_format_context(context_docs)}\n\n"
            f"Question: {query}\nAnswer:"
        )
        try:
            return self.client.text_generation(prompt, max_new_tokens=400, temperature=0.2).strip()
        except Exception:
            completion = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{_format_context(context_docs)}\n\nQuestion: {query}",
                    },
                ],
                max_tokens=400,
                temperature=0.2,
            )
            return completion.choices[0].message.content or ""


class OllamaLLM:
    """Local models through Ollama's OpenAI-compatible endpoint."""

    def __init__(self, model_name: str | None = None):
        from langchain_openai import ChatOpenAI

        self.model_name = model_name or settings.LLM_MODEL or "llama3.2"
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key="ollama",
            base_url=f"{settings.OLLAMA_BASE_URL}/v1",
            temperature=0.2,
        )

    def generate(self, query: str, context_docs: List[dict]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        if not context_docs:
            return ExtractiveLLM().generate(query, context_docs)
        human = (
            f"Context:\n{_format_context(context_docs)}\n\n"
            f"Question: {query}"
        )
        response = self.llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]
        )
        return str(response.content)


def build_llm() -> LLMClient:
    provider = settings.LLM_PROVIDER
    try:
        if provider == "groq" and settings.GROQ_API_KEY:
            logger.info("Using Groq LLM")
            return GroqLLM()
        if provider == "huggingface" and settings.HF_TOKEN:
            logger.info("Using Hugging Face Inference LLM")
            return HuggingFaceLLM()
        if provider == "ollama":
            logger.info("Using Ollama LLM at %s", settings.OLLAMA_BASE_URL)
            return OllamaLLM()
        if provider == "extractive":
            return ExtractiveLLM()
        if provider == "groq" and not settings.GROQ_API_KEY:
            logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY is empty; using extractive fallback")
        elif provider == "huggingface" and not settings.HF_TOKEN:
            logger.warning("LLM_PROVIDER=huggingface but HF_TOKEN is empty; using extractive fallback")
        else:
            logger.warning("Unknown or unconfigured LLM_PROVIDER=%s; using extractive fallback", provider)
    except Exception:
        logger.exception("Failed to initialise LLM provider=%s; using extractive fallback", provider)
    return ExtractiveLLM()
