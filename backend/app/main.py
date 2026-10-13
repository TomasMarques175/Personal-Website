from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.rag.pipeline import RAGPipeline
from backend.app.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    rag_pipeline.ensure_index()
    yield


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    indexed = rag_pipeline.vector_store.index is not None
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "indexed_chunks": len(rag_pipeline.vector_store.documents) if indexed else 0,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = rag_pipeline.query(request.message)
    return ChatResponse(answer=result["answer"], source_documents=result["source_documents"])


if settings.FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(settings.FRONTEND_DIR), html=True),
        name="frontend",
    )
