"""
FastAPI-Einstiegspunkt.

Starten:
    uvicorn backend.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Settings beim Start validieren – fehlt OPENAI_API_KEY, schlägt der Start sofort fehl
    settings = get_settings()
    logger.info(
        "Wahl-O-Mat Backend gestartet – LLM: %s | Embeddings: %s",
        settings.llm_model,
        settings.embedding_model,
    )
    yield
    logger.info("Wahl-O-Mat Backend gestoppt.")


app = FastAPI(
    title="Wahl-O-Mat DEMO API",
    description="RAG-basierte Analyse deutscher Parteiprogramme zur Bundestagswahl 2025.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit-Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
