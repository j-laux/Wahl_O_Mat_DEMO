"""
FastAPI-Einstiegspunkt.

Starten:
    uvicorn backend.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("Wahl-O-Mat Backend gestartet.")
    yield
    logging.getLogger(__name__).info("Wahl-O-Mat Backend gestoppt.")


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
