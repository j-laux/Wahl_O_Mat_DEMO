"""
FastAPI-Router: Endpunkte für Ingestion und Abfrage.

/ingest           – Dev/Admin-Werkzeug zum Aufbau des Vector Stores (nicht in der UI)
/query            – User-facing RAG-Abfrage mit HyDE + Structured Output
/factsheet/{party}– Gibt das bei der Ingestion generierte Fact-Sheet zurück
/health           – Liveness-Check
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.ingestion.pipeline import run_pipeline
from backend.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from backend.rag.chain import run_rag
from backend.rag.factsheet import FactSheet, load_factsheet

logger = logging.getLogger(__name__)
router = APIRouter()

_SOURCE_CONTENT_LENGTH = 500


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_pdf(request: IngestRequest) -> IngestResponse:
    """Liest ein Parteiprogramm-PDF ein, speichert Chunks und generiert ein Fact-Sheet."""
    try:
        stored = run_pipeline(party=request.party, file_path=request.file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Fehler bei der Ingestion")
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        party=request.party,
        chunks_stored=stored,
        collection=get_settings().chroma_collection_name,
        message=f"{stored} Chunks erfolgreich in ChromaDB gespeichert.",
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest) -> QueryResponse:
    """Beantwortet eine Frage via RAG-Chain mit HyDE. Gibt strukturierte Antwort zurück."""
    try:
        answer, docs = run_rag(
            question=request.question,
            top_k=request.top_k,
            party_filter=request.parties,
        )
    except Exception as e:
        logger.exception("Fehler in der RAG-Chain")
        raise HTTPException(status_code=500, detail=str(e))

    if not docs:
        raise HTTPException(
            status_code=404,
            detail="Keine relevanten Textstellen gefunden. Bitte zuerst PDFs einlesen.",
        )

    sources = [
        SourceDocument(
            party=d.metadata.get("party", "unbekannt"),
            page=d.metadata.get("page", 0),
            content=d.page_content[:_SOURCE_CONTENT_LENGTH],
        )
        for d in docs
    ]

    return QueryResponse(
        summary=answer.summary,
        positions=answer.positions,
        sources=sources,
    )


@router.get("/factsheet/{party}", response_model=FactSheet, tags=["RAG"])
def get_factsheet(party: str) -> FactSheet:
    """Gibt das bei der Ingestion generierte Fact-Sheet einer Partei zurück."""
    factsheet = load_factsheet(party)
    if factsheet is None:
        raise HTTPException(
            status_code=404,
            detail=f"Kein Fact-Sheet für '{party}' gefunden. Bitte zuerst das PDF einlesen.",
        )
    return factsheet


@router.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok"}
