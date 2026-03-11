"""
FastAPI-Router: Endpunkte für Ingestion und Abfrage.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.ingestion.pipeline import run_pipeline
from backend.ingestion.vector_store import similarity_search, _COLLECTION_NAME
from backend.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_pdf(request: IngestRequest) -> IngestResponse:
    """Liest ein Parteiprogramm-PDF ein und speichert Chunks in ChromaDB."""
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
        collection=_COLLECTION_NAME,
        message=f"{stored} Chunks erfolgreich in ChromaDB gespeichert.",
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest) -> QueryResponse:
    """Beantwortet eine Frage auf Basis der eingelesenen Parteiprogramme."""
    try:
        docs = similarity_search(
            query=request.question,
            top_k=request.top_k,
            party_filter=request.parties,
        )
    except Exception as e:
        logger.exception("Fehler bei der Vektordatenbankabfrage")
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
            content=d.page_content[:500],
        )
        for d in docs
    ]

    # Einfache Antwort: Gibt die relevantesten Textstellen zurück.
    # In der nächsten Iteration wird hier eine LLM-Kette eingebaut.
    answer = (
        f"Gefundene {len(docs)} relevante Textstellen. "
        "LLM-Zusammenfassung folgt in der nächsten Iteration."
    )

    return QueryResponse(answer=answer, sources=sources)


@router.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok"}
