"""
FastAPI-Router: Endpunkte für Ingestion und Abfrage.

/ingest              – Dev/Admin-Werkzeug zum Aufbau des Vector Stores (nicht in der UI)
/query               – User-facing RAG-Abfrage mit HyDE + Structured Output
/compare             – Parteienvergleich mit HyDE + Stance Detection
/factsheet/{party}   – Gibt das bei der Ingestion generierte Fact-Sheet zurück
/embeddings/map      – UMAP-reduzierte 2D-Koordinaten aller Chunk-Embeddings
/health              – Liveness-Check
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.ingestion.pipeline import run_pipeline
from backend.models.schemas import (
    CompareRequest,
    CompareResponse,
    EmbeddingPoint,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from backend.rag.chain import run_rag
from backend.rag.compare import docs_to_sources, run_compare
from backend.rag.embeddings_map import get_embedding_map
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
    except Exception:
        logger.exception("Fehler bei der Ingestion")
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Ingestion. Details im Server-Log.")

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
    except Exception:
        logger.exception("Fehler in der RAG-Chain")
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Abfrage. Details im Server-Log.")

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


@router.post("/compare", response_model=CompareResponse, tags=["RAG"])
def compare(request: CompareRequest) -> CompareResponse:
    """Vergleicht Parteipositionen via HyDE-Retrieval + Stance Detection."""
    try:
        answer, party_docs = run_compare(
            question=request.question,
            parties=request.parties,
            top_k_per_party=request.top_k_per_party,
        )
    except Exception:
        logger.exception("Fehler in der Compare-Chain")
        raise HTTPException(status_code=500, detail="Interner Fehler beim Vergleich. Details im Server-Log.")

    if not party_docs:
        raise HTTPException(
            status_code=404,
            detail="Keine relevanten Textstellen gefunden. Bitte zuerst PDFs einlesen.",
        )

    return CompareResponse(
        question=request.question,
        summary=answer.summary,
        comparisons=answer.comparisons,
        sources=docs_to_sources(party_docs),
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


@router.get("/embeddings/map", response_model=list[EmbeddingPoint], tags=["Visualisierung"])
def embeddings_map() -> list[EmbeddingPoint]:
    """UMAP-reduzierte 2D-Koordinaten aller Chunk-Embeddings. Ergebnis wird serverseitig gecacht."""
    try:
        return get_embedding_map()
    except Exception:
        logger.exception("Fehler bei der Embedding-Visualisierung")
        raise HTTPException(status_code=500, detail="Fehler beim Berechnen der Embedding-Karte. Details im Server-Log.")


@router.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok"}
