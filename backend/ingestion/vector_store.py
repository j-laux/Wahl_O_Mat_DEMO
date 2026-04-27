"""
Vector Store: Wrapper um ChromaDB via LangChain.
Stellt add_chunks() und similarity_search() bereit.
Der Rest der App kennt keine ChromaDB-Details.
"""

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Gibt die ChromaDB-Collection zurück (Singleton, einmalig initialisiert)."""
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_path,
    )


def add_chunks(chunks: list[Document]) -> int:
    """Fügt Chunks in die ChromaDB-Collection ein. Gibt Anzahl gespeicherter Chunks zurück."""
    if not chunks:
        logger.warning("Keine Chunks zum Einfügen übergeben.")
        return 0

    settings = get_settings()
    get_vector_store().add_documents(chunks)
    logger.info(
        "%d Chunks in Collection '%s' gespeichert (Pfad: %s)",
        len(chunks),
        settings.chroma_collection_name,
        settings.chroma_db_path,
    )
    return len(chunks)


def similarity_search(
    query: str,
    top_k: int = 5,
    party_filter: list[str] | None = None,
) -> list[Document]:
    """
    Ähnlichkeitssuche in ChromaDB.

    Args:
        query:        Suchanfrage als natürlichsprachiger Text.
        top_k:        Maximale Anzahl zurückgegebener Dokumente.
        party_filter: Optionale Partei-Whitelist; None = alle Parteien.

    Returns:
        Liste der relevantesten Documents mit Metadaten (party, page, source).
    """
    where_filter = None
    if party_filter:
        where_filter = (
            {"party": party_filter[0]}
            if len(party_filter) == 1
            else {"party": {"$in": party_filter}}
        )

    return get_vector_store().similarity_search(query, k=top_k, filter=where_filter)
