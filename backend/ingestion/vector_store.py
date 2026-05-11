"""
Vector Store: Wrapper um ChromaDB via LangChain.
Stellt add_chunks() und similarity_search() bereit.
Der Rest der App kennt keine ChromaDB-Details.

Embedding-Provider wird via EMBEDDING_PROVIDER in .env gesteuert:
  EMBEDDING_PROVIDER=openai        → OpenAI text-embedding-*  (Standard)
  EMBEDDING_PROVIDER=huggingface   → lokales HuggingFace-Modell via sentence-transformers

Achtung: Bei Wechsel des Providers oder Modells muss ChromaDB neu ingestiert werden,
da jedes Modell eine andere Vektordimension erzeugt.
"""

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _get_embeddings(settings: Settings) -> Embeddings:
    """Factory: gibt das konfigurierte Embedding-Objekt zurück."""
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        encode_kwargs: dict = {"normalize_embeddings": True}
        query_encode_kwargs: dict = {"normalize_embeddings": True}

        # Die E5-Modellfamilie erwartet feste Instruction-Prefixe an Query und Passage.
        # Ohne diese Prefixe sind die Embeddings semantisch unbrauchbar.
        # Quelle: https://huggingface.co/intfloat/multilingual-e5-large
        if "e5" in settings.embedding_model.lower():
            encode_kwargs["prompt"] = "passage: "
            query_encode_kwargs["prompt"] = "query: "

        logger.info(
            "Embeddings: HuggingFace '%s' (encode_kwargs=%s, query_encode_kwargs=%s)",
            settings.embedding_model,
            encode_kwargs,
            query_encode_kwargs,
        )
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
        )

    logger.info("Embeddings: OpenAI '%s'", settings.embedding_model)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Gibt die ChromaDB-Collection zurück (Singleton, einmalig initialisiert)."""
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_get_embeddings(settings),
        persist_directory=settings.chroma_db_path,
    )


def delete_party(party: str) -> int:
    """
    Löscht alle Chunks einer Partei aus der Collection.

    Args:
        party: Kurzname der Partei (muss exakt dem Metadatum entsprechen).

    Returns:
        Anzahl der gelöschten Chunks (0 wenn keine vorhanden waren).
    """
    store = get_vector_store()
    existing = store.get(where={"party": party})
    ids: list[str] = existing["ids"]
    count = len(ids)
    if count:
        store.delete(ids=ids)
        logger.info("Partei '%s': %d alte Chunks gelöscht.", party, count)
    return count


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
