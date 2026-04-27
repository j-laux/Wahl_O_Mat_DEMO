"""
Vector Store: Wrapper um ChromaDB via LangChain.
Stellt Funktionen bereit zum Einfügen von Chunks und zum späteren Retrieval.
"""

import logging
import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "parteiprogramme")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Gibt die ChromaDB-Collection zurück (Singleton, wird einmalig initialisiert)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=_EMBEDDING_MODEL),
        persist_directory=_CHROMA_DB_PATH,
    )


def add_chunks(chunks: list[Document]) -> int:
    """
    Fügt Chunks in die ChromaDB-Collection ein.

    Args:
        chunks: Liste von LangChain-Documents mit Metadaten.

    Returns:
        Anzahl der gespeicherten Chunks.
    """
    if not chunks:
        logger.warning("Keine Chunks zum Einfügen übergeben.")
        return 0

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    logger.info(
        "%d Chunks in Collection '%s' gespeichert (Pfad: %s)",
        len(chunks),
        COLLECTION_NAME,
        _CHROMA_DB_PATH,
    )
    return len(chunks)


def similarity_search(
    query: str,
    top_k: int = 5,
    party_filter: list[str] | None = None,
) -> list[Document]:
    """
    Führt eine Ähnlichkeitssuche in ChromaDB durch.

    Args:
        query:         Suchanfrage.
        top_k:         Maximale Anzahl zurückgegebener Dokumente.
        party_filter:  Optionale Liste von Parteinamen zum Filtern.

    Returns:
        Liste der relevantesten Documents.
    """
    vector_store = get_vector_store()
    where_filter = None
    if party_filter:
        if len(party_filter) == 1:
            where_filter = {"party": party_filter[0]}
        else:
            where_filter = {"party": {"$in": party_filter}}

    results = vector_store.similarity_search(
        query,
        k=top_k,
        filter=where_filter,
    )
    return results
