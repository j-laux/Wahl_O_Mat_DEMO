"""
PDF-Loader: Liest ein Parteiprogramm-PDF ein und gibt eine Liste von
LangChain-Documents zurück, die Seitentext + Metadaten enthalten.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path, party: str) -> list[Document]:
    """
    Lädt ein PDF und reichert jede Seite mit party-Metadaten an.

    Args:
        file_path: Pfad zur PDF-Datei.
        party:     Kurzname der Partei (wird als Metadatum gespeichert).

    Returns:
        Liste von LangChain Documents (eine pro Seite).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF nicht gefunden: {path}")

    logger.info("Lade PDF: %s (Partei: %s)", path, party)
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    # Metadaten normalisieren
    for doc in docs:
        doc.metadata["party"] = party
        doc.metadata["source"] = path.name
        # PyPDFLoader liefert "page" bereits als 0-basiert → auf 1-basiert korrigieren
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    logger.info("  → %d Seiten geladen", len(docs))
    return docs
