"""
Text-Chunker: Teilt LangChain-Documents in kleinere, überlappende Chunks auf.
Die Chunking-Parameter werden aus den Umgebungsvariablen CHUNK_SIZE und
CHUNK_OVERLAP gelesen (Defaults: 1000 / 200 Zeichen).
"""

import logging
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Teilt eine Liste von Documents in Chunks auf.

    Args:
        docs: LangChain-Documents (z.B. aus pdf_loader.load_pdf).

    Returns:
        Liste von Chunk-Documents mit erhaltenen Metadaten.
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # Chunk-Index pro Seite für bessere Nachvollziehbarkeit
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "Chunking: %d Seiten → %d Chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
