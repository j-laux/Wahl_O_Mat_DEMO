"""
Text-Chunker: Teilt LangChain-Documents in kleinere, überlappende Chunks auf.
Chunk-Größe und Overlap werden aus den zentralen Settings gelesen.
"""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings

logger = logging.getLogger(__name__)


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Teilt eine Liste von Documents in überlappende Chunks auf.

    Args:
        docs: LangChain-Documents (z.B. aus pdf_loader.load_pdf).

    Returns:
        Chunk-Documents mit erhaltenen Metadaten + chunk_index.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "Chunking: %d Seiten → %d Chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
