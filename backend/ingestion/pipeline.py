"""
Ingestion-Pipeline: Orchestriert PDF-Laden → Chunking → Speichern in ChromaDB.
Kann direkt als Skript aufgerufen werden:

    python -m backend.ingestion.pipeline --party SPD --file data/pdfs/spd.pdf
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Projektroot zum Python-Pfad hinzufügen, damit relative Imports funktionieren
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from backend.ingestion.chunker import split_documents
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.vector_store import add_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(party: str, file_path: str | Path) -> int:
    """
    Führt die vollständige Ingestion-Pipeline aus.

    Args:
        party:     Kurzname der Partei (z.B. 'SPD').
        file_path: Pfad zur PDF-Datei.

    Returns:
        Anzahl der in ChromaDB gespeicherten Chunks.
    """
    logger.info("=== Starte Ingestion-Pipeline für Partei: %s ===", party)

    # 1. PDF laden
    docs = load_pdf(file_path, party=party)

    # 2. In Chunks aufteilen
    chunks = split_documents(docs)

    # 3. In ChromaDB speichern
    stored = add_chunks(chunks)

    logger.info("=== Pipeline abgeschlossen: %d Chunks gespeichert ===", stored)
    return stored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestion-Pipeline: PDF → ChromaDB"
    )
    parser.add_argument("--party", required=True, help="Kurzname der Partei, z.B. SPD")
    parser.add_argument("--file", required=True, help="Pfad zur PDF-Datei")
    args = parser.parse_args()

    count = run_pipeline(party=args.party, file_path=args.file)
    print(f"\n✓ {count} Chunks erfolgreich in ChromaDB gespeichert.")
