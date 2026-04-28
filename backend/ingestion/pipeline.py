"""
Ingestion-Pipeline: Orchestriert PDF-Laden → Chunking → Speichern in ChromaDB.
Kann direkt als Skript aufgerufen werden:

    python -m backend.ingestion.pipeline --party SPD --file data/pdfs/spd.pdf
"""

import argparse
import logging
from pathlib import Path

from backend.ingestion.chunker import split_documents
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.vector_store import add_chunks, delete_party
from backend.rag.factsheet import generate_factsheet

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

    # 0. Bestehende Chunks der Partei löschen (verhindert Duplikate bei erneutem Einlesen)
    deleted = delete_party(party)
    if deleted:
        logger.info("  → %d bestehende Chunks ersetzt.", deleted)

    # 1. PDF laden
    docs = load_pdf(file_path, party=party)

    # 2. In Chunks aufteilen
    chunks = split_documents(docs)

    # 3. In ChromaDB speichern
    stored = add_chunks(chunks)

    # 4. Fact-Sheet generieren (best-effort – Fehler blockieren nicht die Ingestion)
    try:
        generate_factsheet(party)
    except Exception:
        logger.warning("Fact-Sheet-Generierung fehlgeschlagen, Ingestion wird fortgesetzt.", exc_info=True)

    logger.info("=== Pipeline abgeschlossen: %d Chunks gespeichert ===", stored)
    return stored


if __name__ == "__main__":
    # Aufruf: python -m backend.ingestion.pipeline --party SPD --file data/pdfs/spd.pdf
    # .env wird von get_settings() automatisch eingelesen.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Ingestion-Pipeline: PDF → ChromaDB")
    parser.add_argument("--party", required=True, help="Kurzname der Partei, z.B. SPD")
    parser.add_argument("--file", required=True, help="Pfad zur PDF-Datei")
    args = parser.parse_args()

    count = run_pipeline(party=args.party, file_path=args.file)
    print(f"\n✓ {count} Chunks erfolgreich in ChromaDB gespeichert.")
