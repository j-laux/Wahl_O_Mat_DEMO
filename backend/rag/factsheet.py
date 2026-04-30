"""
Fact-Sheet-Generierung nach der Ingestion.

Extrahiert aus einer gleichmäßigen Stichprobe der gespeicherten Chunks:
- Zentrale Themen des Parteiprogramms
- Konkrete Kernversprechen / Maßnahmen
- Politische Positionierung

Das Ergebnis wird als JSON in data/factsheets/ gespeichert und kann über
GET /api/v1/factsheet/{party} abgerufen werden.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.ingestion.vector_store import get_vector_store
from backend.rag.chain import get_llm

logger = logging.getLogger(__name__)

_FACTSHEET_DIR = Path("data/factsheets")

_FACTSHEET_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Du bist Experte für deutsche Parteipolitik. "
        "Analysiere die folgenden Textausschnitte aus dem Parteiprogramm der Partei '{party}' "
        "zur Bundestagswahl 2025 und extrahiere die wichtigsten Informationen strukturiert.\n\n"
        "Textausschnitte:\n{context}",
    ),
    ("human", "Erstelle ein Fact-Sheet für die Partei {party}."),
])

# ── Schemas ───────────────────────────────────────────────────────────────────


class FactSheetContent(BaseModel):
    """LLM-Ausgabe-Schema (ohne Metadaten – werden programmatisch gesetzt)."""

    top_topics: list[str] = Field(
        description="3 bis 5 zentrale Themen des Parteiprogramms als kurze Stichworte."
    )
    key_promises: list[str] = Field(
        description="3 bis 5 konkrete Maßnahmen oder Versprechen aus dem Programm."
    )
    political_position: str = Field(
        description="Politische Positionierung der Partei in 2-3 Sätzen."
    )


class FactSheet(BaseModel):
    """Vollständiges Fact-Sheet inkl. Metadaten für Persistenz und API."""

    party: str
    top_topics: list[str]
    key_promises: list[str]
    political_position: str
    generated_at: str


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────


def _party_to_filename(party: str) -> str:
    return party.lower().replace("/", "_").replace(" ", "_")


def _sample_chunks(party: str, max_chunks: int = 20) -> list[str]:
    """
    Holt eine gleichmäßige Stichprobe von Chunks aus ChromaDB.
    Verteilt die Auswahl über das gesamte Dokument statt nur den Anfang zu nehmen.
    """
    result = get_vector_store().get(where={"party": party}, include=["documents"])
    all_chunks: list[str] = result.get("documents") or []

    if not all_chunks:
        raise ValueError(f"Keine Chunks für Partei '{party}' in ChromaDB gefunden.")

    step = max(1, len(all_chunks) // max_chunks)
    return all_chunks[::step][:max_chunks]


# ── Öffentliche API ───────────────────────────────────────────────────────────


def generate_factsheet(party: str) -> FactSheet:
    """
    Generiert ein Fact-Sheet für eine Partei aus den gespeicherten Chunks.

    Ablauf:
      1. Gleichmäßige Stichprobe von Chunks aus ChromaDB (max. 20)
      2. LLM extrahiert strukturiert Themen, Versprechen, Positionierung
      3. Ergebnis wird als JSON in data/factsheets/ gespeichert

    Args:
        party: Kurzname der Partei (muss in ChromaDB vorhanden sein).

    Returns:
        Das generierte FactSheet.
    """
    sample = _sample_chunks(party)
    context = "\n\n---\n\n".join(sample)

    llm = get_llm()
    structured_llm = llm.with_structured_output(FactSheetContent)
    chain = _FACTSHEET_PROMPT | structured_llm
    content: FactSheetContent = chain.invoke({"context": context, "party": party})

    factsheet = FactSheet(
        party=party,
        top_topics=content.top_topics,
        key_promises=content.key_promises,
        political_position=content.political_position,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    _FACTSHEET_DIR.mkdir(parents=True, exist_ok=True)
    path = _FACTSHEET_DIR / f"{_party_to_filename(party)}.json"
    path.write_text(factsheet.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Fact-Sheet für '%s' gespeichert: %s", party, path)

    return factsheet


def load_factsheet(party: str) -> FactSheet | None:
    """Liest ein gespeichertes Fact-Sheet von Disk. Gibt None zurück wenn nicht vorhanden."""
    path = _FACTSHEET_DIR / f"{_party_to_filename(party)}.json"
    if not path.exists():
        return None
    return FactSheet.model_validate_json(path.read_text(encoding="utf-8"))
