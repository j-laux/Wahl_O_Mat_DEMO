from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    party: str = Field(..., description="Kurzname der Partei, z.B. 'SPD', 'CDU', 'Grüne'")
    file_path: str = Field(..., description="Pfad zur PDF-Datei relativ zum Projektroot")


class IngestResponse(BaseModel):
    party: str
    chunks_stored: int
    collection: str
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="Frage des Nutzers")
    parties: list[str] | None = Field(
        default=None,
        description="Filtert Ergebnisse auf diese Parteien. None = alle Parteien.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDocument(BaseModel):
    party: str
    page: int
    content: str


class PartyPosition(BaseModel):
    """Position einer einzelnen Partei zu einer Frage – Teil der strukturierten LLM-Antwort."""
    party: str = Field(description="Kurzname der Partei, z.B. 'SPD', 'Grüne', 'CDU'")
    position: str = Field(description="Position der Partei zu dieser Frage in 2-3 Sätzen, nah am Quelltext")


class QueryResponse(BaseModel):
    summary: str
    positions: list[PartyPosition]
    sources: list[SourceDocument]


# ── Phase 2: Parteienvergleich ────────────────────────────────────────────────

class CompareRequest(BaseModel):
    question: str = Field(..., description="Vergleichsfrage, z.B. 'Was planen die Parteien beim Klimaschutz?'")
    parties: list[str] = Field(..., min_length=2, description="Mindestens 2 Parteien für den Vergleich")
    top_k_per_party: int = Field(default=3, ge=1, le=10, description="Chunks pro Partei")


class PartyComparison(BaseModel):
    """Strukturierter Vergleichseintrag einer Partei inkl. Stance-Klassifizierung."""
    party: str = Field(description="Kurzname der Partei")
    position: str = Field(description="Position der Partei in 2-3 Sätzen, nah am Quelltext")
    key_points: list[str] = Field(description="2-3 konkrete Kernpunkte als kurze Stichworte")
    stance: Literal["progressiv", "konservativ", "neutral", "unklar"] = Field(
        description=(
            "Politische Haltung auf diesem Thema. "
            "progressiv = für Veränderung/Ausweitung; "
            "konservativ = für Beibehaltung/Stabilität; "
            "neutral = ausgewogen oder nicht eindeutig; "
            "unklar = Kontext nicht ausreichend"
        )
    )


class CompareResponse(BaseModel):
    question: str
    summary: str
    comparisons: list[PartyComparison]
    sources: list[SourceDocument]
