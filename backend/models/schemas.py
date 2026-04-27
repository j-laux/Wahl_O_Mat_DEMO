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


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
