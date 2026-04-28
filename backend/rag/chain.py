"""
RAG-Chain mit HyDE (Hypothetical Document Embeddings).

Pipeline (zwei LLM-Calls pro Anfrage):

    Nutzerfrage
        │
        ▼  [1] HyDE-Chain
    Hypothetischer Programmtext
        │
        ▼  Embedding + ChromaDB similarity_search
    Relevante Chunks (party, page, content)
        │
        ▼  [2] Antwort-Chain (with_structured_output)
    RagAnswer { summary, positions: [PartyPosition] }

Warum HyDE?
    Die Embedding-Distanz zwischen einer kurzen Frage und einem langen Fließtext
    ist größer als zwischen zwei inhaltlich ähnlichen Texten. HyDE überbrückt
    diese Lücke, indem der Suchvektor aus einem synthetischen Antwort-Text
    statt aus der Original-Frage berechnet wird.
"""

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.ingestion.vector_store import similarity_search
from backend.models.schemas import PartyPosition

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Du bist Experte für deutsche Parteipolitik. "
        "Schreibe einen kurzen Absatz (2–3 Sätze), wie er in einem deutschen Parteiprogramm "
        "zur Bundestagswahl 2025 stehen könnte, als Antwort auf die folgende Frage. "
        "Antworte nur mit dem hypothetischen Programmtext, ohne Einleitung oder Erklärung.",
    ),
    ("human", "{question}"),
])

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Du bist ein sachlicher Politikanalyst. "
        "Beantworte die Frage ausschließlich auf Basis der folgenden Textausschnitte "
        "aus deutschen Parteiprogrammen zur Bundestagswahl 2025.\n\n"
        "Regeln:\n"
        "- Beschreibe nur Parteien, die im Kontext tatsächlich zu dieser Frage Stellung nehmen\n"
        "- Bleib sachlich und nah am Quelltext\n"
        "- Wenn der Kontext die Frage nicht beantworten kann, sage das in 'summary' explizit\n\n"
        "Kontext:\n{context}",
    ),
    ("human", "{question}"),
])

# ── LLM-Schema für strukturierte Ausgabe ──────────────────────────────────────


class RagAnswer(BaseModel):
    """Strukturierte LLM-Antwort. Wird intern verwendet; die API gibt QueryResponse zurück."""

    summary: str = Field(
        description="Übergreifende Zusammenfassung der Parteipositionen zu dieser Frage in 2-4 Sätzen."
    )
    positions: list[PartyPosition] = Field(
        description=(
            "Eine Einheit pro Partei, die im Kontext zu dieser Frage Stellung nimmt. "
            "Nur Parteien aufnehmen, die tatsächlich im Kontext vorkommen."
        )
    )


# ── LLM-Singleton ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Gibt den ChatOpenAI-Singleton zurück (geteilt mit factsheet.py)."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key.get_secret_value(),
    )


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────


def _format_context(docs: list[Document]) -> str:
    """Formatiert Chunks als zitierbaren Kontext-Block für den RAG-Prompt."""
    parts = [
        f"[{doc.metadata.get('party', 'unbekannt')}, S. {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc in docs
    ]
    return "\n\n---\n\n".join(parts)


# ── Öffentliche API ───────────────────────────────────────────────────────────


def run_rag(
    question: str,
    top_k: int = 5,
    party_filter: list[str] | None = None,
) -> tuple[RagAnswer, list[Document]]:
    """
    Führt die vollständige RAG-Pipeline mit HyDE aus.

    Args:
        question:     Frage des Nutzers in natürlicher Sprache.
        top_k:        Anzahl der zu holenden Chunks aus ChromaDB.
        party_filter: Optionale Partei-Whitelist; None = alle Parteien.

    Returns:
        Tuple (RagAnswer, retrieved Documents).
        Bei leerem Retrieval-Ergebnis: (RagAnswer mit Hinweis, []).
    """
    llm = get_llm()

    # Schritt 1: HyDE – hypothetischen Programmtext generieren
    hyde_chain = _HYDE_PROMPT | llm | StrOutputParser()
    hypothetical_doc = hyde_chain.invoke({"question": question})
    logger.debug("HyDE-Dokument: %s", hypothetical_doc[:120])

    # Schritt 2: Retrieval mit hypothetischem Text statt Original-Frage
    docs = similarity_search(
        query=hypothetical_doc,
        top_k=top_k,
        party_filter=party_filter,
    )
    logger.info("Retrieval: %d Chunks gefunden (HyDE, top_k=%d)", len(docs), top_k)

    if not docs:
        return RagAnswer(
            summary="Keine relevanten Textstellen gefunden. Bitte zuerst Parteiprogramme einlesen.",
            positions=[],
        ), []

    # Schritt 3: Strukturierte Antwort auf Basis der gefundenen Chunks
    structured_llm = llm.with_structured_output(RagAnswer)
    answer_chain = _RAG_PROMPT | structured_llm
    answer: RagAnswer = answer_chain.invoke({
        "question": question,
        "context": _format_context(docs),
    })

    return answer, docs
