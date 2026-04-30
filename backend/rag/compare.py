"""
Parteienvergleich-Chain mit HyDE und Stance Detection.

Pipeline:

    Frage + Parteien-Liste
        │
        ▼  [1] HyDE – einmal für alle Parteien
    Hypothetischer Programmtext
        │
        ▼  [2] Retrieval – pro Partei separat (gleiche Repräsentation garantiert)
    Chunks je Partei { "SPD": [...], "Grüne": [...], ... }
        │
        ▼  [3] Vergleichs-Chain (with_structured_output)
    CompareAnswer { summary, comparisons: [PartyComparison + stance] }

Stance Detection ist Teil desselben LLM-Calls wie der Vergleich –
die Klassifizierung (progressiv/konservativ/neutral/unklar) wird direkt
aus dem Kontext abgeleitet, ohne separaten Prompt.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.ingestion.vector_store import similarity_search
from backend.models.schemas import PartyComparison, SourceDocument
from backend.rag.chain import HYDE_PROMPT, get_llm

logger = logging.getLogger(__name__)

_COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Du bist ein sachlicher Politikanalyst. "
        "Vergleiche die Positionen der Parteien zur gestellten Frage, "
        "ausschließlich auf Basis der folgenden Textausschnitte aus deutschen "
        "Parteiprogrammen zur Bundestagswahl 2025.\n\n"
        "Regeln:\n"
        "- Nimm nur Parteien auf, für die ausreichend Kontext vorhanden ist\n"
        "- Bleib nah am Quelltext; keine eigenen Einschätzungen\n"
        "- Stance-Klassifizierung muss aus dem Text ableitbar sein\n"
        "- 'unklar' wenn der Kontext für eine Klassifizierung nicht ausreicht\n\n"
        "Kontext:\n{context}",
    ),
    ("human", "{question}"),
])

# ── Internes LLM-Schema ───────────────────────────────────────────────────────


class CompareAnswer(BaseModel):
    """LLM-Ausgabe für den Parteienvergleich (intern; API gibt CompareResponse zurück)."""

    summary: str = Field(
        description="Übergreifende Zusammenfassung des Vergleichs in 2-3 Sätzen."
    )
    comparisons: list[PartyComparison] = Field(
        description="Eine Einheit pro Partei, für die ausreichend Kontext vorhanden ist."
    )


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────


def _format_compare_context(party_docs: dict[str, list[Document]]) -> str:
    """Formatiert Chunks parteiweise als beschriftete Blöcke für den Vergleichs-Prompt."""
    parts = []
    for party, docs in party_docs.items():
        lines = "\n".join(
            f"[S. {doc.metadata.get('page', '?')}] {doc.page_content}"
            for doc in docs
        )
        parts.append(f"=== {party} ===\n{lines}")
    return "\n\n".join(parts)


def docs_to_sources(party_docs: dict[str, list[Document]], max_chars: int = 500) -> list[SourceDocument]:
    """Flacht party_docs zu einer flachen SourceDocument-Liste für die API-Response ab."""
    sources = []
    for docs in party_docs.values():
        for doc in docs:
            sources.append(SourceDocument(
                party=doc.metadata.get("party", "unbekannt"),
                page=doc.metadata.get("page", 0),
                content=doc.page_content[:max_chars],
            ))
    return sources


# ── Öffentliche API ───────────────────────────────────────────────────────────


def run_compare(
    question: str,
    parties: list[str],
    top_k_per_party: int = 3,
) -> tuple[CompareAnswer, dict[str, list[Document]]]:
    """
    Führt den Parteienvergleich mit HyDE und Stance Detection aus.

    Args:
        question:        Vergleichsfrage des Nutzers.
        parties:         Liste der zu vergleichenden Parteien (mind. 2).
        top_k_per_party: Anzahl Chunks pro Partei für das Retrieval.

    Returns:
        Tuple (CompareAnswer, party_docs).
        party_docs: dict { party: [Document, ...] } für die Quellenangaben.
    """
    llm = get_llm()

    # Schritt 1: HyDE – einmal für alle Parteien
    hyde_chain = HYDE_PROMPT | llm | StrOutputParser()
    hypothetical_doc = hyde_chain.invoke({"question": question})
    logger.debug("HyDE-Dokument: %s", hypothetical_doc[:120])

    # Schritt 2: Retrieval pro Partei – gleiche Chunk-Anzahl für jede Partei
    party_docs: dict[str, list[Document]] = {}
    for party in parties:
        docs = similarity_search(
            query=hypothetical_doc,
            top_k=top_k_per_party,
            party_filter=[party],
        )
        if docs:
            party_docs[party] = docs
        else:
            logger.warning("Keine Chunks für Partei '%s' gefunden – wird übersprungen.", party)

    logger.info(
        "Retrieval: %d von %d Parteien mit Chunks gefunden.",
        len(party_docs), len(parties),
    )

    if not party_docs:
        return CompareAnswer(
            summary="Keine relevanten Textstellen gefunden. Bitte zuerst Parteiprogramme einlesen.",
            comparisons=[],
        ), {}

    # Schritt 3: Strukturierter Vergleich + Stance Detection in einem LLM-Call
    structured_llm = llm.with_structured_output(CompareAnswer)
    answer_chain = _COMPARE_PROMPT | structured_llm
    answer: CompareAnswer = answer_chain.invoke({
        "question": question,
        "context": _format_compare_context(party_docs),
    })

    return answer, party_docs
