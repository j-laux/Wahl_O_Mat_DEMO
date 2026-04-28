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
        ▼  [2] Antwort-Chain
    Antwort mit Quellenangaben [Partei, S. X]

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

from backend.config import get_settings
from backend.ingestion.vector_store import similarity_search

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
        "- Zitiere für jede Aussage die Quelle in eckigen Klammern, z.B. [SPD, S. 12]\n"
        "- Vergleiche die Positionen der Parteien sachlich, wenn mehrere vorhanden sind\n"
        "- Wenn die Textausschnitte die Frage nicht beantworten können, sage das explizit\n\n"
        "Kontext:\n{context}",
    ),
    ("human", "{question}"),
])

# ── LLM-Singleton ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
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
) -> tuple[str, list[Document]]:
    """
    Führt die vollständige RAG-Pipeline mit HyDE aus.

    Args:
        question:     Frage des Nutzers in natürlicher Sprache.
        top_k:        Anzahl der zu holenden Chunks aus ChromaDB.
        party_filter: Optionale Partei-Whitelist; None = alle Parteien.

    Returns:
        Tuple (Antworttext, retrieved Documents).
        Bei leerem Retrieval-Ergebnis: ("", []).
    """
    llm = _get_llm()

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
        return "", []

    # Schritt 3: Antwort auf Basis der gefundenen Chunks generieren
    answer_chain = _RAG_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "question": question,
        "context": _format_context(docs),
    })

    return answer, docs
