"""
Section-aware Chunker: Teilt Wahlprogramme an inhaltlichen Grenzen.

Strategie:
  1. PyMuPDF get_toc() → wenn vorhanden, Abschnittsgrenzen direkt aus PDF-Outline
     (funktioniert für PDFs mit klickbarem Inhaltsverzeichnis)
  2. Schriftgrößen-Heuristik → Überschriften haben größere Schrift als Fließtext
     (Fallback für PDFs ohne oder mit nicht-klickbarem TOC)

Lange Abschnitte werden sekundär mit RecursiveCharacterTextSplitter gesplittet.

Metadaten pro Chunk: party, source, page, section, chunk_method, chunk_index
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_MAX_SECTION_CHARS = 1500
_SECONDARY_OVERLAP = 50
_HEADING_SIZE_RATIO = 1.15   # Überschrift-Schrift muss >= body_size * ratio sein
_MAX_HEADING_LEN = 150
_MIN_SECTION_CHARS = 50      # Abschnitte kürzer als dies werden übersprungen


def chunk_by_sections(pdf_path: str | Path, party: str) -> list[Document]:
    """
    Lädt ein PDF und teilt es an Abschnittsgrenzen in Chunks auf.

    Args:
        pdf_path: Pfad zur PDF-Datei.
        party:    Kurzname der Partei (für Metadaten).

    Returns:
        Liste von Chunk-Documents mit Metadaten (party, source, page, section).
    """
    path = Path(pdf_path)
    doc = fitz.open(str(path))

    toc = doc.get_toc()
    if toc:
        logger.info("PDF '%s': TOC mit %d Einträgen – verwende TOC-Split.", path.name, len(toc))
        sections = _sections_from_toc(doc, toc)
    else:
        logger.info("PDF '%s': Kein TOC – verwende Schriftgrößen-Heuristik.", path.name)
        sections = _sections_from_fontsize(doc)

    chunks = _build_chunks(sections, party, str(path))
    logger.info(
        "Section-aware Chunking '%s': %d Abschnitte → %d Chunks",
        party, len(sections), len(chunks),
    )
    return chunks


# ── TOC-basierte Extraktion ────────────────────────────────────────────────────

def _sections_from_toc(
    doc: fitz.Document, toc: list
) -> list[tuple[str, str, int]]:
    """Gibt Liste von (section_title, text, start_page_0indexed) zurück."""
    levels = {entry[0] for entry in toc}
    target_level = 1 if 1 in levels else min(levels)

    top_entries = [
        (title, page - 1)
        for level, title, page in toc
        if level == target_level
    ]

    sections = []
    for i, (title, start_page) in enumerate(top_entries):
        end_page = top_entries[i + 1][1] if i + 1 < len(top_entries) else len(doc)
        text = _extract_text(doc, start_page, end_page)
        if len(text.strip()) >= _MIN_SECTION_CHARS:
            sections.append((title, text, start_page))

    return sections


# ── Schriftgrößen-Heuristik ───────────────────────────────────────────────────

def _sections_from_fontsize(
    doc: fitz.Document,
) -> list[tuple[str, str, int]]:
    """Erkennt Überschriften über Schriftgröße; gibt (title, text, page) zurück."""
    body_size = _dominant_font_size(doc)
    threshold = body_size * _HEADING_SIZE_RATIO

    heading_pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            block_text, is_heading = _classify_block(block, threshold)
            if is_heading and _MIN_SECTION_CHARS // 10 < len(block_text) < _MAX_HEADING_LEN:
                heading_pages.append((page_num, block_text))

    if not heading_pages:
        logger.warning("Keine Überschriften erkannt – verwende Volltext als eine Section.")
        full_text = _extract_text(doc, 0, len(doc))
        return [("Volltext", full_text, 0)]

    sections = []
    for i, (page_num, title) in enumerate(heading_pages):
        next_page = heading_pages[i + 1][0] if i + 1 < len(heading_pages) else len(doc)
        text = _extract_text(doc, page_num, next_page)
        if len(text.strip()) >= _MIN_SECTION_CHARS:
            sections.append((title, text, page_num))

    return sections


def _dominant_font_size(doc: fitz.Document) -> float:
    """Gibt die häufigste Schriftgröße zurück (entspricht Fließtext-Größe)."""
    sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def _classify_block(block: dict, threshold: float) -> tuple[str, bool]:
    """Gibt (block_text, is_heading) zurück."""
    text_parts: list[str] = []
    all_large = True
    for line in block["lines"]:
        for span in line["spans"]:
            t = span["text"].strip()
            if t:
                text_parts.append(t)
                if span["size"] < threshold:
                    all_large = False
    return " ".join(text_parts), all_large


# ── Text-Extraktion und sekundäres Splitting ──────────────────────────────────

def _extract_text(doc: fitz.Document, start: int, end: int) -> str:
    """Extrahiert Text von Seite start bis end (exklusiv), 0-indiziert."""
    parts = [
        doc[p].get_text("text")
        for p in range(start, min(end, len(doc)))
        if doc[p].get_text("text").strip()
    ]
    return "\n\n".join(parts)


def _build_chunks(
    sections: list[tuple[str, str, int]],
    party: str,
    source: str,
) -> list[Document]:
    """Sekundärer Split für lange Abschnitte; gibt finale Chunks zurück."""
    secondary = RecursiveCharacterTextSplitter(
        chunk_size=_MAX_SECTION_CHARS,
        chunk_overlap=_SECONDARY_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for title, text, start_page in sections:
        base_meta = {
            "party": party,
            "source": source,
            "page": start_page,
            "section": title,
            "chunk_method": "section_aware",
        }
        if len(text) <= _MAX_SECTION_CHARS:
            chunks.append(Document(page_content=text, metadata=base_meta.copy()))
        else:
            for sub in secondary.create_documents([text], metadatas=[base_meta]):
                chunks.append(sub)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
