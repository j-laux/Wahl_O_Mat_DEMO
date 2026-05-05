"""
UMAP-Reduktion der gespeicherten Chunk-Embeddings auf 2D.

Wird einmalig pro Server-Session berechnet (lru_cache) und über
GET /api/v1/embeddings/map an das Frontend geliefert.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from umap import UMAP

from backend.ingestion.vector_store import get_vector_store
from backend.models.schemas import EmbeddingPoint

logger = logging.getLogger(__name__)

_PARTY_ORDER = ["SPD", "CDU/CSU", "Grüne", "FDP", "AfD", "Linke", "BSW"]


@lru_cache(maxsize=1)
def get_embedding_map() -> list[EmbeddingPoint]:
    """
    Lädt alle Embeddings aus ChromaDB und reduziert sie mit UMAP auf 2D.

    Ergebnis wird gecacht – UMAP läuft nur einmal pro Server-Session.
    """
    store = get_vector_store()
    result = store.get(include=["embeddings", "documents", "metadatas"])

    embeddings: list = result.get("embeddings") or []
    documents: list = result.get("documents") or []
    metadatas: list = result.get("metadatas") or []

    if not embeddings:
        logger.warning("Keine Embeddings in ChromaDB gefunden.")
        return []

    logger.info("UMAP: %d Embeddings werden auf 2D reduziert ...", len(embeddings))

    coords = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    ).fit_transform(np.array(embeddings))

    logger.info("UMAP abgeschlossen.")

    return [
        EmbeddingPoint(
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            party=metadatas[i].get("party", "unbekannt"),
            preview=documents[i][:150].replace("\n", " "),
        )
        for i in range(len(embeddings))
    ]
