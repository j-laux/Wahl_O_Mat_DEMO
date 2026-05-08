"""
RAGAS-Evaluation auf Basis der 38 Wahl-O-Mat Thesen als Ground-Truth.

Metriken:
  - faithfulness       : Ist die Antwort im abgerufenen Kontext verankert?
  - answer_relevancy   : Beantwortet die Antwort die gestellte Frage?
  - context_precision  : Sind die retrievten Chunks für die Frage relevant?
  - context_recall     : Decken die Chunks die Ground-Truth ab?
  - answer_correctness : Stimmt die Antwort mit der offiziellen Parteiposition überein?

Voraussetzung:
    python -m evaluation.ground_truth.build_dataset

Nutzung:
    python -m evaluation.evaluate_ground_truth            # alle 266 Einträge
    python -m evaluation.evaluate_ground_truth --sample 35  # ~5 pro Partei

Ausgabe: evaluation/results_ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from backend.config import get_settings
from backend.rag.chain import run_rag

os.environ.setdefault("OPENAI_API_KEY", get_settings().openai_api_key.get_secret_value())

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth" / "ground_truth.json"
RESULTS_FILE = Path(__file__).parent / "results_ground_truth.json"


def _load_entries(sample: int | None, seed: int) -> list[dict]:
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground-Truth-Datei nicht gefunden: {GROUND_TRUTH_FILE}\n"
            "Zuerst ausführen: python -m evaluation.ground_truth.build_dataset"
        )
    entries: list[dict] = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))

    if sample is not None:
        # Stratified sample: gleichmäßig über alle Parteien
        by_party: dict[str, list[dict]] = {}
        for e in entries:
            by_party.setdefault(e["party"], []).append(e)
        per_party = max(1, sample // len(by_party))
        rng = random.Random(seed)
        sampled: list[dict] = []
        for party_entries in by_party.values():
            sampled.extend(rng.sample(party_entries, min(per_party, len(party_entries))))
        entries = sampled

    return entries


def _collect_rows(entries: list[dict]) -> dict[str, list]:
    rows: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, entry in enumerate(entries, 1):
        logger.info(
            "[%d/%d] %s | %s", i, len(entries), entry["party"], entry["these_titel"]
        )
        try:
            answer, docs = run_rag(
                question=entry["question"],
                top_k=10,
                party_filter=[entry["party"]],
            )
        except Exception as exc:
            logger.warning("Übersprungen (Fehler): %s", exc)
            continue

        if not docs:
            logger.warning("Keine Chunks – Eintrag übersprungen.")
            continue

        rows["question"].append(entry["question"])
        rows["answer"].append(answer.summary)
        rows["contexts"].append([doc.page_content for doc in docs])
        rows["ground_truth"].append(entry["ground_truth"])

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS Ground-Truth Evaluation")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Stratified Sample-Größe (default: alle 266 Einträge)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries = _load_entries(args.sample, args.seed)
    logger.info("Geladen: %d Einträge aus %s", len(entries), GROUND_TRUTH_FILE)

    rows = _collect_rows(entries)
    n = len(rows["question"])
    logger.info("RAG-Läufe abgeschlossen: %d/%d auswertbar.", n, len(entries))

    if n == 0:
        logger.error("Keine auswertbaren Einträge – Abbruch.")
        sys.exit(1)

    logger.info("Starte RAGAS-Evaluation (5 Metriken) ...")
    dataset = Dataset.from_dict(rows)
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "answer_correctness": round(float(result["answer_correctness"]), 4),
        "n_evaluated": n,
        "n_total": len(entries),
        "sample": args.sample,
    }

    logger.info("Ergebnis: %s", scores)
    RESULTS_FILE.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Gespeichert: %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
