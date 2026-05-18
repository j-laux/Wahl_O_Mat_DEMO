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
    python -m evaluation.evaluate_ground_truth                              # alle 266 Einträge
    python -m evaluation.evaluate_ground_truth --sample 35                  # ~5 pro Partei
    python -m evaluation.evaluate_ground_truth --sample 35 --label k5_base  # gelabelter Run

Ausgabe (pro Run, in evaluation/runs/):
    {label}_raw.csv      — eine Zeile pro evaluiertem Sample (Per-Row-Scores)
    {label}_scores.json  — Aggregate inkl. Bootstrap-95%-CIs je Metrik
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from backend.config import get_settings
from backend.rag.chain import run_rag

os.environ.setdefault("OPENAI_API_KEY", get_settings().openai_api_key.get_secret_value())

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth" / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "runs"

METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
    "answer_correctness": "factual_correctness(mode=f1)",
}


def _bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile-Bootstrap-CI für den Mittelwert.

    Resamplet `values` mit Zurücklegen, berechnet je Resample den Mittelwert,
    gibt das (alpha/2, 1-alpha/2)-Quantil der Bootstrap-Verteilung zurück.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


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
        "party": [],
        "these_nr": [],
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
                top_k=5,
                party_filter=[entry["party"]],
            )
        except Exception as exc:
            logger.warning("Übersprungen (Fehler): %s", exc)
            continue

        if not docs:
            logger.warning("Keine Chunks – Eintrag übersprungen.")
            continue

        rows["party"].append(entry["party"])
        rows["these_nr"].append(entry["these_nr"])
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
    parser.add_argument(
        "--label",
        type=str,
        default="default",
        help="Run-Label, wird Teil der Output-Dateinamen (z.B. 'baseline_k5')",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Anzahl Bootstrap-Resamples für CIs (default: 10000)",
    )
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
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=gt,
        )
        for q, a, c, gt in zip(
            rows["question"], rows["answer"], rows["contexts"], rows["ground_truth"]
        )
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            FactualCorrectness(),
        ],
    )
    df = result.to_pandas()

    # Per-Row-Scores zusammen mit Identifizierungs-Spalten persistieren —
    # Grundlage für Bootstrap, Fehleranalyse und spätere Re-Aggregation.
    df_out = df.copy()
    df_out.insert(0, "party", rows["party"])
    df_out.insert(0, "these_nr", rows["these_nr"])

    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / f"{args.label}_raw.csv"
    scores_path = RESULTS_DIR / f"{args.label}_scores.json"
    df_out.to_csv(raw_path, index=False)
    logger.info("Per-Row-Scores: %s", raw_path)

    scores: dict = {
        "label": args.label,
        "n_evaluated": n,
        "n_total": len(entries),
        "sample": args.sample,
        "metrics": {},
    }
    for name, col in METRIC_COLUMNS.items():
        values = df[col].to_numpy()
        mean = float(np.nanmean(values))
        lo, hi = _bootstrap_ci(values, n_resamples=args.bootstrap_resamples, seed=args.seed)
        scores["metrics"][name] = {
            "mean": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        }

    logger.info("Ergebnis: %s", scores)
    scores_path.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Aggregate: %s", scores_path)


if __name__ == "__main__":
    main()
