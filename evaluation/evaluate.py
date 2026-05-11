"""
RAGAS-Evaluation der Wahl-O-Mat RAG-Pipeline.

Metriken:
  - faithfulness       : Ist die Antwort im abgerufenen Kontext verankert?
  - answer_relevancy   : Beantwortet die Antwort die gestellte Frage?

Verwendung (vom Projektroot aus):
    pip install -r requirements-eval.txt
    python -m evaluation.evaluate

Ausgabe: evaluation/results.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# pysqlite3-Swap muss vor jedem chromadb-Import erfolgen
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Projektroot ins sys.path, damit `backend` importierbar ist
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.metrics import Faithfulness, ResponseRelevancy  # noqa: E402

from backend.config import get_settings  # noqa: E402
from backend.rag.chain import run_rag  # noqa: E402

# RAGAS initialisiert ChatOpenAI direkt über os.environ, nicht über Pydantic Settings
os.environ.setdefault("OPENAI_API_KEY", get_settings().openai_api_key.get_secret_value())

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.json"


def _load_questions() -> list[dict]:
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def _collect_rows(questions: list[dict]) -> dict[str, list]:
    rows: dict[str, list] = {"question": [], "answer": [], "contexts": []}

    for i, item in enumerate(questions, 1):
        question = item["question"]
        parties = item.get("parties")
        logger.info("[%d/%d] %s", i, len(questions), question[:80])

        try:
            answer, docs = run_rag(question=question, top_k=5, party_filter=parties)
        except Exception as exc:
            logger.warning("Übersprungen (Fehler): %s", exc)
            continue

        if not docs:
            logger.warning("Keine Chunks – Frage %d übersprungen.", i)
            continue

        rows["question"].append(question)
        rows["answer"].append(answer.summary)
        rows["contexts"].append([doc.page_content for doc in docs])

    return rows


def main() -> None:
    questions = _load_questions()
    logger.info("Geladen: %d Fragen aus %s", len(questions), QUESTIONS_FILE)

    rows = _collect_rows(questions)
    n = len(rows["question"])
    logger.info("RAG-Läufe abgeschlossen: %d/%d Fragen auswertbar.", n, len(questions))

    if n == 0:
        logger.error("Keine auswertbaren Fragen – Abbruch.")
        sys.exit(1)

    # Hinweis: HyDE zieht bereits semantisch passende Chunks, was faithfulness
    # tendenziell nach oben verzerrt. Der Score ist ein relativer Vergleichswert,
    # kein absolutes Qualitätsmaß. answer_relevancy ist davon nicht betroffen.
    logger.info("Starte RAGAS-Evaluation (faithfulness + answer_relevancy) ...")
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
        )
        for q, a, c in zip(rows["question"], rows["answer"], rows["contexts"])
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset, metrics=[Faithfulness(), ResponseRelevancy()])
    df = result.to_pandas()

    scores = {
        "faithfulness": round(df["faithfulness"].mean(), 4),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 4),
        "n_evaluated": n,
        "n_total": len(questions),
    }

    logger.info("Ergebnis: %s", scores)
    RESULTS_FILE.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Gespeichert: %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
