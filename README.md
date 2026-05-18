# Wahl-O-Mat DEMO

[🇬🇧 English version](README.en.md)

RAG-basierte Analyseplattform für die **Bundestagswahl 2025** – Version 1 eines mehrstufigen
politischen Wissenssystems. Nutzer können Fragen zu politischen Themen stellen, Antworten
mit Quellenangaben erhalten und die Positionen mehrerer Parteien strukturiert vergleichen –
inklusive automatischer Stance-Klassifizierung (progressiv / konservativ / neutral).

> **Vision:** Das System soll drei Wissensebenen zusammenführen: **(1) Wahlversprechen** aus
> Parteiprogrammen (diese Version), **(2) Abstimmungsverhalten** via Abgeordnetenwatch und
> **(3) aktuelle Rhetorik** aus Nachrichtenquellen – um Wählerinnen und Wählern ein vollständiges
> Bild jenseits von Wahlkampf-Kommunikation zu geben.

---
![til](/graphics/demo.gif)
---
## Features

| Feature | Beschreibung |
|---|---|
| **RAG mit HyDE** | Fragen werden vor dem Retrieval in hypothetischen Programmtext umgewandelt für bessere Chunk-Qualität |
| **Structured Output** | LLM gibt typisiertes JSON zurück (Pydantic + OpenAI Function Calling), kein String-Parsing |
| **Parteienvergleich** | `/compare` stellt Parteipositionen gegenüber, pro Partei mit Stance-Klassifizierung |
| **Stance Detection** | LLM klassifiziert jede Partei als progressiv / konservativ / neutral / unklar |
| **Fact-Sheets** | Bei der Ingestion automatisch generiert: Kernthemen, Versprechen, Positionierung |
| **Deduplizierung** | Erneutes Einlesen ersetzt bestehende Chunks, keine Duplikate in ChromaDB |

---

## Architektur

Das Projekt trennt bewusst zwei Workflows:

**Dev-Workflow (einmalig, offline):**
```
PDF-Dateien  →  Ingestion-Pipeline  →  ChromaDB (persistent auf Disk)
                 └─ PyPDFLoader                    + Fact-Sheet (JSON)
                 └─ RecursiveCharacterTextSplitter
                 └─ OpenAI Embeddings
```

**User-Workflow – Einzelabfrage (`/query`):**
```
Nutzerfrage
    │
    ▼  [1] HyDE: LLM generiert hypothetischen Programmtext
Hypothetischer Text
    │
    ▼  Embedding + ChromaDB Similarity Search
Relevante Chunks (Partei, Seite, Inhalt)
    │
    ▼  [2] RAG: LLM beantwortet Original-Frage (with_structured_output)
{ summary, positions: [{ party, position }], sources }
```

**User-Workflow – Parteienvergleich (`/compare`):**
```
Frage + Parteien-Liste
    │
    ▼  [1] HyDE: einmal für alle Parteien
Hypothetischer Text
    │
    ▼  Retrieval pro Partei separat (gleiche Chunk-Anzahl je Partei)
Chunks je Partei { "SPD": [...], "Grüne": [...], ... }
    │
    ▼  [2] Vergleichs-Chain (with_structured_output)
{ summary, comparisons: [{ party, position, key_points, stance }], sources }
```

**Warum HyDE?**
Die Embedding-Distanz zwischen einer kurzen Nutzerfrage und einem langen Fließtext
ist strukturell größer als zwischen zwei inhaltlich ähnlichen Texten. HyDE überbrückt
diesen Raum durch einen synthetischen Antwort-Text als Suchvektor.
Trade-off: +1 LLM-Call pro Anfrage.

**Warum Retrieval pro Partei beim Vergleich?**
Ein globales Top-K würde größere oder sprachlich dominantere Parteiprogramme bevorzugen.
Pro-Partei-Retrieval garantiert gleiche Repräsentation im Kontext des LLM.

---

## Tech-Stack

| Komponente      | Technologie                                |
|-----------------|--------------------------------------------|
| Frontend        | Streamlit                                  |
| Backend         | FastAPI + Uvicorn                          |
| Konfiguration   | Pydantic Settings (`pydantic-settings`)    |
| RAG-Framework   | LangChain (LCEL)                           |
| Vector Store    | ChromaDB (lokal persistent)               |
| Embeddings      | OpenAI `text-embedding-3-small`            |
| LLM             | OpenAI `gpt-4o-mini`                       |
| PDF-Parsing     | LangChain `PyPDFLoader`                    |

---

## Setup

### 1. Repository klonen & Umgebung erstellen

```bash
git clone <repo-url>
cd Wahl_O_Mat_DEMO

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Benötigt Python 3.10+.

### 2. Umgebungsvariablen konfigurieren

```bash
cp .env.example .env
# .env öffnen und OPENAI_API_KEY eintragen
```

Alle anderen Werte haben sinnvolle Defaults und müssen nicht angepasst werden.

### 3. Parteiprogramme einlesen (Dev-Schritt, einmalig)

Die Parteiprogramme werden vorab verarbeitet und als ChromaDB auf Disk gespeichert.
Endnutzer brauchen diesen Schritt nicht auszuführen.

```bash
python -m backend.ingestion.pipeline --party SPD    --file data/pdfs/spd_programm.pdf
python -m backend.ingestion.pipeline --party Grüne  --file data/pdfs/gruene_programm.pdf
python -m backend.ingestion.pipeline --party CDU    --file data/pdfs/cdu_programm.pdf
# ... weitere Parteien
```

Pro Partei wird automatisch ein Fact-Sheet unter `data/factsheets/` gespeichert.
Wird eine Partei erneut eingelesen, werden die alten Chunks automatisch ersetzt.

---

## Verwendung

### Backend starten

```bash
uvicorn backend.main:app --reload
# API:          http://localhost:8000
# Swagger-Docs: http://localhost:8000/docs
```

### Frontend starten

```bash
streamlit run frontend/app.py
# UI läuft auf http://localhost:8501
```

---

## Projektstruktur

```
Wahl_O_Mat_DEMO/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI-Endpunkte
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF → LangChain Documents + Metadaten
│   │   ├── chunker.py         # RecursiveCharacterTextSplitter
│   │   ├── vector_store.py    # ChromaDB-Wrapper (add, delete, search)
│   │   └── pipeline.py        # Orchestrierung: Delete → Load → Chunk → Store → Fact-Sheet
│   ├── models/
│   │   └── schemas.py         # Pydantic Request/Response-Modelle
│   ├── rag/
│   │   ├── chain.py           # LCEL-Chain: HyDE → Retrieval → Structured Answer
│   │   ├── compare.py         # Parteienvergleich mit Stance Detection
│   │   └── factsheet.py       # Fact-Sheet-Generierung bei Ingestion
│   ├── config.py              # Zentrale Konfiguration via Pydantic Settings
│   └── main.py                # FastAPI App-Instanz mit Lifespan-Hook
├── frontend/
│   └── app.py                 # Streamlit-UI (Query-Tab + Vergleichs-Tab)
├── data/
│   ├── pdfs/                  # Parteiprogramme als PDF (nicht im Repo)
│   ├── chroma_db/             # ChromaDB-Daten (nicht im Repo)
│   └── factsheets/            # Generierte Fact-Sheets als JSON
├── evaluation/
│   ├── ground_truth/
│   │   └── build_dataset.py   # BpB Excel → ground_truth.json (Datensatz gitignored)
│   ├── evaluate.py            # RAGAS Eval 1: faithfulness + answer_relevancy
│   ├── evaluate_ground_truth.py  # RAGAS Eval 2: 5 Metriken mit Ground Truth
│   └── questions.json         # Testset für Eval 1
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Evaluation (RAGAS)

Die Pipeline wird auf zwei Ebenen mit [RAGAS](https://github.com/explodinggradients/ragas) evaluiert:
eine allgemeine Qualitätsmessung und eine Ground-Truth-Validierung gegen offizielle Parteipositionierungen.

### Eval 1 – Allgemeine Pipeline-Qualität

Keine Referenzantworten nötig – nur Frage, Antwort und Kontext.

**Testset:** 25 Fragen zu 10 Themen (Klimaschutz, Wirtschaft, Wohnen, Migration, Bildung, Rente,
Digitalisierung, Europa, Sicherheit, Gesundheit) – Mix aus Einzelpartei-Filter und allen Parteien.

| Metrik               | Score | Beschreibung                                              |
|----------------------|-------|-----------------------------------------------------------|
| **faithfulness**     | 0.77  | Ist die Antwort in den abgerufenen Chunks verankert?     |
| **answer_relevancy** | 0.92  | Beantwortet die Antwort die gestellte Frage?             |

Die hohe answer_relevancy zeigt, dass HyDE + Retrieval konsistent thematisch passende Chunks findet.
Der Faithfulness-Wert von 0.77 ist für politischen Fachtext erwartbar – das LLM synthetisiert
gelegentlich über mehrere Chunks hinweg statt strikt zu zitieren.

```bash
pip install -r requirements-eval.txt
python -m evaluation.evaluate
# → evaluation/results.json
```

### Eval 2 – Ground-Truth-Validierung (Wahl-O-Mat Thesen)

Validierung gegen den offiziellen BpB Wahl-O-Mat 2025 Datensatz: 38 politische Thesen mit
den offiziellen Parteipositionierungen (stimme zu / neutral / stimme nicht zu) und Begründungen
als Referenzantworten. Ermöglicht drei zusätzliche Metriken, die Retrieval-Qualität und
inhaltliche Korrektheit direkt messen.

**Setup:** gpt-4o-mini als Generator und als RAGAS-Judge, n=35 (stratifiziertes Sample
5 Thesen × 7 Parteien, fixer Seed). Jede Kondition wurde **3× wiederholt** (identische
Pipeline, neuer RAGAS-Judge-Call), um End-to-End-Stochastik zu quantifizieren. Berichtet
werden Mittelwert ± σ über die 3 Läufe sowie 2·SE(Δ) = `2·√(σ₁²/3 + σ₂²/3)` als Schwelle,
ab der ein Unterschied zwischen zwei Konditionen nicht mehr im Rauschen liegt.

**Embedder-Vergleich** (RecursiveSplit, k=5):

| Metrik              | OpenAI Ø ± σ        | e5-large Ø ± σ      | Δ (OpenAI − e5) | 2·SE(Δ) | Verdikt                  |
|---------------------|---------------------|---------------------|-----------------|---------|--------------------------|
| faithfulness        | 0.945 ± 0.014       | 0.953 ± 0.036       | −0.009          | 0.045   | im Rauschen              |
| answer_relevancy    | 0.718 ± 0.017       | 0.712 ± 0.030       | +0.006          | 0.040   | im Rauschen              |
| context_precision   | **0.754 ± 0.007**   | 0.670 ± 0.027       | **+0.084**      | 0.032   | **signifikant**          |
| context_recall      | **0.581 ± 0.023**   | 0.537 ± 0.020       | **+0.045**      | 0.035   | **signifikant** (knapp)  |
| answer_correctness  | **0.337 ± 0.015**   | 0.281 ± 0.011       | **+0.056**      | 0.021   | **signifikant**          |

OpenAI = `text-embedding-3-small`. e5 = `intfloat/multilingual-e5-large` mit den vom Modell
geforderten Instruction-Prefixen (`"query: "` / `"passage: "`). Jeder Embedder-Wechsel
erfordert ein vollständiges Re-Ingest der ChromaDB (unterschiedliche Vektordimensionen).

**Interpretation:**
- **Embedder beeinflusst Retrieval, nicht Generation.** `context_precision` und
  `context_recall` steigen mit OpenAI signifikant; `faithfulness` und `answer_relevancy`
  bleiben praktisch identisch. Plausibel, weil derselbe Generator-LLM die finalen Antworten
  erzeugt und seine Beschränkungen den Retrieval-Vorsprung weitgehend nivellieren.
- **Endqualitätsgewinn ist real, aber klein.** `answer_correctness` +0.056 — messbar, aber
  kein Game-Changer.
- **OpenAI ist auch *stabiler*.** σ_OpenAI < σ_e5 quer durch alle Metriken — der deutsche
  Korpus wird konsistenter retrievet und reduziert die Streuung im Judge-Input.

**Verworfene Hypothesen.** Vorherige Iterationen variierten Top-K (k=5 vs. k=10) und
Chunking (RecursiveSplit vs. Section-aware) jeweils mit OpenAI. Alle gemessenen Deltas
zur Baseline lagen unter dem Rausch-Floor (σ ≈ 0.02–0.04 je Metrik). Nicht als
Verbesserung reproduzierbar — daher hier nicht als separate Tabellenzeilen geführt.

**Bekannte methodische Limitationen** (offene Roadmap):
- Die Frageformulierung verwirft den These-Text und fragt generisch
  „Was ist die Position der {Partei} zum Thema {Titel}?" — der Gap zwischen breiter Frage
  und spezifischer BpB-Begründung erklärt vermutlich einen Teil des ~0.58-Niveaus bei
  `context_recall`.
- `answer_correctness` vergleicht ausführlichen RAG-Fließtext gegen formelhafte
  BpB-Kurzbegründung — der niedrige Absolut-Wert (~0.34) reflektiert auch eine semantisch
  unfaire Vergleichsbasis, nicht reine Korrektheit. Eine Stance-Accuracy
  (zu / neutral / nicht zu) als kategoriale Metrik fehlt bisher.
- Faithfulness ist durch HyDE strukturell aufgebläht; eine Ablation mit/ohne HyDE fehlt.

```bash
python -m evaluation.evaluate_ground_truth --sample 35 --label baseline_k5_openai
python -m evaluation.evaluate_ground_truth                                          # Vollständiger Lauf (38×7=266)
# → evaluation/runs/{label}_scores.json  (Aggregate, gepusht)
# → evaluation/runs/{label}_raw.csv      (Per-Row, gitignored — enthält BpB-Begründungen)
```

> *Ground Truth: © Bundeszentrale für politische Bildung. Datensatz nicht im Repo enthalten;
> nur Analyseergebnisse werden veröffentlicht. Nutzung zu analytischen Zwecken gemäß BpB-Lizenz.*

---

## API-Endpunkte

| Methode | Pfad                      | Beschreibung                                             |
|---------|---------------------------|----------------------------------------------------------|
| POST    | `/api/v1/ingest`          | PDF einlesen (Admin/Dev – nicht in der UI)              |
| POST    | `/api/v1/query`           | Frage stellen, strukturierte Antwort mit Quellen        |
| POST    | `/api/v1/compare`         | Parteien vergleichen mit Stance-Klassifizierung         |
| GET     | `/api/v1/factsheet/{party}` | Fact-Sheet einer Partei abrufen                       |
| GET     | `/api/v1/health`          | Liveness-Check                                           |

Vollständige Dokumentation: `http://localhost:8000/docs`

---

## Architektur-Entscheidungen

**ChromaDB statt Qdrant/Pinecone**
Bewusst für lokales Setup ohne externe Dienste gewählt. Der gesamte ChromaDB-Zugriff
ist hinter `add_chunks()`, `delete_party()` und `similarity_search()` in `vector_store.py`
gekapselt – ein Wechsel auf einen anderen Vector Store würde nur diese Datei betreffen.

**Offline-Ingestion statt Live-Upload**
Parteiprogramme werden einmalig vorab verarbeitet. Das trennt den Dev-Workflow
(Daten aufbereiten) klar vom User-Workflow (Daten abfragen) und reduziert
Latenz und Kosten für Endnutzer.

**HyDE vor dem Retrieval**
Verbesserung der Retrieval-Qualität bei politischem Fachtext durch semantische
Brücke zwischen Fragenformulierung und Programmtext. Trade-off: +1 LLM-Call pro Anfrage.

**Pro-Partei-Retrieval beim Vergleich**
Statt eines globalen Top-K wird für jede Partei separat abgerufen. Garantiert gleiche
Repräsentation unabhängig von Programmlänge oder Sprachdichte.

**Stance Detection im selben LLM-Call wie der Vergleich**
Kein separater Klassifizierungs-Prompt. Die Stance wird aus dem Kontext inline abgeleitet –
spart einen LLM-Call und hält die Klassifizierung konsistent mit der generierten Position.

**LCEL für die RAG-Chains**
LangChain Expression Language erlaubt deklarative Komposition der Pipeline-Schritte
(`prompt | llm | parser`). Einzelne Stufen lassen sich austauschen ohne die
Orchestrierungslogik anzufassen.

**Pydantic Settings als Single Source of Truth**
Alle Konfigurationswerte an einer Stelle, typisiert und validiert. `openai_api_key`
als `SecretStr` verhindert versehentliches Loggen. Fehlt der Key beim Start,
schlägt der Lifespan-Hook sofort fehl – kein stilles Versagen beim ersten API-Call.

**Synchrone Endpunkte**
FastAPI führt synchrone Endpunkte in einem Thread Pool aus. Für dieses Demo
akzeptabel; für Production wäre async Ingestion mit `BackgroundTasks` der nächste Schritt.
