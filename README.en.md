# Wahl-O-Mat DEMO

RAG-based analysis platform for the **2025 German federal election (Bundestagswahl 2025)** —
Version 1 of a multi-layer political knowledge system. Users can ask questions about political
topics, receive answers with source references, and compare the positions of multiple parties
in a structured format — including automatic stance classification (progressive / conservative / neutral).

> **Vision:** The system is designed to integrate three knowledge layers: **(1) electoral promises**
> from party manifestos (this version), **(2) voting behaviour** via Abgeordnetenwatch, and
> **(3) current rhetoric** from news sources — giving voters a complete picture beyond
> campaign communications.

---

## Features

| Feature | Description |
|---|---|
| **RAG with HyDE** | Questions are converted into hypothetical manifesto text before retrieval for better chunk quality |
| **Structured Output** | LLM returns typed JSON (Pydantic + OpenAI Function Calling), no string parsing |
| **Party Comparison** | `/compare` contrasts party positions, each with stance classification |
| **Stance Detection** | LLM classifies each party as progressive / conservative / neutral / unclear |
| **Fact-Sheets** | Auto-generated during ingestion: core topics, promises, political positioning |
| **Deduplication** | Re-ingesting a party replaces existing chunks — no duplicates in ChromaDB |

---

## Architecture

The project deliberately separates two workflows:

**Dev workflow (one-time, offline):**
```
PDF files  →  Ingestion pipeline  →  ChromaDB (persistent on disk)
               └─ PyPDFLoader                   + Fact-Sheet (JSON)
               └─ RecursiveCharacterTextSplitter
               └─ OpenAI Embeddings
```

**User workflow – single query (`/query`):**
```
User question
    │
    ▼  [1] HyDE: LLM generates hypothetical manifesto text
Hypothetical text
    │
    ▼  Embedding + ChromaDB Similarity Search
Relevant chunks (party, page, content)
    │
    ▼  [2] RAG: LLM answers the original question (with_structured_output)
{ summary, positions: [{ party, position }], sources }
```

**User workflow – party comparison (`/compare`):**
```
Question + list of parties
    │
    ▼  [1] HyDE: once for all parties
Hypothetical text
    │
    ▼  Retrieval per party separately (equal chunk count per party)
Chunks per party { "SPD": [...], "Grüne": [...], ... }
    │
    ▼  [2] Comparison chain (with_structured_output)
{ summary, comparisons: [{ party, position, key_points, stance }], sources }
```

**Why HyDE?**
The embedding distance between a short user question and a long piece of running text
is structurally larger than between two semantically similar texts. HyDE bridges
this gap by using a synthetic answer text as the search vector instead of the original question.
Trade-off: +1 LLM call per request.

**Why per-party retrieval for comparison?**
A global top-K would favour larger or linguistically dominant party manifestos.
Per-party retrieval guarantees equal representation in the LLM context regardless of manifesto length.

---

## Tech Stack

| Component       | Technology                                 |
|-----------------|--------------------------------------------|
| Frontend        | Streamlit                                  |
| Backend         | FastAPI + Uvicorn                          |
| Configuration   | Pydantic Settings (`pydantic-settings`)    |
| RAG Framework   | LangChain (LCEL)                           |
| Vector Store    | ChromaDB (locally persistent)              |
| Embeddings      | OpenAI `text-embedding-3-small`            |
| LLM             | OpenAI `gpt-4o-mini`                       |
| PDF Parsing     | LangChain `PyPDFLoader`                    |

---

## Setup

### 1. Clone the repository & create a virtual environment

```bash
git clone <repo-url>
cd Wahl_O_Mat_DEMO

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Requires Python 3.10+.

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

All other values have sensible defaults and do not need to be changed.

### 3. Ingest party manifestos (dev step, one-time only)

Party manifestos are pre-processed and stored as a ChromaDB on disk.
End users do not need to run this step.

```bash
python -m backend.ingestion.pipeline --party SPD    --file data/pdfs/spd_programm.pdf
python -m backend.ingestion.pipeline --party Grüne  --file data/pdfs/gruene_programm.pdf
python -m backend.ingestion.pipeline --party CDU    --file data/pdfs/cdu_programm.pdf
# ... further parties
```

A fact-sheet is automatically saved to `data/factsheets/` for each party.
Re-ingesting a party automatically replaces the existing chunks.

---

## Usage

### Start the backend

```bash
uvicorn backend.main:app --reload
# API:          http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Start the frontend

```bash
streamlit run frontend/app.py
# UI runs at http://localhost:8501
```

---

## Project Structure

```
Wahl_O_Mat_DEMO/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF → LangChain Documents + metadata
│   │   ├── chunker.py         # RecursiveCharacterTextSplitter
│   │   ├── vector_store.py    # ChromaDB wrapper (add, delete, search)
│   │   └── pipeline.py        # Orchestration: Delete → Load → Chunk → Store → Fact-Sheet
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── rag/
│   │   ├── chain.py           # LCEL chain: HyDE → Retrieval → Structured Answer
│   │   ├── compare.py         # Party comparison with Stance Detection
│   │   └── factsheet.py       # Fact-Sheet generation during ingestion
│   ├── config.py              # Central configuration via Pydantic Settings
│   └── main.py                # FastAPI app instance with lifespan hook
├── frontend/
│   └── app.py                 # Streamlit UI (Query tab + Comparison tab)
├── evaluation/
│   ├── ground_truth/
│   │   └── build_dataset.py   # BpB Excel → ground_truth.json (dataset gitignored)
│   ├── evaluate.py            # RAGAS Eval 1: faithfulness + answer_relevancy
│   ├── evaluate_ground_truth.py  # RAGAS Eval 2: 5 metrics with ground truth
│   └── questions.json         # Test set for Eval 1
├── data/
│   ├── pdfs/                  # Party manifestos as PDFs (not in repo)
│   ├── chroma_db/             # ChromaDB data (not in repo)
│   └── factsheets/            # Generated fact-sheets as JSON
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-eval.txt
└── README.md
```

---

## Evaluation (RAGAS)

The pipeline is evaluated at two levels using [RAGAS](https://github.com/explodinggradients/ragas):
a general quality assessment and a ground-truth validation against official party positions.

### Eval 1 – General Pipeline Quality

No reference answers required — only question, answer, and retrieved context.

**Test set:** 25 questions across 10 topics (climate, economy, housing, migration, education, pensions,
digitalisation, Europe, security, healthcare) — mix of single-party filter and all-party queries.

| Metric               | Score | Description                                               |
|----------------------|-------|-----------------------------------------------------------|
| **faithfulness**     | 0.77  | Is the answer grounded in the retrieved chunks?          |
| **answer_relevancy** | 0.92  | Does the answer actually address the question asked?     |

The high answer_relevancy confirms that HyDE + retrieval consistently pulls thematically relevant chunks.
The faithfulness score of 0.77 is expected for dense political text — the LLM occasionally
synthesises across chunks rather than citing strictly.

```bash
pip install -r requirements-eval.txt
python -m evaluation.evaluate
# → evaluation/results.json
```

### Eval 2 – Ground-Truth Validation (Wahl-O-Mat Theses)

Validation against the official BpB Wahl-O-Mat 2025 dataset: 38 political theses with official
party positions (agree / neutral / disagree) and reasoning as reference answers. Enables three
additional metrics that directly measure retrieval quality and factual correctness.

**Experiment results** (gpt-4o-mini, n=35, stratified sample 5 theses × 7 parties):

| Variant                                           | faithfulness | answer_relevancy | ctx_precision | ctx_recall | answer_correctness |
|---------------------------------------------------|-------------|-----------------|--------------|-----------|-------------------|
| RecursiveSplit k=5 + OpenAI (baseline)            | 0.919       | 0.738           | 0.746        | **0.621** | **0.641**         |
| RecursiveSplit k=10 + OpenAI                      | 0.959       | 0.764           | 0.746        | 0.625     | 0.611             |
| Section-aware chunking k=5 + OpenAI               | 0.939       | **0.767**       | **0.760**    | 0.608     | 0.606             |
| RecursiveSplit k=5 + multilingual-e5-large        | **0.969**   | 0.720           | 0.700        | 0.536     | 0.281             |

OpenAI embedding is `text-embedding-3-small`. The e5 run uses the instruction prefixes
(`"query: "` / `"passage: "`) required by the model and was evaluated against a freshly
re-ingested ChromaDB (an earlier run without prefixes was discarded as invalid).

**Interpretation:** The three OpenAI variants sit so close together on `context_recall`
(0.608 – 0.625) that, at n=35 and without bootstrap confidence intervals, the differences are
plausibly within LLM-judge noise — these data **do not support** the claim that top-K or
section-aware chunking moves `context_recall`. The switch to `multilingual-e5-large` shows a
much larger effect (Δ `context_recall` ≈ −0.086, Δ `answer_correctness` ≈ −0.36) that exceeds
typical judge variance, but is not formally certified without CIs. Plausible reason: e5-large is
primarily optimised for cross-lingual retrieval, whereas `text-embedding-3-small` was trained on
a heavily German-inclusive corpus. What these data actually support is: **one specific
multilingual embedder does not improve this configuration** — the broader hypothesis that
embedding alignment is the bottleneck remains open.

**Known methodological limitations** (open roadmap):
- n=35 is statistically thin; bootstrap CIs on per-row scores are missing.
- LLM-judge variance has not been measured via repeated runs.
- The question template discards the thesis text and asks the generic form
  "What is the position of {party} on the topic {title}?" — the gap between this broad question
  and the specific BpB reasoning probably accounts for part of the ~0.62 `context_recall` ceiling.
- `answer_correctness` compares verbose RAG prose against terse BpB official reasoning — the
  baseline value (~0.64) partly reflects a semantically unfair comparison rather than pure
  correctness. A categorical stance-accuracy metric (agree / neutral / disagree) is still missing.
- Faithfulness is structurally inflated by HyDE; an ablation with/without HyDE is missing.
- The four variants vary non-orthogonally (no section-aware × k=10, no e5 × k=10) — effects
  cannot be cleanly isolated.

```bash
python -m evaluation.evaluate_ground_truth --sample 35   # dev run
python -m evaluation.evaluate_ground_truth               # full run (38×7=266)
# → evaluation/results_ground_truth.json
```

> *Ground truth: © Bundeszentrale für politische Bildung. Dataset not included in this repository;
> only analysis results are published. Use for analytical purposes in accordance with the BpB licence.*

---

## API Endpoints

| Method | Path                        | Description                                              |
|--------|-----------------------------|----------------------------------------------------------|
| POST   | `/api/v1/ingest`            | Ingest a PDF (admin/dev — not exposed in the UI)        |
| POST   | `/api/v1/query`             | Ask a question, get a structured answer with sources    |
| POST   | `/api/v1/compare`           | Compare parties with stance classification              |
| GET    | `/api/v1/factsheet/{party}` | Retrieve the fact-sheet for a party                     |
| GET    | `/api/v1/health`            | Liveness check                                           |

Full documentation: `http://localhost:8000/docs`

---

## Architecture Decisions

**ChromaDB instead of Qdrant/Pinecone**
Deliberately chosen for a local setup without external services. All ChromaDB access
is encapsulated behind `add_chunks()`, `delete_party()`, and `similarity_search()` in `vector_store.py` —
switching to a different vector store would only affect that one file.

**Offline ingestion instead of live upload**
Party manifestos are processed once in advance. This clearly separates the dev workflow
(preparing data) from the user workflow (querying data) and reduces latency and costs for end users.

**HyDE before retrieval**
Improves retrieval quality for political text by building a semantic bridge between
the question phrasing and manifesto language. Trade-off: +1 LLM call per request.

**Per-party retrieval for comparison**
Instead of a global top-K, retrieval is performed separately for each party. This guarantees
equal representation regardless of manifesto length or linguistic density.

**Stance detection in the same LLM call as the comparison**
No separate classification prompt. Stance is derived inline from the context —
saves one LLM call and keeps the classification consistent with the generated position.

**LCEL for RAG chains**
LangChain Expression Language allows declarative composition of pipeline steps
(`prompt | llm | parser`). Individual stages can be swapped out without touching
the orchestration logic.

**Pydantic Settings as single source of truth**
All configuration values in one place, typed and validated. `openai_api_key`
as `SecretStr` prevents accidental logging. If the key is missing at startup,
the lifespan hook fails immediately — no silent failure on the first API call.

**Synchronous endpoints**
FastAPI runs synchronous endpoints in a thread pool. Acceptable for this demo;
for production, async ingestion with `BackgroundTasks` would be the natural next step.
