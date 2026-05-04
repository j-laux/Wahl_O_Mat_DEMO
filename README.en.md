# Wahl-O-Mat DEMO

RAG-based web app for analysing German party manifestos for the **2025 federal election (Bundestagswahl 2025)**.
Users can ask questions about political topics, receive answers with source references,
and compare the positions of multiple parties in a structured format — including automatic
stance classification (progressive / conservative / neutral).

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
│   ├── questions.json         # 25 test questions across 10 topics
│   └── evaluate.py            # RAGAS evaluation script
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

The pipeline is evaluated with [RAGAS](https://github.com/explodinggradients/ragas) on two metrics:

| Metric              | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| **faithfulness**    | Is the answer grounded in the retrieved chunks?                   |
| **answer_relevancy**| Does the answer actually address the question asked?              |

Both metrics require no manually created reference answers — only question, answer, and context.

**Test set:** 25 questions across 10 topics (climate, economy, housing, migration, education, pensions,
digitalisation, Europe, security, healthcare) — mix of single-party filter and all-party queries.

**Results** (gpt-4o-mini, 25/25 questions, 7 parties):

| Metric              | Score |
|---------------------|-------|
| faithfulness        | 0.77  |
| answer_relevancy    | 0.92  |

The high answer_relevancy confirms that HyDE + retrieval consistently pulls the right chunks.
The faithfulness score of 0.77 is expected for dense political text — the LLM occasionally
synthesises across chunks rather than citing strictly.

### Run the evaluation

```bash
pip install -r requirements-eval.txt
python -m evaluation.evaluate
# Output: evaluation/results.json
```

> Run from the project root so that `.env` and ChromaDB are found.
> Requires all party manifestos to be ingested into ChromaDB (~50 LLM calls + RAGAS calls).

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
