# Wahl-O-Mat DEMO

RAG-basierte Web-App zur Analyse deutscher Parteiprogramme zur **Bundestagswahl 2025**.
Nutzer können Fragen zu politischen Themen stellen und erhalten Antworten, die direkt aus den offiziellen Parteiprogrammen belegt sind.

---

## Architektur

```
┌─────────────────┐        HTTP        ┌──────────────────────────┐
│  Streamlit UI   │ ◄────────────────► │  FastAPI Backend          │
│  (frontend/)    │                    │  (backend/)               │
└─────────────────┘                    │                           │
                                       │  ┌─────────────────────┐  │
                                       │  │ Ingestion-Pipeline  │  │
                                       │  │  PDF → Chunks       │  │
                                       │  │       ↓             │  │
                                       │  │  ChromaDB           │  │
                                       │  └─────────────────────┘  │
                                       │                           │
                                       │  ┌─────────────────────┐  │
                                       │  │ RAG-Chain           │  │
                                       │  │  Query → Retrieval  │  │
                                       │  │       → LLM         │  │
                                       │  └─────────────────────┘  │
                                       └──────────────────────────┘
```

## Tech-Stack

| Komponente   | Technologie                           |
|--------------|---------------------------------------|
| Frontend     | Streamlit                             |
| Backend      | FastAPI + Uvicorn                     |
| RAG-Framework| LangChain                             |
| Vector Store | ChromaDB (lokal persistent)           |
| Embeddings   | OpenAI `text-embedding-3-small`       |
| LLM          | OpenAI `gpt-4o-mini`                  |
| PDF-Parsing  | LangChain PyPDFLoader                 |

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

### 2. Umgebungsvariablen konfigurieren

```bash
cp .env.example .env
# .env öffnen und OPENAI_API_KEY eintragen
```

### 3. PDFs ablegen

Parteiprogramme als PDF in `data/pdfs/` ablegen, z.B.:
```
data/pdfs/spd_programm.pdf
data/pdfs/cdu_programm.pdf
```

---

## Verwendung

### Backend starten

```bash
uvicorn backend.main:app --reload
# API läuft auf http://localhost:8000
# Swagger-Docs: http://localhost:8000/docs
```

### Frontend starten

```bash
streamlit run frontend/app.py
# UI läuft auf http://localhost:8501
```

### PDF direkt über die CLI einlesen

```bash
python -m backend.ingestion.pipeline --party SPD --file data/pdfs/spd_programm.pdf
```

---

## Projektstruktur

```
Wahl_O_Mat_DEMO/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI-Endpunkte (/ingest, /query, /health)
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF → LangChain Documents
│   │   ├── chunker.py         # Documents → Chunks
│   │   ├── vector_store.py    # ChromaDB-Wrapper
│   │   └── pipeline.py        # Orchestrierung (CLI-Skript)
│   ├── models/
│   │   └── schemas.py         # Pydantic-Modelle (Request/Response)
│   └── main.py                # FastAPI App-Instanz
├── frontend/
│   └── app.py                 # Streamlit-UI
├── data/
│   ├── pdfs/                  # Parteiprogramme (nicht im Repo)
│   └── chroma_db/             # ChromaDB-Daten (nicht im Repo)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## API-Endpunkte

| Methode | Pfad              | Beschreibung                              |
|---------|-------------------|-------------------------------------------|
| POST    | `/api/v1/ingest`  | PDF einlesen und in ChromaDB speichern    |
| POST    | `/api/v1/query`   | Frage stellen, relevante Chunks abrufen   |
| GET     | `/api/v1/health`  | Gesundheitsstatus des Backends            |

Vollständige Dokumentation: `http://localhost:8000/docs`
