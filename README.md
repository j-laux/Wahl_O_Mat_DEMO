# Wahl-O-Mat DEMO

RAG-basierte Web-App zur Analyse deutscher Parteiprogramme zur **Bundestagswahl 2025**.
Nutzer können Fragen zu politischen Themen stellen und erhalten Antworten, die direkt aus den offiziellen Parteiprogrammen belegt sind – mit Quellenangaben (Partei + Seite).

> **Portfolio-Projekt** – Fokus liegt auf der KI-Pipeline und der Architektur, nicht auf einem Produktlaunch.

---

## Architektur

Das Projekt trennt bewusst zwei Workflows:

**Dev-Workflow (einmalig, offline):**
```
PDF-Dateien  →  Ingestion-Pipeline  →  ChromaDB (persistent auf Disk)
                 └─ PyPDFLoader
                 └─ RecursiveCharacterTextSplitter
                 └─ OpenAI Embeddings
```

**User-Workflow (pro Anfrage):**
```
Nutzerfrage
    │
    ▼  [1] HyDE: LLM generiert hypothetischen Programmtext
Hypothetischer Text
    │
    ▼  Embedding + ChromaDB Similarity Search
Relevante Chunks (Partei, Seite, Inhalt)
    │
    ▼  [2] RAG: LLM beantwortet Original-Frage mit Kontext
Antwort mit Quellenangaben [Partei, S. X]
```

**Warum HyDE?**
Die Embedding-Distanz zwischen einer kurzen Nutzerfrage und einem langen Fließtext aus einem Parteiprogramm ist strukturell größer als die Distanz zwischen zwei inhaltlich ähnlichen Texten. HyDE überbrückt diesen Raum, indem der Suchvektor nicht aus der Frage, sondern aus einem synthetischen Antwort-Text berechnet wird. Kostet einen zweiten LLM-Call – verbessert die Retrieval-Qualität bei politischem Fachtext messbar.

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
python -m backend.ingestion.pipeline --party SPD --file data/pdfs/spd_programm.pdf
python -m backend.ingestion.pipeline --party CDU --file data/pdfs/cdu_programm.pdf
# ... weitere Parteien
```

Wird eine Partei erneut eingelesen, werden die alten Chunks automatisch ersetzt (Deduplizierung).

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
│   │   └── routes.py          # FastAPI-Endpunkte (/ingest, /query, /health)
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF → LangChain Documents + Metadaten
│   │   ├── chunker.py         # RecursiveCharacterTextSplitter
│   │   ├── vector_store.py    # ChromaDB-Wrapper (add, delete, search)
│   │   └── pipeline.py        # Orchestrierung: Delete → Load → Chunk → Store
│   ├── models/
│   │   └── schemas.py         # Pydantic Request/Response-Modelle
│   ├── rag/
│   │   └── chain.py           # LCEL-Chain: HyDE → Retrieval → Antwort
│   ├── config.py              # Zentrale Konfiguration via Pydantic Settings
│   └── main.py                # FastAPI App-Instanz mit Lifespan-Hook
├── frontend/
│   └── app.py                 # Streamlit-UI (Query + Filter)
├── data/
│   ├── pdfs/                  # Parteiprogramme als PDF (nicht im Repo)
│   └── chroma_db/             # ChromaDB-Daten (nicht im Repo)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## API-Endpunkte

| Methode | Pfad             | Beschreibung                                        |
|---------|------------------|-----------------------------------------------------|
| POST    | `/api/v1/ingest` | PDF einlesen (Admin/Dev – nicht in der UI)         |
| POST    | `/api/v1/query`  | Frage stellen, Antwort mit Quellenangaben erhalten |
| GET     | `/api/v1/health` | Liveness-Check                                      |

Vollständige Dokumentation: `http://localhost:8000/docs`

---

## Architektur-Entscheidungen

**ChromaDB statt Qdrant/Pinecone**
Bewusst für lokales Setup ohne externe Dienste gewählt. Der gesamte ChromaDB-Zugriff ist in `vector_store.py` hinter `add_chunks()`, `delete_party()` und `similarity_search()` gekapselt – ein Wechsel auf einen anderen Vector Store würde nur diese Datei betreffen.

**Offline-Ingestion statt Live-Upload**
Parteiprogramme werden einmalig vorab verarbeitet. Das reduziert Latenz und Kosten für Endnutzer und trennt den Dev-Workflow (Daten aufbereiten) klar vom User-Workflow (Daten abfragen).

**HyDE vor dem Retrieval**
Verbesserung der Retrieval-Qualität bei politischem Fachtext durch semantische Brücke zwischen Fragenformulierung und Programmtext. Trade-off: +1 LLM-Call pro Anfrage.

**LCEL für die RAG-Chain**
LangChain Expression Language erlaubt deklarative Komposition der Pipeline-Schritte (`prompt | llm | parser`). Einzelne Stufen (z.B. LLM-Modell, Output-Parser) lassen sich austauschen ohne die Orchestrierungslogik anzufassen.

**Pydantic Settings als Single Source of Truth**
Alle Konfigurationswerte (Modellnamen, Pfade, Chunking-Parameter) an einer Stelle, typisiert und validiert. `openai_api_key` als `SecretStr` verhindert versehentliches Loggen des Keys. Fehlt der Key beim Start, schlägt der Lifespan-Hook sofort fehl – kein stilles Versagen beim ersten API-Call.

**Synchrone Endpunkte**
FastAPI führt synchrone Endpunkte in einem Thread Pool aus. Für dieses Demo akzeptabel; für Production wäre async Ingestion mit `BackgroundTasks` der nächste Schritt.
