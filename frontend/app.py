"""
Streamlit-Frontend: Wahl-O-Mat DEMO
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

PARTIES = ["SPD", "CDU/CSU", "Grüne", "FDP", "AfD", "Linke", "BSW"]

st.set_page_config(
    page_title="Wahl-O-Mat DEMO",
    page_icon="🗳️",
    layout="wide",
)

st.title("🗳️ Wahl-O-Mat DEMO")
st.caption("RAG-basierte Analyse deutscher Parteiprogramme zur Bundestagswahl 2025")

# ── Sidebar: Filter ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filter")
    selected_parties = st.multiselect(
        "Nur diese Parteien durchsuchen",
        PARTIES,
        default=[],
        placeholder="Alle Parteien",
    )
    top_k = st.slider("Anzahl Textstellen", min_value=1, max_value=20, value=5)

# ── Hauptbereich: Abfrage ─────────────────────────────────────────────────────
st.header("💬 Frage stellen")
question = st.text_area(
    "Deine Frage",
    placeholder="Was sagen die Parteien zum Thema Klimaschutz?",
    height=100,
)

if st.button("Antwort suchen", type="primary"):
    if not question.strip():
        st.warning("Bitte eine Frage eingeben.")
    else:
        with st.spinner("Suche in Parteiprogrammen..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": question,
                        "parties": selected_parties if selected_parties else None,
                        "top_k": top_k,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                st.subheader("Antwort")
                st.write(data["answer"])

                st.subheader("Relevante Textstellen")
                for i, src in enumerate(data["sources"], 1):
                    with st.expander(
                        f"[{i}] {src['party']} – Seite {src['page']}", expanded=i == 1
                    ):
                        st.write(src["content"])

            except requests.HTTPError as e:
                st.error(f"API-Fehler: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Verbindungsfehler: {e}")
