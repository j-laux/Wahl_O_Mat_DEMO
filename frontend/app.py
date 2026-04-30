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

_STANCE_BADGE = {
    "progressiv":   "🟢 progressiv",
    "konservativ":  "🔵 konservativ",
    "neutral":      "⚫ neutral",
    "unklar":       "🟡 unklar",
}

st.set_page_config(
    page_title="Wahl-O-Mat DEMO",
    page_icon="🗳️",
    layout="wide",
)

st.title("🗳️ Wahl-O-Mat DEMO")
st.caption("RAG-basierte Analyse deutscher Parteiprogramme zur Bundestagswahl 2025")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Partei-Fact-Sheet")
    factsheet_party = st.selectbox("Partei auswählen", PARTIES, key="factsheet_select")
    if st.button("Fact-Sheet laden"):
        try:
            resp = requests.get(f"{API_BASE}/factsheet/{factsheet_party}", timeout=10)
            if resp.status_code == 404:
                st.warning("Noch kein Fact-Sheet vorhanden. Bitte zuerst das PDF einlesen.")
            else:
                resp.raise_for_status()
                fs = resp.json()
                st.markdown("**Politische Positionierung**")
                st.write(fs["political_position"])
                st.markdown("**Kernthemen**")
                for topic in fs["top_topics"]:
                    st.markdown(f"- {topic}")
                st.markdown("**Kernversprechen**")
                for promise in fs["key_promises"]:
                    st.markdown(f"- {promise}")
                st.caption(f"Generiert: {fs['generated_at'][:10]}")
        except Exception as e:
            st.error(f"Fehler: {e}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_query, tab_compare = st.tabs(["💬 Frage stellen", "⚖️ Parteien vergleichen"])

# ── Tab 1: Einzelabfrage ──────────────────────────────────────────────────────
with tab_query:
    col_filter, col_main = st.columns([1, 3])

    with col_filter:
        st.markdown("**Filter**")
        selected_parties = st.multiselect(
            "Parteien",
            PARTIES,
            default=[],
            placeholder="Alle Parteien",
            key="query_parties",
        )
        top_k = st.slider("Textstellen", min_value=1, max_value=20, value=5, key="query_topk")

    with col_main:
        question = st.text_area(
            "Deine Frage",
            placeholder="Was sagen die Parteien zum Thema Klimaschutz?",
            height=100,
            key="query_input",
        )

        if st.button("Antwort suchen", type="primary", key="query_submit"):
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

                        st.subheader("Zusammenfassung")
                        st.write(data["summary"])

                        if data["positions"]:
                            st.subheader("Positionen der Parteien")
                            cols = st.columns(min(len(data["positions"]), 3))
                            for i, pos in enumerate(data["positions"]):
                                with cols[i % len(cols)]:
                                    st.markdown(f"**{pos['party']}**")
                                    st.write(pos["position"])

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

# ── Tab 2: Parteienvergleich ──────────────────────────────────────────────────
with tab_compare:
    col_filter2, col_main2 = st.columns([1, 3])

    with col_filter2:
        st.markdown("**Parteien auswählen**")
        compare_parties = st.multiselect(
            "Mindestens 2",
            PARTIES,
            default=["SPD", "Grüne"],
            key="compare_parties",
        )
        top_k_per_party = st.slider(
            "Textstellen pro Partei", min_value=1, max_value=10, value=3, key="compare_topk"
        )

    with col_main2:
        compare_question = st.text_area(
            "Vergleichsfrage",
            placeholder="Was planen die Parteien beim Thema Wohnungsbau?",
            height=100,
            key="compare_input",
        )

        if st.button("Vergleich starten", type="primary", key="compare_submit"):
            if not compare_question.strip():
                st.warning("Bitte eine Frage eingeben.")
            elif len(compare_parties) < 2:
                st.warning("Bitte mindestens 2 Parteien auswählen.")
            else:
                with st.spinner("Vergleiche Parteiprogramme..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/compare",
                            json={
                                "question": compare_question,
                                "parties": compare_parties,
                                "top_k_per_party": top_k_per_party,
                            },
                            timeout=90,
                        )
                        resp.raise_for_status()
                        data = resp.json()

                        st.subheader("Zusammenfassung")
                        st.write(data["summary"])

                        st.subheader("Parteienvergleich")
                        cols = st.columns(min(len(data["comparisons"]), 3))
                        for i, cmp in enumerate(data["comparisons"]):
                            with cols[i % len(cols)]:
                                badge = _STANCE_BADGE.get(cmp["stance"], cmp["stance"])
                                st.markdown(f"**{cmp['party']}** &nbsp; {badge}")
                                st.write(cmp["position"])
                                if cmp["key_points"]:
                                    for point in cmp["key_points"]:
                                        st.markdown(f"- {point}")

                        st.subheader("Relevante Textstellen")
                        for i, src in enumerate(data["sources"], 1):
                            with st.expander(
                                f"[{i}] {src['party']} – Seite {src['page']}", expanded=False
                            ):
                                st.write(src["content"])

                    except requests.HTTPError as e:
                        st.error(f"API-Fehler: {e.response.json().get('detail', str(e))}")
                    except Exception as e:
                        st.error(f"Verbindungsfehler: {e}")
