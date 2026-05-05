"""
Embedding-Karte: UMAP-Visualisierung aller Parteiprogramm-Chunks im Vektorraum.
"""

import os

import plotly.express as px
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

_PARTY_COLORS = {
    "SPD":     "#E3000F",
    "CDU/CSU": "#161615",
    "Grüne":   "#46962b",
    "FDP":     "#FFCC00",
    "AfD":     "#009EE0",
    "Linke":   "#BE3075",
    "BSW":     "#6A0DAD",
}

st.set_page_config(
    page_title="Embedding-Karte – Wahl-O-Mat DEMO",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Parteiprogramme im Vektorraum")
st.caption(
    "Jeder Punkt ist ein Textabschnitt (Chunk) aus einem Parteiprogramm. "
    "Nähe im Raum = semantische Ähnlichkeit. "
    "UMAP reduziert die 1536-dimensionalen OpenAI-Embeddings auf 2D."
)


@st.cache_data(ttl=3600, show_spinner="Berechne Embedding-Karte (UMAP)…")
def load_embedding_map() -> pd.DataFrame:
    resp = requests.get(f"{API_BASE}/embeddings/map", timeout=120)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


try:
    df = load_embedding_map()
except requests.ConnectionError:
    st.error("Backend nicht erreichbar. Bitte zuerst das Backend starten.")
    st.stop()
except Exception as e:
    st.error(f"Fehler beim Laden der Karte: {e}")
    st.stop()

if df.empty:
    st.warning("Keine Embeddings gefunden. Bitte zuerst Parteiprogramme einlesen.")
    st.stop()

# ── Sidebar-Filter ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter")
    available_parties = sorted(df["party"].unique().tolist())
    selected = st.multiselect(
        "Parteien",
        available_parties,
        default=available_parties,
        key="embed_parties",
    )

df_filtered = df[df["party"].isin(selected)] if selected else df

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = px.scatter(
    df_filtered,
    x="x",
    y="y",
    color="party",
    color_discrete_map=_PARTY_COLORS,
    hover_data={"preview": True, "x": False, "y": False},
    labels={"party": "Partei", "preview": "Textausschnitt"},
    height=650,
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.update_layout(
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
    legend_title_text="Partei",
    plot_bgcolor="white",
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"{len(df_filtered):,} Chunks angezeigt von {len(df):,} gesamt · "
    "Hover über einen Punkt für den Textausschnitt"
)
