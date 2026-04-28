"""
Zentrale Anwendungskonfiguration via Pydantic Settings.

Werte werden in dieser Priorität gelesen:
  1. Echte Umgebungsvariablen (z.B. in Produktion oder CI)
  2. .env-Datei im Arbeitsverzeichnis (lokale Entwicklung)
  3. Defaults in dieser Klasse

Verwendung im Code:
    from backend.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unbekannte Env-Vars stillschweigend ignorieren
    )

    # OpenAI – Pflichtfeld, kein Default; fehlt der Key, schlägt der Start fehl
    openai_api_key: SecretStr

    # Embeddings & LLM
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0  # 0 = deterministisch, besser für faktische Q&A

    # ChromaDB
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection_name: str = "parteiprogramme"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Gibt die Singleton-Settings-Instanz zurück (lazy, einmalig initialisiert)."""
    return Settings()
