"""Centralised application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for PostgreSQL and the vector index."""

    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    embedding_model: str
    faiss_index_path: Path
    faiss_metadata_path: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build settings and fail early when a required variable is absent."""
        required = (
            "POSTGRES_HOST",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        )
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError(
                "Variables d'environnement manquantes : " + ", ".join(missing)
            )

        try:
            port = int(os.getenv("POSTGRES_PORT", "5432"))
        except ValueError as error:
            raise ValueError("POSTGRES_PORT doit être un entier.") from error

        return cls(
            postgres_host=os.environ["POSTGRES_HOST"],
            postgres_port=port,
            postgres_db=os.environ["POSTGRES_DB"],
            postgres_user=os.environ["POSTGRES_USER"],
            postgres_password=os.environ["POSTGRES_PASSWORD"],
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            faiss_index_path=PROJECT_ROOT
            / os.getenv("FAISS_INDEX_PATH", "data/faiss/retex.index"),
            faiss_metadata_path=PROJECT_ROOT
            / os.getenv("FAISS_METADATA_PATH", "data/faiss/retex_metadata.json"),
        )
