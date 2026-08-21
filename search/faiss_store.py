"""Persistent FAISS index and RETEX metadata mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissStore:
    """Store normalised embeddings in an inner-product FAISS index."""

    @staticmethod
    def save(
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        """Create and persist a FAISS index plus aligned JSON metadata."""
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Les embeddings doivent former une matrice non vide.")
        if vectors.shape[0] != len(metadata):
            raise ValueError("Le nombre de vecteurs et de métadonnées diffère.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def load(index_path: Path, metadata_path: Path) -> tuple[Any, list[dict[str, Any]]]:
        """Load and validate the matching FAISS index and metadata file."""
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Index introuvable. Exécutez d'abord : python build_index.py"
            )
        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if index.ntotal != len(metadata):
            raise ValueError("Index FAISS et fichier de métadonnées incohérents.")
        return index, metadata
