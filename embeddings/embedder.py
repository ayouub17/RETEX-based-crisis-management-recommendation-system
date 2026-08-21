"""Sentence-Transformers wrapper used by indexing and querying."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class RetexEmbedder:
    """Generate L2-normalised vectors for cosine similarity search."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode text and return a two-dimensional float32 normalised array."""
        if not texts:
            raise ValueError("Au moins un texte est requis pour générer un embedding.")
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
