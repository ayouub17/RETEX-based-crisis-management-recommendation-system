"""High-level semantic RETEX search service."""

from __future__ import annotations

from typing import Any

import numpy as np

from embeddings.embedder import RetexEmbedder


class SemanticSearchEngine:
    """Search similar RETEX cases using cosine similarity."""

    def __init__(
        self,
        index: Any,
        metadata: list[dict[str, Any]],
        embedder: RetexEmbedder,
    ) -> None:
        self._index = index
        self._metadata = metadata
        self._embedder = embedder

    def search(self, crisis_description: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the Top-K closest RETEX cases and their cosine scores."""
        if not crisis_description.strip():
            raise ValueError("La description de la nouvelle crise est obligatoire.")
        if top_k < 1:
            raise ValueError("top_k doit être supérieur ou égal à 1.")

        query_vector = self._embedder.encode([crisis_description])
        limit = min(top_k, self._index.ntotal)
        scores, positions = self._index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32), limit
        )
        return [
            {"similarity_score": float(score), **self._metadata[position]}
            for score, position in zip(scores[0], positions[0])
            if position != -1
        ]
