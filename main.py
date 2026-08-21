"""Minimal Python API ready to integrate into the recommendation system."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from config import Settings
from embeddings.embedder import RetexEmbedder
from search.engine import SemanticSearchEngine
from search.faiss_store import FaissStore
from recommendations.engine import RecommendationEngine


@lru_cache(maxsize=1)
def get_search_engine() -> SemanticSearchEngine:
    """Load the model and persisted FAISS artifacts once per application run."""
    settings = Settings.from_environment()
    index, metadata = FaissStore.load(
        settings.faiss_index_path, settings.faiss_metadata_path
    )
    return SemanticSearchEngine(
        index=index,
        metadata=metadata,
        embedder=RetexEmbedder(settings.embedding_model),
    )


def find_similar_retex(crisis_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the RETEX cases most semantically similar to a crisis."""
    return get_search_engine().search(crisis_text, top_k)


def recommend_for_crisis(
    crisis_text: str,
    top_k: int = 5,
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Return weighted action proposals derived from the closest RETEX cases."""
    similar_cases = find_similar_retex(crisis_text, top_k)
    return RecommendationEngine().recommend(similar_cases, limit)
