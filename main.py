"""Minimal Python API ready to integrate into the recommendation system."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

from config import Settings
from database.postgres import RetexRepository
from embeddings.embedder import RetexEmbedder
from search.engine import SemanticSearchEngine
from search.faiss_store import FaissStore
from recommendations.engine import RecommendationEngine


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return Settings.from_environment()


@lru_cache(maxsize=1)
def _get_metadata() -> list[dict[str, Any]]:
    settings = _get_settings()
    _, metadata = FaissStore.load(
        settings.faiss_index_path, settings.faiss_metadata_path
    )
    return metadata


@lru_cache(maxsize=1)
def get_search_engine() -> SemanticSearchEngine:
    """Load the model and persisted FAISS artifacts once per application run."""
    settings = _get_settings()
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


def get_dashboard_stats() -> dict[str, Any]:
    """Aggregate truthful metrics from the RETEX dataset used by the search index."""
    settings = _get_settings()
    repository = RetexRepository(settings)
    total_retex = repository.fetch_all()

    metadata = _get_metadata()

    crisis_distribution = Counter(
        str(case.get("crisis_type") or "Autre").strip().lower() or "autre"
        for case in metadata
    )
    categories = [
        {"label": label.title(), "count": count}
        for label, count in sorted(crisis_distribution.items(), key=lambda item: (-item[1], item[0]))[:6]
    ]

    return {
        "total_retex": len(total_retex),
        "organizations": len({str(case.get("organization") or "").strip() for case in metadata if case.get("organization")}),
        "countries": len({str(case.get("country") or "").strip() for case in metadata if case.get("country")}),
        "documents": len({str(case.get("url") or "").strip() for case in metadata if case.get("url")}),
        "categories": categories,
    }


def recommend_for_crisis(
    crisis_text: str,
    top_k: int = 5,
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Return weighted action proposals derived from the closest RETEX cases."""
    similar_cases = find_similar_retex(crisis_text, top_k)
    return RecommendationEngine().recommend(similar_cases, limit)
