"""Aggregate and rank operational recommendations from similar RETEX cases."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_SEPARATOR = re.compile(r"(?:\r?\n|;|•|\u2022)+")
_WHITESPACE = re.compile(r"\s+")


def _split_items(value: Any) -> list[str]:
    """Split a database action field into clean, display-ready items."""
    if value is None:
        return []
    return [
        _WHITESPACE.sub(" ", item).strip(" -\t")
        for item in _SEPARATOR.split(str(value))
        if item.strip(" -\t")
    ]


def _key(item: str) -> str:
    """Return a normalised key used to group equivalent recommendations."""
    return _WHITESPACE.sub(" ", item).strip().casefold()


class RecommendationEngine:
    """Build weighted action proposals from semantic-search results.

    A proposal receives the sum of non-negative similarity scores from all RETEX
    cases containing it. This gives more influence to closer cases while keeping
    the provenance needed for expert validation.
    """

    def recommend(
        self,
        similar_cases: list[dict[str, Any]],
        limit: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return ranked ``actions`` and ``recommendations`` with provenance."""
        if limit < 1:
            raise ValueError("limit doit être supérieur ou égal à 1.")
        return {
            "actions": self._rank_items(similar_cases, "actions", limit),
            "recommendations": self._rank_items(
                similar_cases, "recommendations", limit
            ),
        }

    @staticmethod
    def _rank_items(
        similar_cases: list[dict[str, Any]],
        field: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "sources": []}
        )
        for case in similar_cases:
            similarity = max(0.0, float(case.get("similarity_score", 0.0)))
            for item in _split_items(case.get(field)):
                entry = grouped[_key(item)]
                entry.setdefault("text", item)
                entry["score"] += similarity
                entry["sources"].append(
                    {
                        "retex_id": case.get("id"),
                        "title": case.get("title"),
                        "similarity_score": round(similarity, 4),
                    }
                )

        ranked = sorted(
            grouped.values(),
            key=lambda item: (-item["score"], item["text"]),
        )[:limit]
        return [
            {
                "text": item["text"],
                "weighted_score": round(item["score"], 4),
                "source_count": len(item["sources"]),
                "sources": item["sources"],
            }
            for item in ranked
        ]
