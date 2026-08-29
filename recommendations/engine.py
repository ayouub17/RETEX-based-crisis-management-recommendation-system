"""Aggregate and rank operational recommendations from similar RETEX cases."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_SEPARATOR = re.compile(r"(?:\r?\n|;|,|•|\u2022|\s-\s)+")
_WHITESPACE = re.compile(r"\s+")
_GENERIC_TAILS = ("etc", "et al", "autres", "divers")

_ACTION_PRIORITY_RULES = (
    (
        re.compile(
            r"\b(activate|alert|contain|secure|isolate|evacu|lockdown|quarantine|shutdown|protect|stop|warn|mobilis|communicat|notify|basculement|découper|isoler|alerter|protéger|évacuer|sécuriser|désactiver)\b",
            re.IGNORECASE,
        ),
        "urgence_immediate",
    ),
    (
        re.compile(
            r"\b(restore|patch|backup|recover|remed|audit|review|investigate|legal|contract|supplier|prepare|train|rehearse|replace|re-route|reroute|test|monitor|formation|sauvegard|continuit|segmentation|assurance|review|investigation|upgrade)\b",
            re.IGNORECASE,
        ),
        "court_terme",
    ),
)

_RECOMMENDATION_CLASSIFICATION_RULES = (
    (
        re.compile(
            r"\b(governance|board|leadership|policy|architecture|investment|roadmap|capacity|exercise|training|audit|oversight|risk|resilience|gouvernance|direction|stratégie|architecture|investissement|plan de continuité|risque)\b",
            re.IGNORECASE,
        ),
        "structurelle",
    ),
    (
        re.compile(
            r"\b(backup|access control|segregation|incident response|monitoring|password|patching|compliance|legal|supplier|vendor|communication|drill|review|sauvegard|contrôle d'accès|segmentation|monitoring|formation|assurance|cybersécurité)\b",
            re.IGNORECASE,
        ),
        "bonne_pratique_generale",
    ),
)


def _normalise_item(item: str) -> str:
    """Trim punctuation and keep operational wording from RETEX entries."""
    cleaned = _WHITESPACE.sub(" ", item).strip(" \t\n\r-•\u2022.\"'[]()")
    cleaned = cleaned.strip(";,:")
    return cleaned


def _is_meaningful_fragment(item: str) -> bool:
    """Reject low-signal fragments that are too generic to be actionable."""
    cleaned = _normalise_item(item).casefold()
    if not cleaned:
        return False
    if cleaned in _GENERIC_TAILS:
        return False
    if len(cleaned.split()) < 2 and not any(token in cleaned for token in ("plan", "service", "cyber", "backup", "test", "isolement", "sauvegard", "formation", "recovery", "crise", "sécurité")):
        return False
    return True


def _split_items(value: Any) -> list[str]:
    """Split a database action field into clean, display-ready items."""
    if value is None:
        return []
    parts: list[str] = []
    for item in _SEPARATOR.split(str(value)):
        cleaned = _normalise_item(item)
        if cleaned and _is_meaningful_fragment(cleaned):
            parts.append(cleaned)
    return parts


def _key(item: str) -> str:
    """Return a normalised key used to group equivalent recommendations."""
    return _WHITESPACE.sub(" ", item).strip().casefold()


def _item_specificity(item: str) -> float:
    """Prefer detailed, operational items over generic labels."""
    words = [word for word in _WHITESPACE.split(item) if word]
    score = 0.0
    score += min(len(words), 12) * 0.15
    if any(keyword in item.lower() for keyword in ("plan", "continuit", "backup", "sauvegard", "segmentation", "isolement", "formation", "test", "audit", "crise", "service", "sécurité", "réponse")):
        score += 0.6
    if any(keyword in item.lower() for keyword in ("assurance", "cyber", "gestion", "processus", "gouvernance")):
        score += 0.3
    return score


def _infer_priority(item: str) -> str:
    """Classify action priority according to the crisis response sequence."""
    text = item.lower()
    for pattern, priority in _ACTION_PRIORITY_RULES:
        if pattern.search(text):
            return priority
    return "stabilisation"


def _infer_classification(item: str) -> str:
    """Separate structural recommendations from generic industry practices."""
    text = item.lower()
    for pattern, classification in _RECOMMENDATION_CLASSIFICATION_RULES:
        if pattern.search(text):
            return classification
    return "bonne_pratique_generale"


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
        relevant_cases = self._filter_relevant_cases(similar_cases)
        return {
            "actions": self._rank_items(relevant_cases, "actions", limit),
            "recommendations": self._rank_items(
                relevant_cases, "recommendations", limit
            ),
        }

    @staticmethod
    def _filter_relevant_cases(similar_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Discard low-value and structurally mismatched cases before ranking."""
        filtered: list[dict[str, Any]] = []
        for case in similar_cases:
            similarity = max(0.0, float(case.get("similarity_score", 0.0)))
            if similarity < 0.12:
                continue
            crisis_type = str(case.get("crisis_type") or "").lower().strip()
            if crisis_type and crisis_type not in {"cyber", "industriel", "financier", "social", "logistique"}:
                continue
            filtered.append(case)
        return filtered

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
            key=lambda item: (-item["score"] - _item_specificity(item["text"]) * 0.05, -_item_specificity(item["text"]), item["text"]),
        )[:limit]
        output: list[dict[str, Any]] = []
        for item in ranked:
            priority = _infer_priority(item["text"]) if "actions" == field else "n/a"
            classification = _infer_classification(item["text"]) if "recommendations" == field else priority
            output.append(
                {
                    "text": item["text"],
                    "weighted_score": round(item["score"], 4),
                    "source_count": len(item["sources"]),
                    "priority": priority,
                    "classification": classification,
                    "sources": item["sources"],
                }
            )
        return output
