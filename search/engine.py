"""High-level semantic RETEX search service."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from embeddings.embedder import RetexEmbedder


_TRIGGER_HINTS = {
    "cyber": "compromission initiale de l’environnement numérique",
    "industriel": "défaillance technique ou opérationnelle du processus de production",
    "financier": "rupture de confiance, fraude ou volatilité financière",
    "social": "événement social ou humain ayant amplifié la crise",
    "logistique": "rupture de chaîne d’approvisionnement ou de transport",
}

_FAILURE_HINTS = {
    "cyber": "absence de gouvernance de sécurité, de segmentation réseau et de tests de continuité",
    "industriel": "absence de maîtrise des risques opérationnels et de maintenance préventive",
    "financier": "absence de pilotage de risque, de contrôle interne et de plan de continuité",
    "social": "absence de gestion des risques humains, de communication et d’anticipation",
    "logistique": "absence de redondance, de plan de secours et de visibilité sur la chaîne",
}


_CRISIS_TYPE_HINTS: dict[str, set[str]] = {
    "cyber": {
        "cyber",
        "cybersecurity",
        "attaque",
        "ransomware",
        "malware",
        "intrusion",
        "ddos",
        "sécurité",
        "security",
        "hack",
        "breach",
        "data",
        "fuite",
        "données",
        "exfiltration",
        "privacy",
    },
    "industriel": {
        "industriel",
        "industrielle",
        "industrie",
        "usine",
        "production",
        "atelier",
        "incident",
        "accident",
        "safety",
        "process",
        "machin",
        "manufacturing",
        "industrial",
        "facility",
    },
    "financier": {
        "finance",
        "financier",
        "financière",
        "fraud",
        "bank",
        "bancaire",
        "crash",
        "liquidite",
        "market",
        "payment",
        "cash",
        "banking",
        "credit",
    },
    "social": {
        "social",
        "societal",
        "grève",
        "strike",
        "violence",
        "protest",
        "labor",
        "communaut",
        "sanitaire",
        "public health",
        "healthcare",
        "medical",
    },
    "logistique": {
        "logistique",
        "supply",
        "chain",
        "transport",
        "shipping",
        "livraison",
        "distribution",
        "inventory",
        "fleet",
        "delivery",
        "shipment",
        "warehouse",
    },
}


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

    @staticmethod
    def _normalise_tokens(text: str) -> set[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ\s]", " ", text.lower())
        tokens = {token for token in cleaned.split() if len(token) > 2}
        return tokens - {"avec", "pour", "dans", "après", "avant", "entre", "suite", "ayant", "plus", "sans", "leur", "entre", "sur", "cette", "cette", "sont", "donc", "ainsi", "plusieurs", "avoir", "apres", "deux"}

    @classmethod
    def _keyword_overlap(cls, query_text: str, case: dict[str, Any]) -> float:
        case_text = " ".join(
            value for value in (
                case.get("title"),
                case.get("summary"),
                case.get("description"),
                case.get("crisis_type"),
                case.get("recommendations"),
                case.get("actions"),
            ) if value
        )
        if not case_text or not query_text:
            return 0.0
        query_tokens = cls._normalise_tokens(query_text)
        case_tokens = cls._normalise_tokens(case_text)
        if not query_tokens or not case_tokens:
            return 0.0
        return len(query_tokens & case_tokens) / max(len(query_tokens), 1)

    def search(self, crisis_description: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the Top-K closest RETEX cases after filtering irrelevant matches."""
        if not crisis_description.strip():
            raise ValueError("La description de la nouvelle crise est obligatoire.")
        if top_k < 1:
            raise ValueError("top_k doit être supérieur ou égal à 1.")

        query_vector = self._embedder.encode([crisis_description])
        query_type = self._infer_crisis_type(crisis_description)

        def _collect(limit: int, similarity_floor: float, overlap_floor: float, allow_mismatch: bool = False) -> list[dict[str, Any]]:
            scores, positions = self._index.search(
                np.ascontiguousarray(query_vector, dtype=np.float32), limit
            )
            selected: list[dict[str, Any]] = []

            for score, position in zip(scores[0], positions[0]):
                if position == -1 or position >= len(self._metadata):
                    continue
                case = dict(self._metadata[position])
                similarity = float(score)
                case_type = str(case.get("crisis_type") or "").lower().strip()
                overlap = self._keyword_overlap(crisis_description, case)
                compatible = bool(not query_type or not case_type or self._types_compatible(query_type, case_type))

                if similarity < similarity_floor and overlap < overlap_floor and not compatible:
                    continue

                if query_type and case_type and not self._types_compatible(query_type, case_type):
                    if overlap < overlap_floor + 0.04 and not allow_mismatch:
                        continue

                case["similarity_score"] = similarity
                case["trigger"] = self._extract_trigger(case)
                case["domino_effect"] = self._extract_domino_effect(case)
                case["structural_failure"] = self._extract_structural_failure(case, query_type or case_type)
                case["keyword_overlap"] = overlap
                selected.append(case)

            selected.sort(
                key=lambda item: (
                    float(item.get("similarity_score", 0.0)) + float(item.get("keyword_overlap", 0.0)) * 0.7,
                    float(item.get("keyword_overlap", 0.0)),
                    float(item.get("similarity_score", 0.0)),
                ),
                reverse=True,
            )
            return selected

        filtered = _collect(min(max(top_k * 8, 20), self._index.ntotal), 0.05, 0.12)
        if not filtered:
            filtered = _collect(min(max(top_k * 15, 40), self._index.ntotal), 0.02, 0.08, allow_mismatch=True)
        if not filtered:
            filtered = _collect(min(max(top_k * 25, 80), self._index.ntotal), 0.0, 0.0, allow_mismatch=True)

        return filtered[:top_k]

    @staticmethod
    def _extract_trigger(case: dict[str, Any]) -> str:
        for field in ("description", "summary", "title"):
            value = str(case.get(field) or "").strip()
            if value:
                sentence = re.split(r"(?<=[.!?])\s+", value)[0]
                if sentence:
                    return sentence[:220]
        return "Déclencheur non explicitement documenté dans le RETEX."

    @staticmethod
    def _extract_domino_effect(case: dict[str, Any]) -> str:
        description = str(case.get("description") or "")
        summary = str(case.get("summary") or "")
        text = " ".join(part for part in [summary, description] if part)
        if not text:
            return "La propagation de la crise a affecté les opérations critiques et la continuité d’activité."
        if re.search(r"(hôpital|hospital|service de santé|santé)", text, re.IGNORECASE):
            return "Les services critiques de santé ont été perturbés, entraînant une dégradation de la continuité des soins et une saturation de la crise opérationnelle."
        if re.search(r"(ransomware|cyber|attaque|malware|intrusion)", text, re.IGNORECASE):
            return "L’attaque a provoqué une perte de confiance, une interruption des systèmes critiques et une propagation des impacts au niveau organisationnel."
        if re.search(r"(pipeline|transport|livraison|supply|chain|logistique)", text, re.IGNORECASE):
            return "La rupture logistique a provoqué une interruption de la chaîne de valeur et une forte tension sur les activités critiques."
        return "L’incident a généré une cascade d’impact sur les opérations, la sécurité et la continuité d’activité de l’organisation."

    @staticmethod
    def _extract_structural_failure(case: dict[str, Any], crisis_type: str) -> str:
        text = " ".join(
            value for value in (case.get("summary"), case.get("description"), case.get("recommendations"), case.get("actions")) if value
        )
        lowered = text.lower()
        for key, failure in _FAILURE_HINTS.items():
            if crisis_type and key in crisis_type:
                return failure
        if re.search(r"(absence|manque|faible|insuffis|non|peu de)", lowered):
            return "défaut persistent de gouvernance, de préparation et de mécanismes de contrôle."
        return "défaut de gouvernance, de préparation et de mesure de redondance de la structure de crise."

    @staticmethod
    def _infer_crisis_type(text: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = set(normalized.split())
        for crisis_type, hints in _CRISIS_TYPE_HINTS.items():
            if hints & tokens:
                return crisis_type
        return ""

    @staticmethod
    def _types_compatible(query_type: str, case_type: str) -> bool:
        if not query_type or not case_type:
            return True

        normalized_query = query_type.lower().strip()
        normalized_case = case_type.lower().strip()

        if normalized_query == normalized_case:
            return True

        query_family = {
            "cyber": {"cyber", "cybersecurity", "attaque", "ransomware", "malware", "intrusion", "ddos", "sécurité", "security", "breach", "data", "fuite", "données"},
            "industriel": {"industriel", "industrielle", "industrie", "industrial", "facility", "production", "usine", "atelier", "safety", "manufacturing"},
            "financier": {"financier", "financière", "finance", "bank", "bancaire", "market", "payment", "cash", "banking", "credit"},
            "social": {"social", "societal", "sanitaire", "healthcare", "medical", "grève", "strike", "violence", "protest", "labor", "communaut"},
            "logistique": {"logistique", "supply", "chain", "transport", "shipping", "livraison", "distribution", "inventory", "fleet", "delivery", "shipment", "warehouse"},
        }

        return any(
            normalized_query in family and normalized_case in family
            for family in query_family.values()
        )
