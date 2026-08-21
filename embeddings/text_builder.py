"""Build a consistent semantic representation for every RETEX case."""

from __future__ import annotations

from typing import Any


SEARCH_FIELDS = (
    ("Titre", "title"),
    ("Type de crise", "crisis_type"),
    ("Résumé", "summary"),
    ("Description", "description"),
    ("Actions menées", "actions"),
    ("Recommandations", "recommendations"),
)


def build_retex_text(retex: dict[str, Any]) -> str:
    """Concatenate meaningful RETEX fields, skipping null or empty values."""
    parts: list[str] = []
    for label, field in SEARCH_FIELDS:
        value = retex.get(field)
        if value is not None and str(value).strip():
            parts.append(f"{label} : {str(value).strip()}")

    if not parts:
        raise ValueError(f"Le RETEX {retex.get('id', '?')} ne contient aucun texte.")
    return "\n".join(parts)
