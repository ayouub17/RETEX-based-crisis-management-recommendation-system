"""Command-line interface that turns similar RETEX into action proposals."""

from __future__ import annotations

import argparse
from typing import Any

from main import find_similar_retex
from recommendations.engine import RecommendationEngine


def _print_section(title: str, items: list[dict[str, Any]]) -> None:
    """Print one ranked recommendation section."""
    print(f"\n{title}")
    if not items:
        print("Aucun élément disponible dans les RETEX similaires.")
        return
    for rank, item in enumerate(items, start=1):
        print(
            f"{rank}. {item['text']} "
            f"(poids : {item['weighted_score']:.3f}, sources : {item['source_count']})"
        )


def main() -> None:
    """Search RETEX then display weighted actions and recommendations."""
    parser = argparse.ArgumentParser(description="Recommandations RETEX")
    parser.add_argument("query", help="Description de la nouvelle crise")
    parser.add_argument("--top-k", type=int, default=5, help="RETEX à analyser")
    parser.add_argument("--limit", type=int, default=10, help="Éléments à afficher")
    args = parser.parse_args()

    similar_cases = find_similar_retex(args.query, args.top_k)
    proposals = RecommendationEngine().recommend(similar_cases, args.limit)
    _print_section("=== Actions proposées ===", proposals["actions"])
    _print_section("=== Recommandations proposées ===", proposals["recommendations"])


if __name__ == "__main__":
    main()
