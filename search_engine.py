"""Command-line interface for semantic RETEX search."""

from __future__ import annotations

import argparse
from typing import Any

from config import Settings
from embeddings.embedder import RetexEmbedder
from search.engine import SemanticSearchEngine
from search.faiss_store import FaissStore


def format_result(result: dict[str, Any]) -> str:
    """Produce a readable, recommendation-oriented search result."""
    return (
        f"[{result['similarity_score']:.3f}] {result.get('title') or 'Sans titre'}\n"
        f"Organisation : {result.get('organization') or 'Non renseignée'}\n"
        f"Actions : {result.get('actions') or 'Non renseignées'}\n"
        f"Recommandations : {result.get('recommendations') or 'Non renseignées'}"
    )


def main() -> None:
    """Run a Top-K search from a crisis description passed on the CLI."""
    parser = argparse.ArgumentParser(description="Recherche sémantique RETEX")
    parser.add_argument("query", help="Description de la nouvelle crise")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de résultats")
    args = parser.parse_args()

    settings = Settings.from_environment()
    index, metadata = FaissStore.load(
        settings.faiss_index_path, settings.faiss_metadata_path
    )
    engine = SemanticSearchEngine(
        index, metadata, RetexEmbedder(settings.embedding_model)
    )
    for rank, result in enumerate(engine.search(args.query, args.top_k), start=1):
        print(f"\n--- Résultat {rank} ---\n{format_result(result)}")


if __name__ == "__main__":
    main()
