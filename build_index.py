"""Build the persistent semantic-search index from PostgreSQL RETEX records."""

from __future__ import annotations

from config import Settings
from database.postgres import RetexRepository
from embeddings.embedder import RetexEmbedder
from embeddings.text_builder import build_retex_text
from search.faiss_store import FaissStore


def main() -> None:
    """Fetch cases, embed their semantic text, and save FAISS artifacts."""
    settings = Settings.from_environment()
    cases = RetexRepository(settings).fetch_all()
    if not cases:
        raise RuntimeError("La table retex ne contient aucun cas à indexer.")

    texts = [build_retex_text(case) for case in cases]
    vectors = RetexEmbedder(settings.embedding_model).encode(texts)
    FaissStore.save(
        vectors=vectors,
        metadata=cases,
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
    )
    print(
        f"Index créé : {len(cases)} RETEX, dimension {vectors.shape[1]}.\n"
        f"FAISS : {settings.faiss_index_path}\n"
        f"Métadonnées : {settings.faiss_metadata_path}"
    )


if __name__ == "__main__":
    main()
