"""Generate embeddings for all RETEX cases (diagnostic command)."""

from __future__ import annotations

from config import Settings
from database.postgres import RetexRepository
from embeddings.embedder import RetexEmbedder
from embeddings.text_builder import build_retex_text


def main() -> None:
    """Read RETEX records and confirm embedding generation works."""
    settings = Settings.from_environment()
    cases = RetexRepository(settings).fetch_all()
    if not cases:
        raise RuntimeError("La table retex ne contient aucun cas.")
    vectors = RetexEmbedder(settings.embedding_model).encode(
        [build_retex_text(case) for case in cases]
    )
    print(f"{len(cases)} embeddings générés. Dimension : {vectors.shape[1]}.")


if __name__ == "__main__":
    main()
