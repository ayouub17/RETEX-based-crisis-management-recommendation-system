"""Repository used to retrieve RETEX cases from PostgreSQL."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
from psycopg2.extras import RealDictCursor

from config import Settings


class RetexRepository:
    """Read-only gateway to the ``retex`` table."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @contextmanager
    def _connection(self) -> Generator[Any, None, None]:
        """Open and always close one PostgreSQL connection."""
        connection = psycopg2.connect(
            host=self._settings.postgres_host,
            port=self._settings.postgres_port,
            dbname=self._settings.postgres_db,
            user=self._settings.postgres_user,
            password=self._settings.postgres_password,
            connect_timeout=10,
        )
        try:
            yield connection
        finally:
            connection.close()

    def fetch_all(self) -> list[dict[str, Any]]:
        """Return all RETEX cases in a stable order for index reproducibility."""
        query = """
            SELECT id, source, title, crisis_type, organization, country,
                   report_date, summary, description, actions,
                   recommendations, url
            FROM retex
            ORDER BY id;
        """
        with self._connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]
