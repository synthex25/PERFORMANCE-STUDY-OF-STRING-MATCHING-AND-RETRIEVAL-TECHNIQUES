"""SQLite persistence for document text and privacy-conscious search metadata."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class Storage:
    """Small SQLite repository; connections are short-lived per operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, content TEXT NOT NULL,
                    digest TEXT NOT NULL UNIQUE, category TEXT NOT NULL DEFAULT 'General',
                    tags TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(name);
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY, query TEXT NOT NULL, algorithm TEXT NOT NULL,
                    match_count INTEGER NOT NULL, elapsed_seconds REAL NOT NULL,
                    documents_searched INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_search_created ON search_history(created_at);
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def add_document(self, name: str, content: str, category: str = "General", tags: str = "") -> bool:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO documents(name,content,digest,category,tags,created_at) VALUES(?,?,?,?,?,?)",
                (Path(name).name, content, digest, category.strip() or "General", tags, datetime.now(timezone.utc).isoformat()),
            )
            return cursor.rowcount == 1

    def documents(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute("SELECT * FROM documents ORDER BY name COLLATE NOCASE, id"))

    def delete_document(self, document_id: int) -> bool:
        with self._connect() as connection:
            return connection.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount == 1

    def log_search(self, query: str, algorithm: str, matches: int, elapsed: float, documents: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO search_history(query,algorithm,match_count,elapsed_seconds,documents_searched,created_at) VALUES(?,?,?,?,?,?)",
                (query, algorithm, matches, elapsed, documents, datetime.now(timezone.utc).isoformat()),
            )

    def history(self, limit: int = 500) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute("SELECT * FROM search_history ORDER BY id DESC LIMIT ?", (max(1, min(limit, 5000)),)))

    def clear_history(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM search_history")
