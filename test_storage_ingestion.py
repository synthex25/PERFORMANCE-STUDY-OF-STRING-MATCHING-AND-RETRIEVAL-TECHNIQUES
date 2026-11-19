from pathlib import Path

import pytest

from core.ingestion import extract_text
from core.search_engine import SearchEngine
from core.storage import Storage


def test_sqlite_documents_and_history_persist(tmp_path: Path) -> None:
    path = tmp_path / "app.sqlite3"
    store = Storage(path)
    assert store.add_document("notes.txt", "Persistent retrieval text")
    assert not store.add_document("copy.txt", "Persistent retrieval text")
    store.log_search("retrieval", "KMP + Rabin–Karp", 1, 0.001, 1)
    assert Storage(path).documents()[0]["content"] == "Persistent retrieval text"
    assert Storage(path).history()[0]["query"] == "retrieval"


def test_search_engine_loads_persistent_documents_and_logs(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    Storage(database).add_document("stored.txt", "durable matching corpus")
    engine = SearchEngine(tmp_path / "empty", storage_path=database)
    result = engine.search("durable")
    assert result["match_count"] == 1
    assert Storage(database).history()[0]["query"] == "durable"


def test_ingestion_validates_type_size_and_empty_text() -> None:
    assert extract_text("a.txt", b"hello\n") == "hello\n"
    with pytest.raises(ValueError):
        extract_text("a.exe", b"hello")
    with pytest.raises(ValueError):
        extract_text("a.txt", b"\xff")
    with pytest.raises(ValueError):
        extract_text("a.txt", b" ")
