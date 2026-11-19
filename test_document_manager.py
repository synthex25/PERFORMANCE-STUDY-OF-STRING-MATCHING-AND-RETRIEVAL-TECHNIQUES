"""Tests for document loading and indexing preparation."""

from pathlib import Path

from algorithms.bloom_filter import BloomFilter
from algorithms.trie import Trie
from core.document_manager import DocumentManager


def test_load_documents_preserves_original_and_derives_terms(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.txt").write_text("Trie, Search!", encoding="utf-8")
    (tmp_path / "a.txt").write_text("KMP search.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("not loaded", encoding="utf-8")

    documents = DocumentManager(tmp_path).load_documents()

    assert [document.name for document in documents] == [
        "a.txt",
        "b.txt",
        "empty.txt",
    ]
    assert documents[0].original_text == "KMP search."
    assert documents[0].normalized_text == "kmp search."
    assert documents[0].tokens == ("kmp", "search")
    assert documents[0].unique_terms == frozenset({"kmp", "search"})


def test_missing_directory_returns_no_documents(tmp_path: Path) -> None:
    manager = DocumentManager(tmp_path / "missing")

    assert manager.load_documents() == []
    assert manager.all_unique_terms() == set()


def test_index_terms_populates_trie_and_bloom_filter(tmp_path: Path) -> None:
    (tmp_path / "topics.txt").write_text(
        "Algorithms and data structures", encoding="utf-8"
    )
    manager = DocumentManager(tmp_path)
    manager.load_documents()
    trie = Trie()
    bloom_filter = BloomFilter(capacity=10)

    assert manager.index_terms(trie, bloom_filter) == 4
    assert trie.search("algorithms")
    assert trie.search("DATA")
    assert bloom_filter.might_contain("structures")
