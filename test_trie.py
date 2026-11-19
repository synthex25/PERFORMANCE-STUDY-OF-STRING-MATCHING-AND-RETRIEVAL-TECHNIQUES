"""Tests for the Trie module."""

import pytest

from algorithms.trie import Trie


def build_trie() -> Trie:
    trie = Trie()
    for word in ("algorithm", "algorithms", "algorithmic", "data", "database"):
        trie.insert(word)
    return trie


def test_insert_and_exact_search_are_case_insensitive() -> None:
    trie = Trie()
    trie.insert("Algorithm")

    assert trie.search("algorithm")
    assert trie.search("ALGORITHM")
    assert not trie.search("algo")
    assert not trie.search("algorithms")


def test_prefix_search_traverses_existing_and_missing_paths() -> None:
    trie = build_trie()

    assert trie.starts_with("alg")
    assert trie.starts_with("DATA")
    assert not trie.starts_with("quantum")


def test_autocomplete_returns_sorted_unique_suggestions() -> None:
    trie = build_trie()
    trie.insert("Algorithm")

    assert trie.autocomplete("alg") == [
        "algorithm",
        "algorithmic",
        "algorithms",
    ]
    assert trie.autocomplete("data") == ["data", "database"]


def test_autocomplete_obeys_limit_and_handles_missing_prefix() -> None:
    trie = build_trie()

    assert trie.autocomplete("alg", limit=2) == ["algorithm", "algorithmic"]
    assert trie.autocomplete("alg", limit=0) == []
    assert trie.autocomplete("missing") == []


def test_empty_inputs_are_safe() -> None:
    trie = Trie()

    assert not trie.search("")
    assert not trie.starts_with("")
    assert trie.autocomplete("") == []

    trie.insert("")
    assert trie.autocomplete("") == []


def test_non_string_input_is_rejected() -> None:
    trie = Trie()

    with pytest.raises(TypeError):
        trie.insert(123)  # type: ignore[arg-type]
