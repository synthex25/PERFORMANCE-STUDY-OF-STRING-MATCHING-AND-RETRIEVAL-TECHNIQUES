"""Tests for text normalization and tokenization."""

import pytest

from core.text_processor import normalize_text, tokenize_words, unique_words


def test_normalize_text_preserves_searchable_characters() -> None:
    assert normalize_text("KMP: DATA Structures!") == "kmp: data structures!"


def test_tokenize_words_normalizes_and_removes_punctuation() -> None:
    text = "KMP, Trie-based search; KMP works."

    assert tokenize_words(text) == [
        "kmp",
        "trie",
        "based",
        "search",
        "kmp",
        "works",
    ]
    assert unique_words(text) == {"kmp", "trie", "based", "search", "works"}


def test_text_processor_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        tokenize_words(123)  # type: ignore[arg-type]
