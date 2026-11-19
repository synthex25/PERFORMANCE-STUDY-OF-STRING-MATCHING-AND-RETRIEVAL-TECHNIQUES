"""Text normalization and tokenization helpers.

The processor keeps search text lightly normalized so pattern positions remain
useful while deriving lowercase word terms for the Trie and Bloom Filter.
"""

from __future__ import annotations

import re


WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize case without removing characters from searchable text."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return text.casefold()


def tokenize_words(text: str) -> list[str]:
    """Return normalized word tokens while ignoring punctuation."""
    normalized_text = normalize_text(text)
    return [match.group(0) for match in WORD_PATTERN.finditer(normalized_text)]


def unique_words(text: str) -> set[str]:
    """Return the unique normalized terms found in ``text``."""
    return set(tokenize_words(text))
