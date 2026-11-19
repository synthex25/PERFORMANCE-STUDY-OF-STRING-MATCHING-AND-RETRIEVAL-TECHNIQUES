"""Trie data structure for case-insensitive prefix search and autocomplete.

The trie stores one node per character. Inserting or looking up a word takes
O(L) time, where L is the word length. Autocomplete takes O(P + R), plus the
cost of visiting returned branches, where P is the prefix length and R is the
number of characters visited while collecting suggestions. The stored trie
uses O(N) space for the characters across all inserted words.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """A node containing character edges and a word-end marker."""

    children: dict[str, "TrieNode"] = field(default_factory=dict)
    is_end_of_word: bool = False


class Trie:
    """A case-insensitive trie supporting lookup and autocomplete."""

    def __init__(self) -> None:
        self.root = TrieNode()

    @staticmethod
    def _normalize(value: str) -> str:
        """Return the canonical representation used by the trie."""
        if not isinstance(value, str):
            raise TypeError("Trie values must be strings")
        return value.casefold()

    def insert(self, word: str) -> None:
        """Insert a word, ignoring duplicate insertions and empty strings."""
        normalized_word = self._normalize(word)
        if not normalized_word:
            return

        current_node = self.root
        for character in normalized_word:
            if character not in current_node.children:
                current_node.children[character] = TrieNode()
            current_node = current_node.children[character]
        current_node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Return whether the complete word exists in the trie."""
        normalized_word = self._normalize(word)
        if not normalized_word:
            return False

        current_node = self.root
        for character in normalized_word:
            if character not in current_node.children:
                return False
            current_node = current_node.children[character]
        return current_node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Return whether at least one inserted word has the given prefix."""
        normalized_prefix = self._normalize(prefix)
        if not normalized_prefix:
            return bool(self.root.children)

        current_node = self.root
        for character in normalized_prefix:
            if character not in current_node.children:
                return False
            current_node = current_node.children[character]
        return True

    def autocomplete(self, prefix: str, limit: int = 10) -> list[str]:
        """Return up to ``limit`` unique words beginning with ``prefix``."""
        if limit <= 0:
            return []

        normalized_prefix = self._normalize(prefix)
        current_node = self.root
        for character in normalized_prefix:
            if character not in current_node.children:
                return []
            current_node = current_node.children[character]

        suggestions: list[str] = []

        def collect(node: TrieNode, current_word: str) -> None:
            if len(suggestions) >= limit:
                return
            if node.is_end_of_word:
                suggestions.append(current_word)
            for character in sorted(node.children):
                collect(node.children[character], current_word + character)

        collect(current_node, normalized_prefix)
        return suggestions
