"""Integrated Trie, Bloom Filter, KMP, and Rabin-Karp search workflow."""

from __future__ import annotations

from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

from algorithms.bloom_filter import BloomFilter
from algorithms.kmp import kmp_search
from algorithms.rabin_karp import rabin_karp_search
from algorithms.trie import Trie
from core.document_manager import DocumentManager
from core.text_processor import normalize_text, tokenize_words
from core.storage import Storage


def _get_context_bounds(
    text: str,
    position: int,
    pattern: str,
    window: int = 80,
) -> tuple[int, int, int, int]:
    """Calculate validated context and match bounds for one occurrence."""
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Text and pattern must be strings")
    if not isinstance(position, int) or not isinstance(window, int):
        raise TypeError("Position and window must be integers")
    if not pattern:
        raise ValueError("Pattern must not be empty")
    if window < 0:
        raise ValueError("Window must not be negative")

    match_start = position
    match_end = position + len(pattern)
    if match_start < 0 or match_end > len(text):
        raise ValueError("Match position is outside the text")
    matched_text = text[match_start:match_end]
    if matched_text.casefold() != pattern.casefold():
        raise ValueError(
            "Match position does not contain the requested pattern"
        )

    start = max(0, match_start - window)
    end = min(len(text), match_end + window)

    def is_word_character(character: str) -> bool:
        return character.isalnum() or character == "_"

    while start > 0 and is_word_character(text[start - 1]):
        start -= 1
    while end < len(text) and is_word_character(text[end]):
        end += 1
    return start, end, match_start, match_end


def get_match_context(
    text: str,
    position: int,
    pattern: str,
    window: int = 70,
) -> str:
    """Return a readable, word-boundary-safe context around one match.

    ``position`` is the zero-based character offset in ``text``. The initial
    window is expanded to include complete words on both sides, so the exact
    matched substring is never clipped and the document text is preserved.
    """
    start, end, _, _ = _get_context_bounds(text, position, pattern, window)
    return text[start:end]


def get_highlighted_match_context(
    text: str,
    position: int,
    pattern: str,
    window: int = 70,
) -> str:
    """Return escaped HTML highlighting only the exact matched substring."""
    start, end, match_start, match_end = _get_context_bounds(
        text, position, pattern, window
    )
    prefix = escape(text[start:match_start])
    matched_text = escape(text[match_start:match_end])
    suffix = escape(text[match_end:end])
    return (
        f'{prefix}<span class="search-highlight" '
        'style="background-color:#ffe08a;color:#111;padding:0 2px;">'
        f"{matched_text}</span>{suffix}"
    )


class SearchEngine:
    """Index documents once and provide autocomplete and exact text search."""

    def __init__(
        self,
        documents_directory: str | Path,
        bloom_capacity: int = 1_000,
        bloom_false_positive_probability: float = 0.01,
        storage_path: str | Path | None = None,
    ) -> None:
        self.storage = Storage(storage_path) if storage_path is not None else None
        self.document_manager = DocumentManager(documents_directory)
        self.documents = self.document_manager.load_documents()
        if self.storage is not None:
            for document in self.documents:
                self.storage.add_document(document.name, document.original_text)
            known = {document.name for document in self.documents}
            for row in self.storage.documents():
                if row["name"] not in known:
                    document = self.document_manager._create_record(row["name"], row["content"])
                    self.document_manager.documents.append(document)
                    known.add(document.name)
        self.trie = Trie()
        self.bloom_filter = BloomFilter(
            capacity=bloom_capacity,
            false_positive_probability=bloom_false_positive_probability,
        )
        self.indexed_term_count = self.document_manager.index_terms(
            self.trie,
            self.bloom_filter,
        )

    @property
    def document_count(self) -> int:
        """Return the number of successfully indexed documents."""
        return len(self.documents)

    def add_document(self, name: str, original_text: str) -> bool:
        """Add and index an in-memory document without rebuilding indexes."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Document name must be a non-empty string")
        if not isinstance(original_text, str):
            raise TypeError("Document text must be a string")
        name = Path(name).name
        if Path(name).suffix.casefold() not in {".txt", ".pdf", ".docx"}:
            raise ValueError("Only .txt, .pdf, and .docx documents are supported")
        if any(document.name == name for document in self.documents):
            return False
        if self.storage is not None and not self.storage.add_document(name, original_text):
            return False
        document = self.document_manager.add_document_text(name, original_text)
        if document is None:
            return False

        self.documents.append(document)
        for term in document.unique_terms:
            self.trie.insert(term)
            self.bloom_filter.add(term)
        self.indexed_term_count = len(self.document_manager.all_unique_terms())
        return True

    def autocomplete(self, prefix: str, limit: int = 10) -> list[str]:
        """Return Trie suggestions for a normalized prefix."""
        if not isinstance(prefix, str):
            raise TypeError("Prefix must be a string")
        return self.trie.autocomplete(normalize_text(prefix), limit)

    def search(self, query: str) -> dict[str, Any]:
        """Search all documents after preliminary Bloom-filter screening."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        search_started = perf_counter()
        normalized_query = normalize_text(query).strip()
        response: dict[str, Any] = {
            "query": normalized_query,
            "bloom_filter": {
                "status": "Definitely Not Present",
                "scope": "term-level preliminary check",
                "bits": self.bloom_filter.num_bits,
                "hash_functions": self.bloom_filter.num_hashes,
                "inserted_items": self.bloom_filter.item_count,
                "false_positive_rate": (
                    self.bloom_filter.calculate_false_positive_rate()
                ),
            },
            "search_executed": False,
            "kmp_executed": False,
            "rabin_karp_executed": False,
            "results": [],
            "algorithms_agree": True,
            "validation_error": None,
        }
        if not normalized_query:
            return response

        query_terms = tokenize_words(normalized_query)
        is_single_term_query = (
            len(query_terms) == 1 and query_terms[0] == normalized_query
        )
        is_prefix_candidate = self.trie.starts_with(normalized_query)
        bloom_might_contain = self.bloom_filter.might_contain(normalized_query)

        # The Bloom Filter contains words, not arbitrary substrings or phrases.
        # Only reject a single whole-word query when it is neither an indexed
        # term nor a Trie prefix; all other queries require exact verification.
        safe_term_rejection = (
            is_single_term_query
            and not bloom_might_contain
            and not is_prefix_candidate
        )
        if safe_term_rejection:
            if self.storage is not None:
                self.storage.log_search(normalized_query, "KMP + Rabin–Karp", 0, perf_counter() - search_started, len(self.documents))
            return response

        response["bloom_filter"]["status"] = "Possibly Present"
        response["search_executed"] = True
        response["kmp_executed"] = True
        response["rabin_karp_executed"] = True
        for document in self.documents:
            kmp_result = kmp_search(document.normalized_text, normalized_query)
            rabin_karp_result = rabin_karp_search(
                document.normalized_text,
                normalized_query,
            )
            kmp_positions = kmp_result["positions"]
            rabin_karp_positions = rabin_karp_result["positions"]
            if kmp_positions != rabin_karp_positions:
                response["algorithms_agree"] = False
                response["validation_error"] = (
                    f"Algorithm validation failed for {document.name}: "
                    f"KMP={kmp_positions}, Rabin-Karp={rabin_karp_positions}"
                )

            if not kmp_positions:
                continue
            contexts = [
                get_match_context(
                    document.original_text,
                    position,
                    normalized_query,
                )
                for position in kmp_positions
            ]
            highlighted_contexts = [
                get_highlighted_match_context(
                    document.original_text,
                    position,
                    normalized_query,
                )
                for position in kmp_positions
            ]
            response["results"].append(
                {
                    "document": document.name,
                    "kmp_positions": kmp_positions,
                    "rabin_karp_positions": rabin_karp_positions,
                    "kmp_comparisons": kmp_result["comparisons"],
                    "hash_comparisons": rabin_karp_result["hash_comparisons"],
                    "character_verifications": rabin_karp_result[
                        "character_verifications"
                    ],
                    "kmp_time": kmp_result["execution_time"],
                    "rabin_karp_time": rabin_karp_result["execution_time"],
                    "contexts": contexts,
                    "highlighted_contexts": highlighted_contexts,
                }
            )

        response["execution_time"] = perf_counter() - search_started
        response["match_count"] = sum(len(item["kmp_positions"]) for item in response["results"])
        if self.storage is not None:
            self.storage.log_search(normalized_query, "KMP + Rabin–Karp", response["match_count"], response["execution_time"], len(self.documents))
        return response
