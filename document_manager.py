"""Load text documents and prepare them for search indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.text_processor import normalize_text, tokenize_words, unique_words

if TYPE_CHECKING:
    from algorithms.bloom_filter import BloomFilter
    from algorithms.trie import Trie


@dataclass(frozen=True)
class DocumentRecord:
    """A document with both original content and derived search terms."""

    name: str
    path: Path
    original_text: str
    normalized_text: str
    tokens: tuple[str, ...]
    unique_terms: frozenset[str]


class DocumentManager:
    """Read and process ``.txt`` documents from a configured directory."""

    def __init__(self, documents_directory: str | Path) -> None:
        self.documents_directory = Path(documents_directory)
        self.documents: list[DocumentRecord] = []

    @staticmethod
    def _process_file(file_path: Path) -> DocumentRecord | None:
        """Read one file, returning ``None`` for unreadable documents."""
        try:
            original_text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

        return DocumentRecord(
            name=file_path.name,
            path=file_path,
            original_text=original_text,
            normalized_text=normalize_text(original_text),
            tokens=tuple(tokenize_words(original_text)),
            unique_terms=frozenset(unique_words(original_text)),
        )

    @staticmethod
    def _create_record(name: str, original_text: str) -> DocumentRecord:
        """Create a processed record from text supplied by the application."""
        return DocumentRecord(
            name=name,
            path=Path(name).name,
            original_text=original_text,
            normalized_text=normalize_text(original_text),
            tokens=tuple(tokenize_words(original_text)),
            unique_terms=frozenset(unique_words(original_text)),
        )

    def add_document_text(
        self,
        name: str,
        original_text: str,
    ) -> DocumentRecord | None:
        """Add an in-memory text document unless its name already exists."""
        if not isinstance(name, str) or not name:
            raise ValueError("Document name must be a non-empty string")
        if not isinstance(original_text, str):
            raise TypeError("Document text must be a string")
        safe_name = Path(name).name
        if Path(safe_name).suffix.casefold() not in {".txt", ".pdf", ".docx"}:
            raise ValueError("Only .txt, .pdf, and .docx documents are supported")
        if any(document.name == safe_name for document in self.documents):
            return None

        document = self._create_record(safe_name, original_text)
        self.documents.append(document)
        return document

    def load_documents(self) -> list[DocumentRecord]:
        """Load all readable text files in name order.

        Missing directories and directories without text files produce an
        empty list rather than raising an error for normal user input.
        """
        self.documents = []
        if not self.documents_directory.is_dir():
            return self.documents

        for file_path in sorted(self.documents_directory.glob("*.txt")):
            document = self._process_file(file_path)
            if document is not None:
                self.documents.append(document)
        return list(self.documents)

    def all_unique_terms(self) -> set[str]:
        """Return the union of searchable terms across loaded documents."""
        terms: set[str] = set()
        for document in self.documents:
            terms.update(document.unique_terms)
        return terms

    def index_terms(self, trie: Trie, bloom_filter: BloomFilter) -> int:
        """Add all unique document terms to the provided indexes.

        Return the number of distinct terms indexed. The caller owns the
        index instances so they can be reused by the search engine.
        """
        terms = self.all_unique_terms()
        for term in terms:
            trie.insert(term)
            bloom_filter.add(term)
        return len(terms)
