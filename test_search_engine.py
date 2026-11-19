"""Tests for the SearchEngine integration."""

from pathlib import Path

from core.search_engine import (
    SearchEngine,
    get_highlighted_match_context,
    get_match_context,
)
from core import search_engine as search_engine_module


def make_engine(tmp_path: Path) -> SearchEngine:
    (tmp_path / "algorithms.txt").write_text(
        "Algorithm methods such as KMP and Rabin-Karp support "
        "efficient search.",
        encoding="utf-8",
    )
    (tmp_path / "structures.txt").write_text(
        "A trie supports prefix search and autocomplete.", encoding="utf-8"
    )
    return SearchEngine(tmp_path, bloom_capacity=20)


def test_engine_indexes_documents_and_supports_autocomplete(
    tmp_path: Path,
) -> None:
    engine = make_engine(tmp_path)

    assert engine.document_count == 2
    assert engine.indexed_term_count > 0
    assert engine.autocomplete("AUT") == ["autocomplete"]
    assert "algorithm" in engine.autocomplete("algo")


def test_existing_query_runs_both_algorithms_and_agrees(
    tmp_path: Path,
) -> None:
    engine = make_engine(tmp_path)

    result = engine.search("search")

    assert result["bloom_filter"]["status"] == "Possibly Present"
    assert result["search_executed"] is True
    assert result["kmp_executed"] is True
    assert result["rabin_karp_executed"] is True
    assert result["validation_error"] is None
    assert result["algorithms_agree"] is True
    assert len(result["results"]) == 2
    for document_result in result["results"]:
        assert document_result["kmp_positions"] == document_result[
            "rabin_karp_positions"
        ]
        assert document_result["contexts"]


def test_bloom_filter_rejects_definitely_absent_query(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    result = engine.search("quantumxyz")

    assert result["bloom_filter"]["status"] == "Definitely Not Present"
    assert result["search_executed"] is False
    assert result["results"] == []


def test_bloom_false_positive_still_returns_no_matches(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.bloom_filter.add("not-in-documents")

    result = engine.search("not-in-documents")

    assert result["bloom_filter"]["status"] == "Possibly Present"
    assert result["search_executed"] is True
    assert result["results"] == []


def test_empty_and_invalid_queries_are_safe(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    assert engine.search("")["search_executed"] is False
    assert engine.search("   ")["results"] == []


def test_phrase_search_is_not_rejected_by_word_level_bloom_filter(
    tmp_path: Path,
) -> None:
    engine = make_engine(tmp_path)

    result = engine.search("efficient search")

    assert result["bloom_filter"]["status"] == "Possibly Present"
    assert result["search_executed"] is True
    assert result["results"][0]["kmp_positions"] == result["results"][0][
        "rabin_karp_positions"
    ]


def test_algorithm_disagreement_is_exposed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine = make_engine(tmp_path)

    def mismatching_search(text: str, pattern: str) -> dict[str, object]:
        return {
            "positions": [999],
            "hash_comparisons": 1,
            "character_verifications": 1,
            "execution_time": 0.0,
        }

    monkeypatch.setattr(
        search_engine_module,
        "rabin_karp_search",
        mismatching_search,
    )
    result = engine.search("search")

    assert result["algorithms_agree"] is False
    assert "KMP=" in result["validation_error"]
    assert "Rabin-Karp=" in result["validation_error"]


def test_uploaded_document_is_added_and_searchable(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    original_count = engine.document_count
    original_terms = engine.indexed_term_count

    assert engine.add_document(
        "uploaded_notes.txt",
        "Quantum indexing enables uploaded document retrieval.",
    )
    assert engine.document_count == original_count + 1
    assert engine.indexed_term_count > original_terms

    result = engine.search("quantum")
    uploaded_result = next(
        item for item in result["results"]
        if item["document"] == "uploaded_notes.txt"
    )
    assert uploaded_result["kmp_positions"] == uploaded_result[
        "rabin_karp_positions"
    ]
    assert uploaded_result["contexts"]


def test_duplicate_uploaded_document_name_is_ignored(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    assert engine.add_document("notes.txt", "first content")
    assert not engine.add_document("notes.txt", "replacement content")


def test_match_context_preserves_complete_word_boundaries() -> None:
    text = (
        "Algorithms provide systematic methods for solving computational "
        "problems."
    )
    position = text.casefold().index("algo")

    context = get_match_context(text, position, "algo", window=8)

    assert context.casefold().count("algo") == 1
    assert context.startswith("Algorithms")
    assert context.endswith("provide")


def test_highlighting_marks_only_the_exact_case_insensitive_match() -> None:
    text = "Algorithms provide methods."
    position = text.casefold().index("algo")

    highlighted = get_highlighted_match_context(text, position, "algo")

    assert '<span class="search-highlight"' in highlighted
    assert ">Algo</span>" in highlighted
    assert ">rithms</span>" not in highlighted
    assert ">and</span>" not in highlighted


def test_each_position_gets_its_own_context_and_highlight() -> None:
    text = "algo first, then another algo near the end."
    positions = [0, text.index("algo", 1)]

    contexts = [
        get_match_context(text, position, "algo", window=8)
        for position in positions
    ]
    highlighted = [
        get_highlighted_match_context(text, position, "algo", window=8)
        for position in positions
    ]

    assert len(contexts) == len(positions)
    assert contexts[0] != contexts[1]
    assert all(context.casefold().count("algo") >= 1 for context in contexts)
    assert all(
        context.count('class="search-highlight"') == 1
        for context in highlighted
    )


def test_highlighting_preserves_spaces_around_the_exact_match() -> None:
    text = "An algorithmic design explains how an algorithm transforms input."
    first_position = text.casefold().index("algorithm")
    second_position = text.casefold().index("algorithm", first_position + 1)

    first = get_highlighted_match_context(
        text,
        first_position,
        "algorithm",
        window=8,
    )
    second = get_highlighted_match_context(
        text,
        second_position,
        "algorithm",
        window=8,
    )

    assert "An <span" in first
    assert "</span>ic design" in first
    assert "an <span" in second
    assert "</span> transforms" in second

# End of SearchEngine tests.