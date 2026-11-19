"""Measured performance experiments for the project algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from algorithms.bloom_filter import BloomFilter
from algorithms.kmp import kmp_search
from algorithms.rabin_karp import rabin_karp_search
from core.text_processor import normalize_text, tokenize_words
from utils.timing import median_runtime


DEFAULT_TEXT_SIZES = (
    1_024,
    5_120,
    10_240,
    51_200,
    102_400,
    512_000,
    1_048_576,
)
DEFAULT_PATTERN_SIZES = (3, 5, 10, 20, 50)
FILLER_TEXT = "algorithms data structures search indexing performance "


def _validate_sizes(sizes: Iterable[int], name: str) -> list[int]:
    """Validate and materialize positive benchmark sizes."""
    validated = list(sizes)
    if not validated or any(
        not isinstance(size, int) or size <= 0 for size in validated
    ):
        raise ValueError(f"{name} must contain positive integers")
    return validated


def build_benchmark_text(size: int, pattern: str) -> str:
    """Build deterministic text of exactly ``size`` characters."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Text size must be a positive integer")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("Pattern must be a non-empty string")
    repeated_text = (pattern + " " + FILLER_TEXT) * (
        size // len(FILLER_TEXT) + 2
    )
    return repeated_text[:size]


def _fit_source_text(size: int, source_text: str) -> str:
    """Repeat a real source corpus to an exact benchmark length."""
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("Source text must be a non-empty string")
    return (source_text * (size // len(source_text) + 1))[:size]


def _benchmark_pair(
    text: str,
    pattern: str,
    repeats: int,
) -> dict[str, object]:
    """Measure KMP and Rabin-Karp on one text-pattern pair."""
    kmp_result, kmp_time = median_runtime(
        lambda: kmp_search(text, pattern), repeats
    )
    rabin_result, rabin_time = median_runtime(
        lambda: rabin_karp_search(text, pattern), repeats
    )
    return {
        "text_size": len(text),
        "pattern_size": len(pattern),
        "match_count": len(kmp_result["positions"]),
        "kmp_time": kmp_time,
        "rabin_karp_time": rabin_time,
        "kmp_comparisons": kmp_result["comparisons"],
        "hash_comparisons": rabin_result["hash_comparisons"],
        "character_verifications": rabin_result["character_verifications"],
        "positions_agree": (
            kmp_result["positions"] == rabin_result["positions"]
        ),
    }


def benchmark_text_sizes(
    text_sizes: Iterable[int] = DEFAULT_TEXT_SIZES,
    pattern: str = "algorithm",
    repeats: int = 3,
    source_text: str | None = None,
) -> list[dict[str, object]]:
    """Compare both algorithms while increasing text size."""
    sizes = _validate_sizes(text_sizes, "Text sizes")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("Pattern must be a non-empty string")
    return [
        _benchmark_pair(
            build_benchmark_text(size, pattern)
            if source_text is None
            else _fit_source_text(size, source_text),
            pattern,
            repeats,
        )
        for size in sizes
    ]


def benchmark_pattern_sizes(
    text_size: int = 102_400,
    pattern_sizes: Iterable[int] = DEFAULT_PATTERN_SIZES,
    repeats: int = 3,
    source_text: str | None = None,
) -> list[dict[str, object]]:
    """Compare both algorithms while increasing pattern size."""
    sizes = _validate_sizes(pattern_sizes, "Pattern sizes")
    patterns = [
        ("algorithmicpattern" * (size // 19 + 1))[:size]
        for size in sizes
    ]
    return [
        _benchmark_pair(
            build_benchmark_text(text_size, pattern)
            if source_text is None
            else _fit_source_text(text_size, source_text),
            pattern,
            repeats,
        )
        for pattern in patterns
    ]


def benchmark_bloom_filter(
    text: str,
    queries: Iterable[str],
    false_positive_probability: float = 0.01,
    repeats: int = 3,
) -> dict[str, object]:
    """Compare KMP query time with and without preliminary Bloom filtering."""
    if not isinstance(text, str) or not text:
        raise ValueError("Text must be a non-empty string")
    query_list = list(queries)
    if any(not isinstance(query, str) or not query for query in query_list):
        raise ValueError("Queries must be non-empty strings")

    normalized_text = normalize_text(text)
    normalized_queries = [normalize_text(query) for query in query_list]
    terms = set(tokenize_words(text))
    bloom_filter = BloomFilter(
        capacity=max(1, len(terms)),
        false_positive_probability=false_positive_probability,
    )
    for term in terms:
        bloom_filter.add(term)

    actual_presence = {
        query: bool(kmp_search(normalized_text, query)["positions"])
        for query in set(normalized_queries)
    }

    def without_filter() -> int:
        calls = 0
        for query in normalized_queries:
            kmp_search(normalized_text, query)
            calls += 1
        return calls

    def with_filter() -> int:
        calls = 0
        for query in normalized_queries:
            query_terms = tokenize_words(query)
            is_single_term = len(query_terms) == 1 and query_terms[0] == query
            should_run = (
                not is_single_term or bloom_filter.might_contain(query)
            )
            if should_run:
                kmp_search(normalized_text, query)
                calls += 1
        return calls

    _, without_time = median_runtime(without_filter, repeats)
    _, with_time = median_runtime(with_filter, repeats)
    passed_queries = 0
    for query in normalized_queries:
        query_terms = tokenize_words(query)
        is_single_term = len(query_terms) == 1 and query_terms[0] == query
        if not is_single_term or bloom_filter.might_contain(query):
            passed_queries += 1
    rejected_queries = len(normalized_queries) - passed_queries
    false_positive_count = sum(
        len(tokenize_words(query)) == 1
        and tokenize_words(query)[0] == query
        and bloom_filter.might_contain(query)
        and not actual_presence[query]
        for query in normalized_queries
    )
    return {
        "query_count": len(query_list),
        "rejected_by_bloom": rejected_queries,
        "passed_to_pattern_matching": passed_queries,
        "false_positive_count": false_positive_count,
        "without_bloom_time": without_time,
        "with_bloom_time": with_time,
        "without_bloom_kmp_calls": len(query_list),
        "with_bloom_kmp_calls": passed_queries,
    }
