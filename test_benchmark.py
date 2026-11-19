"""Tests for measured benchmark helpers."""

import pytest

from core.benchmark import (
    benchmark_bloom_filter,
    benchmark_pattern_sizes,
    benchmark_text_sizes,
    build_benchmark_text,
)


def test_build_benchmark_text_has_requested_size() -> None:
    text = build_benchmark_text(128, "algorithm")

    assert len(text) == 128
    assert text.startswith("algorithm")


def test_text_size_benchmark_returns_measured_comparisons() -> None:
    results = benchmark_text_sizes([64, 128], pattern="algorithm", repeats=1)

    assert [row["text_size"] for row in results] == [64, 128]
    assert all(row["kmp_time"] >= 0 for row in results)
    assert all(row["rabin_karp_time"] >= 0 for row in results)
    assert all(row["positions_agree"] for row in results)


def test_pattern_size_benchmark_returns_all_requested_sizes() -> None:
    results = benchmark_pattern_sizes(
        text_size=256,
        pattern_sizes=[3, 5, 10],
        repeats=1,
    )

    assert [row["pattern_size"] for row in results] == [3, 5, 10]
    assert all(row["positions_agree"] for row in results)


def test_bloom_benchmark_counts_rejected_and_passed_queries() -> None:
    result = benchmark_bloom_filter(
        "algorithm data structures",
        ["algorithm", "missingxyz", "structures"],
        repeats=1,
    )

    assert result["query_count"] == 3
    assert result["rejected_by_bloom"] >= 1
    assert result["passed_to_pattern_matching"] >= 2
    assert result["without_bloom_kmp_calls"] == 3
    assert result["with_bloom_kmp_calls"] == result[
        "passed_to_pattern_matching"
    ]
    assert result["without_bloom_time"] >= 0
    assert result["with_bloom_time"] >= 0


def test_bloom_benchmark_does_not_reject_a_phrase() -> None:
    result = benchmark_bloom_filter(
        "search algorithms improve retrieval",
        ["search algorithms"],
        repeats=1,
    )

    assert result["rejected_by_bloom"] == 0
    assert result["passed_to_pattern_matching"] == 1
    assert result["false_positive_count"] == 0


def test_benchmark_rejects_invalid_sizes_and_queries() -> None:
    with pytest.raises(ValueError):
        benchmark_text_sizes([0], repeats=1)
    with pytest.raises(ValueError):
        benchmark_bloom_filter("text", [""])
