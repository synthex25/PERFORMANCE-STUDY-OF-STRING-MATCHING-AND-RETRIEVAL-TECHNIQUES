"""Tests for the KMP module."""

import pytest

from algorithms.kmp import build_lps, kmp_search


def test_build_lps_returns_expected_prefix_table() -> None:
    assert build_lps("ABABCABAB") == [0, 0, 1, 2, 0, 1, 2, 3, 4]


def test_search_finds_the_reference_example() -> None:
    result = kmp_search("ABABDABACDABABCABAB", "ABABCABAB")

    assert result["positions"] == [10]
    assert result["comparisons"] > 0
    assert result["execution_time"] >= 0


def test_search_finds_matches_at_beginning_middle_and_end() -> None:
    assert kmp_search(
        "pattern in the pattern", "pattern"
    )["positions"] == [0, 15]
    assert kmp_search("a needle here", "needle")["positions"] == [2]
    assert kmp_search("ends here", "here")["positions"] == [5]


def test_search_handles_multiple_and_overlapping_matches() -> None:
    assert kmp_search("AAAAA", "AAA")["positions"] == [0, 1, 2]
    assert kmp_search("abababab", "abab")["positions"] == [0, 2, 4]


def test_search_handles_absent_empty_and_long_patterns() -> None:
    assert kmp_search("hello", "xyz")["positions"] == []
    assert kmp_search("hello", "")["positions"] == []
    assert kmp_search("", "hello")["positions"] == []
    assert kmp_search("hi", "hello")["positions"] == []


def test_invalid_inputs_raise_type_error() -> None:
    with pytest.raises(TypeError):
        build_lps(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        kmp_search("text", 123)  # type: ignore[arg-type]
