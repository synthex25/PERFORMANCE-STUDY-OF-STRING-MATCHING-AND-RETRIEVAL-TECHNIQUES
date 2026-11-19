"""Tests for the Rabin-Karp module."""

import pytest

from algorithms.rabin_karp import rabin_karp_search


def test_search_finds_the_reference_example() -> None:
    result = rabin_karp_search("ABABDABACDABABCABAB", "ABABCABAB")

    assert result["positions"] == [10]
    assert result["hash_comparisons"] == 11
    assert result["character_verifications"] > 0
    assert result["execution_time"] >= 0


def test_search_finds_matches_at_beginning_middle_and_end() -> None:
    assert rabin_karp_search(
        "pattern in the pattern", "pattern"
    )["positions"] == [0, 15]
    assert rabin_karp_search("a needle here", "needle")["positions"] == [2]
    assert rabin_karp_search("ends here", "here")["positions"] == [5]


def test_search_handles_multiple_and_overlapping_matches() -> None:
    assert rabin_karp_search("AAAAA", "AAA")["positions"] == [0, 1, 2]
    assert rabin_karp_search("abababab", "abab")["positions"] == [0, 2, 4]


def test_hash_collision_is_verified() -> None:
    result = rabin_karp_search("cdab", "ab", base=256, prime=2)

    assert result["positions"] == [2]
    assert result["hash_comparisons"] == 3
    assert result["character_verifications"] >= 3


def test_search_handles_absent_empty_and_long_patterns() -> None:
    assert rabin_karp_search("hello", "xyz")["positions"] == []
    assert rabin_karp_search("hello", "")["positions"] == []
    assert rabin_karp_search("", "hello")["positions"] == []
    assert rabin_karp_search("hi", "hello")["positions"] == []


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(TypeError):
        rabin_karp_search("text", 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        rabin_karp_search("text", "t", prime=1)
