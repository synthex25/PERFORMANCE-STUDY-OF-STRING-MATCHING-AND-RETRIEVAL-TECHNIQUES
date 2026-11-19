"""Tests for the Bloom filter module."""

import pytest

from algorithms.bloom_filter import BloomFilter


def test_inserted_items_are_never_reported_absent() -> None:
    bloom_filter = BloomFilter(capacity=20, false_positive_probability=0.01)
    for item in ("algorithm", "trie", "database"):
        bloom_filter.add(item)

    assert bloom_filter.might_contain("algorithm")
    assert bloom_filter.might_contain("trie")
    assert bloom_filter.might_contain("database")


def test_definitely_absent_item_is_rejected() -> None:
    bloom_filter = BloomFilter(capacity=20, false_positive_probability=0.01)
    bloom_filter.add("algorithm")

    assert not bloom_filter.might_contain("quantumxyz")


def test_empty_filter_has_no_membership_and_zero_rate() -> None:
    bloom_filter = BloomFilter()

    assert not bloom_filter.might_contain("anything")
    assert bloom_filter.calculate_false_positive_rate() == 0.0


def test_multiple_items_and_false_positive_rate() -> None:
    bloom_filter = BloomFilter(capacity=10, false_positive_probability=0.05)
    for item in ("one", "two", "three", "four"):
        bloom_filter.add(item)

    assert bloom_filter.item_count == 4
    assert all(bloom_filter.might_contain(item) for item in ("one", "two"))
    assert 0 < bloom_filter.calculate_false_positive_rate() < 1


def test_configuration_and_item_validation() -> None:
    with pytest.raises(ValueError):
        BloomFilter(capacity=0)
    with pytest.raises(ValueError):
        BloomFilter(false_positive_probability=1)

    bloom_filter = BloomFilter()
    with pytest.raises(TypeError):
        bloom_filter.add(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bloom_filter.might_contain(123)  # type: ignore[arg-type]
