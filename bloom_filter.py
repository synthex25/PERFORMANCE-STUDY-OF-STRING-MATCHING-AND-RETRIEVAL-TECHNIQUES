"""Bloom filter for fast preliminary membership checks.

A Bloom filter stores membership bits set by multiple deterministic hash
functions. A cleared bit proves an item is definitely absent; set bits only
mean it is possibly present because of hash collisions. With ``M`` bits, ``K``
hashes, and ``N`` inserted items, the expected false-positive probability is
``(1 - exp(-K*N/M)) ** K``. Space usage is O(M), and each add or query uses
O(K) time.
"""

from __future__ import annotations

import math


class BloomFilter:
    """A bit-array Bloom filter with deterministic double hashing."""

    def __init__(
        self,
        capacity: int = 1_000,
        false_positive_probability: float = 0.01,
    ) -> None:
        """Create a filter sized for ``capacity`` expected insertions."""
        if not isinstance(capacity, int):
            raise TypeError("Capacity must be an integer")
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero")
        if not 0 < false_positive_probability < 1:
            raise ValueError(
                "False-positive probability must be between 0 and 1"
            )

        self.capacity = capacity
        self.false_positive_probability = false_positive_probability
        self.num_bits = max(
            1,
            math.ceil(
                -capacity
                * math.log(false_positive_probability)
                / (math.log(2) ** 2)
            ),
        )
        self.num_hashes = max(
            1,
            round(self.num_bits / capacity * math.log(2)),
        )
        self.bit_array = [False] * self.num_bits
        self.item_count = 0

    @staticmethod
    def _validate_item(item: str) -> None:
        if not isinstance(item, str):
            raise TypeError("Bloom filter items must be strings")

    @staticmethod
    def _hash_value(item: str, seed: int) -> int:
        """Return a deterministic 64-bit hash derived from item characters."""
        hash_value = (2_166_136_261 ^ seed) & 0xFFFFFFFFFFFFFFFF
        for character in item:
            hash_value ^= ord(character)
            hash_value = (hash_value * 16_777_619) & 0xFFFFFFFFFFFFFFFF
        return hash_value

    def _hash_indices(self, item: str) -> list[int]:
        """Return the bit positions selected by the hash functions."""
        first_hash = self._hash_value(item, 0)
        second_hash = self._hash_value(item, 1)
        return [
            (first_hash + index * second_hash) % self.num_bits
            for index in range(self.num_hashes)
        ]

    def add(self, item: str) -> None:
        """Add an item to the filter; this never removes existing bits."""
        self._validate_item(item)
        for bit_index in self._hash_indices(item):
            self.bit_array[bit_index] = True
        self.item_count += 1

    def might_contain(self, item: str) -> bool:
        """Return false for definite absence, or true for possible presence."""
        self._validate_item(item)
        return all(self.bit_array[index] for index in self._hash_indices(item))

    def calculate_false_positive_rate(self) -> float:
        """Return the theoretical false-positive rate for current items."""
        if self.item_count == 0:
            return 0.0
        exponent = -self.num_hashes * self.item_count / self.num_bits
        return (1 - math.exp(exponent)) ** self.num_hashes
