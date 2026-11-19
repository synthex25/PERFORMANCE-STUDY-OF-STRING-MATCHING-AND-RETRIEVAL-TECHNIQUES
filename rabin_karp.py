"""Rabin-Karp pattern matching with a rolling hash.

The algorithm hashes the pattern and each text window of the same length. A
rolling update reuses the previous window hash in O(1) time. Matching hashes
are verified character by character because different strings can collide.
With a good modulus, average search time is O(N + M); the worst case is
O(N * M) when many windows collide. Preprocessing uses O(M) time and the
algorithm uses O(1) auxiliary space apart from the returned result.
"""

from __future__ import annotations

import time
from typing import TypedDict


class RabinKarpResult(TypedDict):
    """Structured output produced by :func:`rabin_karp_search`."""

    positions: list[int]
    hash_comparisons: int
    character_verifications: int
    execution_time: float


def _validate_parameters(base: int, prime: int) -> None:
    """Validate rolling-hash configuration values."""
    if not isinstance(base, int) or not isinstance(prime, int):
        raise TypeError("Base and prime must be integers")
    if base <= 1:
        raise ValueError("Base must be greater than 1")
    if prime <= 1:
        raise ValueError("Prime must be greater than 1")


def rabin_karp_search(
    text: str,
    pattern: str,
    base: int = 256,
    prime: int = 1_000_003,
) -> RabinKarpResult:
    """Find every occurrence of ``pattern`` using a rolling hash.

    Hash equality is only a preliminary match: every candidate window is
    verified character by character before its position is returned. Empty
    patterns and patterns longer than the text return no positions.
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Text and pattern must be strings")
    _validate_parameters(base, prime)

    start_time = time.perf_counter()
    positions: list[int] = []
    hash_comparisons = 0
    character_verifications = 0

    text_length = len(text)
    pattern_length = len(pattern)
    if not text or not pattern or pattern_length > text_length:
        return {
            "positions": positions,
            "hash_comparisons": hash_comparisons,
            "character_verifications": character_verifications,
            "execution_time": time.perf_counter() - start_time,
        }

    pattern_hash = 0
    window_hash = 0
    high_order = pow(base, pattern_length - 1, prime)

    for index in range(pattern_length):
        pattern_hash = (base * pattern_hash + ord(pattern[index])) % prime
        window_hash = (base * window_hash + ord(text[index])) % prime

    for window_start in range(text_length - pattern_length + 1):
        hash_comparisons += 1
        if window_hash == pattern_hash:
            matches = True
            for pattern_index in range(pattern_length):
                character_verifications += 1
                if (
                    text[window_start + pattern_index]
                    != pattern[pattern_index]
                ):
                    matches = False
                    break
            if matches:
                positions.append(window_start)

        if window_start < text_length - pattern_length:
            outgoing_value = ord(text[window_start])
            incoming_value = ord(text[window_start + pattern_length])
            window_hash = (
                base * (window_hash - outgoing_value * high_order)
                + incoming_value
            ) % prime

    return {
        "positions": positions,
        "hash_comparisons": hash_comparisons,
        "character_verifications": character_verifications,
        "execution_time": time.perf_counter() - start_time,
    }
