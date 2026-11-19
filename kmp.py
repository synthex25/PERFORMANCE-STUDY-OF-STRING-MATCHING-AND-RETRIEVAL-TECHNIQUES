"""Knuth-Morris-Pratt pattern matching.

KMP preprocesses a pattern into an LPS (longest proper prefix that is also a
suffix) table, then uses that table to avoid restarting comparisons. Building
the table takes O(M) time and O(M) space. Searching takes O(N) time after
preprocessing, for O(N + M) total time and O(M) auxiliary space.
"""

from __future__ import annotations

import time
from typing import TypedDict


class KMPResult(TypedDict):
    """Structured output produced by :func:`kmp_search`."""

    positions: list[int]
    comparisons: int
    execution_time: float


def build_lps(pattern: str) -> list[int]:
    """Build the LPS table for ``pattern`` in O(M) time.

    Each entry records the length of the longest proper prefix ending at that
    position that is also a suffix. The table lets the search reuse work when
    a mismatch occurs.
    """
    if not isinstance(pattern, str):
        raise TypeError("Pattern must be a string")

    lps = [0] * len(pattern)
    prefix_length = 0
    pattern_index = 1

    while pattern_index < len(pattern):
        if pattern[pattern_index] == pattern[prefix_length]:
            prefix_length += 1
            lps[pattern_index] = prefix_length
            pattern_index += 1
        elif prefix_length:
            prefix_length = lps[prefix_length - 1]
        else:
            lps[pattern_index] = 0
            pattern_index += 1

    return lps


def kmp_search(text: str, pattern: str) -> KMPResult:
    """Find every occurrence of ``pattern`` in ``text`` using KMP.

    The returned positions are zero-based and include overlapping matches.
    Empty patterns and patterns longer than the text return no positions. The
    comparison count measures character comparisons during the search phase.
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Text and pattern must be strings")

    start_time = time.perf_counter()
    positions: list[int] = []
    comparisons = 0

    if not text or not pattern or len(pattern) > len(text):
        return {
            "positions": positions,
            "comparisons": comparisons,
            "execution_time": time.perf_counter() - start_time,
        }

    lps = build_lps(pattern)
    text_index = 0
    pattern_index = 0

    while text_index < len(text):
        comparisons += 1
        if text[text_index] == pattern[pattern_index]:
            text_index += 1
            pattern_index += 1

            if pattern_index == len(pattern):
                positions.append(text_index - pattern_index)
                pattern_index = lps[pattern_index - 1]
        elif pattern_index:
            pattern_index = lps[pattern_index - 1]
        else:
            text_index += 1

    return {
        "positions": positions,
        "comparisons": comparisons,
        "execution_time": time.perf_counter() - start_time,
    }
