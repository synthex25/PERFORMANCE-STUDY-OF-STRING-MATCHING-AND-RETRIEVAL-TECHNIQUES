"""Small helpers for repeatable high-resolution timing."""

from __future__ import annotations

import time
from collections.abc import Callable
from statistics import median
from typing import TypeVar


Result = TypeVar("Result")


def median_runtime(
    operation: Callable[[], Result],
    repeats: int = 3,
) -> tuple[Result, float]:
    """Run an operation repeatedly and return its result and median time."""
    if not isinstance(repeats, int):
        raise TypeError("Repeats must be an integer")
    if repeats <= 0:
        raise ValueError("Repeats must be greater than zero")

    durations: list[float] = []
    last_result: Result | None = None
    for _ in range(repeats):
        start_time = time.perf_counter()
        last_result = operation()
        durations.append(time.perf_counter() - start_time)
    return last_result, float(median(durations))
