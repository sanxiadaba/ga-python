"""Permutation-safe crossover operators."""

import random
from collections.abc import Sequence


def _ordered_child(
    segment_parent: Sequence[int],
    fill_parent: Sequence[int],
    start: int,
    end: int,
) -> tuple[int, ...]:
    child: list[int | None] = [None] * len(segment_parent)
    child[start : end + 1] = segment_parent[start : end + 1]
    used = set(segment_parent[start : end + 1])
    fill_values = [
        value for value in (*fill_parent[end + 1 :], *fill_parent[: end + 1]) if value not in used
    ]
    fill_positions = [*range(end + 1, len(child)), *range(0, start)]
    for position, value in zip(fill_positions, fill_values, strict=True):
        child[position] = value
    if any(value is None for value in child):  # pragma: no cover
        raise RuntimeError("ordered crossover produced an incomplete child")
    return tuple(value for value in child if value is not None)


def ordered_crossover(
    first: Sequence[int], second: Sequence[int], rng: random.Random
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Create two children with ordered crossover (OX)."""
    if len(first) != len(second):
        raise ValueError("parents must have the same length")
    if len(first) < 2:
        raise ValueError("parents must contain at least two genes")
    if set(first) != set(second) or len(set(first)) != len(first):
        raise ValueError("parents must be permutations of the same unique values")
    start, end = sorted(rng.sample(range(len(first)), 2))
    return (
        _ordered_child(first, second, start, end),
        _ordered_child(second, first, start, end),
    )
