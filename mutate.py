"""Mutation operators."""

import random
from collections.abc import Sequence


def inversion_mutation(
    gene: Sequence[int], rng: random.Random, probability: float
) -> tuple[int, ...]:
    """Reverse an inclusive, non-empty segment with the requested probability."""
    result = tuple(gene)
    if len(result) < 2 or rng.random() >= probability:
        return result
    start, end = sorted(rng.sample(range(len(result)), 2))
    mutable = list(result)
    mutable[start : end + 1] = reversed(mutable[start : end + 1])
    return tuple(mutable)
