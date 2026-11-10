"""Gene factories."""

import random


def generate_permutation(length: int, rng: random.Random) -> tuple[int, ...]:
    """Generate a random permutation using the caller-owned RNG."""
    if length < 2:
        raise ValueError("permutation length must be at least 2")
    gene = list(range(length))
    rng.shuffle(gene)
    return tuple(gene)
