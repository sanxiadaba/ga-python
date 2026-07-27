"""Geometry helpers shared by the solver and validation baselines."""

from collections.abc import Sequence
from math import dist

Point = tuple[float, float]
DistanceMatrix = tuple[tuple[float, ...], ...]


def build_distance_matrix(points: Sequence[Point]) -> DistanceMatrix:
    """Precompute all pairwise Euclidean distances."""
    return tuple(tuple(dist(first, second) for second in points) for first in points)


def route_length(route: Sequence[int], distance_matrix: DistanceMatrix) -> float:
    """Return the length of a closed route."""
    if not route:
        raise ValueError("route must contain at least one city")
    return sum(
        distance_matrix[route[index]][route[(index + 1) % len(route)]]
        for index in range(len(route))
    )
