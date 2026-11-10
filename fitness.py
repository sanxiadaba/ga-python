"""Fitness function for the bundled TSP problem."""

from collections.abc import Sequence
from functools import cache

from constant import CITY_POS
from utility import DistanceMatrix, build_distance_matrix, route_length

CITY_DISTANCE_MATRIX = build_distance_matrix(CITY_POS)


@cache
def _required_cities(city_count: int) -> frozenset[int]:
    """Return the exact set a valid permutation must contain, built once per size."""
    return frozenset(range(city_count))


def validate_tour(gene: Sequence[int], city_count: int) -> None:
    """Reject incomplete routes and routes containing duplicate cities."""
    if len(gene) != city_count or set(gene) != _required_cities(city_count):
        raise ValueError(f"gene must be a permutation of range({city_count})")


def tsp_fitness(
    gene: Sequence[int], distance_matrix: DistanceMatrix = CITY_DISTANCE_MATRIX
) -> float:
    """Return the total distance of a closed TSP route."""
    validate_tour(gene, len(distance_matrix))
    return route_length(gene, distance_matrix)
