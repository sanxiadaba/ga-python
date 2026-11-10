"""Deterministic baselines used to judge the genetic algorithm."""

from collections.abc import Sequence

from .fitness import CITY_DISTANCE_MATRIX
from .utility import DistanceMatrix, route_length


def nearest_neighbor(
    start: int, distance_matrix: DistanceMatrix = CITY_DISTANCE_MATRIX
) -> tuple[int, ...]:
    if not 0 <= start < len(distance_matrix):
        raise ValueError("start is outside the distance matrix")
    route = [start]
    remaining = set(range(len(distance_matrix))) - {start}
    while remaining:
        next_city = min(
            remaining,
            key=lambda city: (distance_matrix[route[-1]][city], city),
        )
        route.append(next_city)
        remaining.remove(next_city)
    return tuple(route)


def two_opt(
    route: Sequence[int], distance_matrix: DistanceMatrix = CITY_DISTANCE_MATRIX
) -> tuple[tuple[int, ...], float]:
    """Improve a closed route until no 2-opt move reduces its length."""
    best = tuple(route)
    best_distance = route_length(best, distance_matrix)
    while True:
        candidate = best
        candidate_distance = best_distance
        for start in range(len(best) - 1):
            for end in range(start + 2, len(best) + 1):
                if start == 0 and end == len(best):
                    continue
                trial = (*best[:start], *reversed(best[start:end]), *best[end:])
                trial_distance = route_length(trial, distance_matrix)
                if trial_distance < candidate_distance - 1e-12:
                    candidate = trial
                    candidate_distance = trial_distance
        if candidate == best:
            return best, best_distance
        best = candidate
        best_distance = candidate_distance


def best_nearest_neighbor_two_opt(
    distance_matrix: DistanceMatrix = CITY_DISTANCE_MATRIX,
) -> tuple[tuple[int, ...], float]:
    candidates = [
        two_opt(nearest_neighbor(start, distance_matrix), distance_matrix)
        for start in range(len(distance_matrix))
    ]
    return min(candidates, key=lambda result: result[1])
