import pytest

from tsp import (
    CITY_COUNT,
    DISTANCES,
    KNOWN_OPTIMAL_LENGTH,
    KNOWN_OPTIMAL_TOUR,
    route_length,
)

CITIES = list(range(CITY_COUNT))


def test_known_optimal_tour_has_the_verified_length() -> None:
    assert sorted(KNOWN_OPTIMAL_TOUR) == CITIES
    assert route_length(KNOWN_OPTIMAL_TOUR) == pytest.approx(KNOWN_OPTIMAL_LENGTH, abs=1e-12)


def test_route_length_closes_the_cycle() -> None:
    route = (0, 1, 2)
    expected = DISTANCES[0][1] + DISTANCES[1][2] + DISTANCES[2][0]

    assert route_length(route) == pytest.approx(expected)
