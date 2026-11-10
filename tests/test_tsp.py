import random

import pytest

from tsp import (
    CITY_COUNT,
    DISTANCES,
    KNOWN_OPTIMAL_LENGTH,
    KNOWN_OPTIMAL_TOUR,
    route_length,
    two_opt,
)

CITIES = list(range(CITY_COUNT))


def test_known_tour_has_the_verified_length() -> None:
    assert route_length(KNOWN_OPTIMAL_TOUR) == pytest.approx(KNOWN_OPTIMAL_LENGTH, abs=1e-12)


def test_route_length_closes_the_cycle() -> None:
    route = (0, 1, 2)
    expected = DISTANCES[0][1] + DISTANCES[1][2] + DISTANCES[2][0]

    assert route_length(route) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(10))
def test_two_opt_never_makes_a_route_worse(seed: int) -> None:
    route = tuple(random.Random(seed).sample(CITIES, CITY_COUNT))
    improved = two_opt(route)

    assert sorted(improved) == CITIES
    assert route_length(improved) <= route_length(route)


def test_two_opt_leaves_the_optimal_tour_alone() -> None:
    assert route_length(two_opt(KNOWN_OPTIMAL_TOUR)) == pytest.approx(
        KNOWN_OPTIMAL_LENGTH, abs=1e-12
    )
