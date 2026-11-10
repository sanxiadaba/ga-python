import pytest

from ga_tsp.fitness import CITY_DISTANCE_MATRIX
from ga_tsp.heuristics import nearest_neighbor
from ga_tsp.utility import route_length


def test_route_length_rejects_empty_route() -> None:
    with pytest.raises(ValueError, match="at least one city"):
        route_length((), CITY_DISTANCE_MATRIX)


def test_nearest_neighbor_rejects_out_of_range_start() -> None:
    with pytest.raises(ValueError, match="outside"):
        nearest_neighbor(len(CITY_DISTANCE_MATRIX))


def test_route_length_closes_the_cycle() -> None:
    route = (0, 1, 2)
    expected = CITY_DISTANCE_MATRIX[0][1] + CITY_DISTANCE_MATRIX[1][2] + CITY_DISTANCE_MATRIX[2][0]
    assert route_length(route, CITY_DISTANCE_MATRIX) == pytest.approx(expected)
