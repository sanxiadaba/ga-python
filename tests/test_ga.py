import random

import pytest

from ga import inversion_mutation, ordered_crossover, random_route, solve
from tsp import CITY_COUNT, KNOWN_OPTIMAL_LENGTH

CITIES = list(range(CITY_COUNT))


def test_random_route_is_a_permutation() -> None:
    assert sorted(random_route(random.Random(0))) == CITIES


@pytest.mark.parametrize("seed", range(20))
def test_crossover_children_are_permutations(seed: int) -> None:
    rng = random.Random(seed)
    children = ordered_crossover(random_route(rng), random_route(rng), rng)

    assert all(sorted(child) == CITIES for child in children)


def test_mutation_reverses_one_segment() -> None:
    class FixedRandom:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def sample(_population: range, _count: int) -> list[int]:
            return [1, 4]

    route = tuple(range(6))

    assert inversion_mutation(route, FixedRandom(), 1.0) == (0, 4, 3, 2, 1, 5)  # type: ignore[arg-type]
    assert inversion_mutation(route, FixedRandom(), 0.0) == route  # type: ignore[arg-type]


def test_same_seed_gives_the_same_result() -> None:
    assert solve(seed=7) == solve(seed=7)


def test_result_holds_a_permutation_and_the_two_histories() -> None:
    result = solve(population_size=20, generations=5)

    assert sorted(result.route) == CITIES
    assert len(result.best_history) == 5
    assert len(result.mean_history) == 5
    assert all(
        later <= earlier
        for earlier, later in zip(result.best_history, result.best_history[1:], strict=False)
    )
    assert result.best_history[-1] == result.distance


def test_default_run_lands_close_to_the_known_optimum() -> None:
    gap = (solve(seed=0).distance / KNOWN_OPTIMAL_LENGTH - 1) * 100

    assert 0 <= gap < 10
