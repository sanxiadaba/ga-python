import pytest

from ga import solve
from tsp import CITY_COUNT, KNOWN_OPTIMAL_LENGTH


@pytest.mark.parametrize("seed", range(5))
def test_default_run_reaches_the_known_optimum(seed: int) -> None:
    assert solve(seed=seed).distance == pytest.approx(KNOWN_OPTIMAL_LENGTH, abs=1e-12)


def test_same_seed_gives_the_same_result() -> None:
    assert solve(seed=7) == solve(seed=7)


def test_result_holds_a_permutation_and_the_two_histories() -> None:
    result = solve(population_size=10, generations=5)

    assert sorted(result.route) == list(range(CITY_COUNT))
    assert len(result.best_history) == 5
    assert len(result.mean_history) == 5
    assert all(
        later <= earlier
        for earlier, later in zip(result.best_history, result.best_history[1:], strict=False)
    )
    assert result.best_history[-1] == result.distance


def test_pure_genetic_algorithm_stays_behind_the_polished_one() -> None:
    pure = solve(population_size=80, generations=300, seed=0, polish=False)

    assert pure.distance > solve(seed=0).distance
