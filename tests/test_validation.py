import math

from constant import KNOWN_OPTIMAL_LENGTH, KNOWN_OPTIMAL_TOUR
from experiment import run_experiment
from fitness import tsp_fitness
from ga import GAConfig
from heuristics import best_nearest_neighbor_two_opt


def test_known_optimal_tour_has_independently_verified_length() -> None:
    assert math.isclose(
        tsp_fitness(KNOWN_OPTIMAL_TOUR),
        KNOWN_OPTIMAL_LENGTH,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_nearest_neighbor_plus_two_opt_reaches_known_optimum() -> None:
    route, distance = best_nearest_neighbor_two_opt()
    assert sorted(route) == list(range(25))
    assert math.isclose(distance, KNOWN_OPTIMAL_LENGTH, rel_tol=0, abs_tol=1e-12)


def test_multi_seed_experiment_is_reproducible_and_competitive() -> None:
    config = GAConfig(population_size=80, generations=300, seed=0)
    first = run_experiment(10, config)
    second = run_experiment(10, config)
    assert first == second
    assert first.best <= KNOWN_OPTIMAL_LENGTH * 1.000000001
    assert first.mean_gap_percent < 5
    assert first.optimal_runs >= 1
    assert first.runs_within_percent(5) >= 8
