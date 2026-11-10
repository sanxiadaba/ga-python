import math

import pytest

from ga_tsp import GAConfig
from ga_tsp.constant import KNOWN_OPTIMAL_LENGTH, KNOWN_OPTIMAL_TOUR
from ga_tsp.experiment import ExperimentSummary, run_experiment
from ga_tsp.fitness import tsp_fitness
from ga_tsp.heuristics import best_nearest_neighbor_two_opt


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


def test_unseeded_experiment_reports_a_replayable_base_seed() -> None:
    config = GAConfig(population_size=10, generations=2, seed=None)
    first = run_experiment(2, config)
    assert 0 <= first.base_seed < 2**32
    replay = run_experiment(2, GAConfig(population_size=10, generations=2, seed=first.base_seed))
    assert replay.distances == first.distances
    assert replay.base_seed == first.base_seed


def test_experiment_rejects_zero_runs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_experiment(0, GAConfig(generations=1, population_size=10))


def test_experiment_summary_rejects_negative_percent() -> None:
    summary = ExperimentSummary((1.0,), 1.0)
    with pytest.raises(ValueError, match="negative"):
        summary.runs_within_percent(-1)
