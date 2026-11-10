import math
import random
import statistics

import pytest

from ga_tsp import KNOWN_OPTIMAL_LENGTH, GAConfig, TspGA
from ga_tsp.experiment import ExperimentSummary, run_experiment
from ga_tsp.operators import generate_permutation, inversion_mutation, ordered_crossover
from ga_tsp.tsp import (
    CITY_COUNT,
    CITY_DISTANCES,
    KNOWN_OPTIMAL_TOUR,
    baseline_tour,
    nearest_neighbor_tour,
    route_length,
    tsp_fitness,
    two_opt,
    validate_tour,
)

CITIES = list(range(CITY_COUNT))


def _random_tour(seed: int) -> tuple[int, ...]:
    return tuple(random.Random(seed).sample(CITIES, CITY_COUNT))


def test_known_optimal_tour_matches_the_verified_length() -> None:
    assert math.isclose(
        tsp_fitness(KNOWN_OPTIMAL_TOUR), KNOWN_OPTIMAL_LENGTH, rel_tol=0, abs_tol=1e-12
    )


def test_deterministic_baseline_reaches_the_known_optimum() -> None:
    route, distance = baseline_tour()

    assert sorted(route) == CITIES
    assert distance == pytest.approx(KNOWN_OPTIMAL_LENGTH, abs=1e-12)


def test_route_length_closes_the_cycle_and_rejects_empty_routes() -> None:
    route = (0, 1, 2)
    expected = CITY_DISTANCES[0][1] + CITY_DISTANCES[1][2] + CITY_DISTANCES[2][0]

    assert route_length(route) == pytest.approx(expected)
    with pytest.raises(ValueError, match="至少需要一个城市"):
        route_length(())


@pytest.mark.parametrize("seed", range(20))
def test_two_opt_never_worsens_a_route_and_keeps_it_a_permutation(seed: int) -> None:
    gene = _random_tour(seed)
    improved, distance = two_opt(gene)

    assert sorted(improved) == CITIES
    assert distance <= route_length(gene)
    assert distance == pytest.approx(route_length(improved), abs=1e-12)


def test_two_opt_cannot_improve_the_known_optimum() -> None:
    improved, distance = two_opt(KNOWN_OPTIMAL_TOUR)

    assert distance == pytest.approx(KNOWN_OPTIMAL_LENGTH, abs=1e-12)
    assert sorted(improved) == CITIES


def test_nearest_neighbor_rejects_an_out_of_range_start() -> None:
    with pytest.raises(ValueError, match="超出城市范围"):
        nearest_neighbor_tour(CITY_COUNT)


@pytest.mark.parametrize(
    "gene",
    [(0, 1, 1, *range(3, CITY_COUNT)), tuple(range(CITY_COUNT - 1))],
)
def test_fitness_rejects_invalid_routes(gene: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="排列"):
        tsp_fitness(gene)
    with pytest.raises(ValueError, match="排列"):
        validate_tour(gene)


def test_nearest_neighbor_tour_is_a_permutation() -> None:
    route = nearest_neighbor_tour(0)

    assert sorted(route) == CITIES


def test_gene_factory_is_a_seeded_permutation() -> None:
    first = generate_permutation(CITY_COUNT, random.Random(7))
    second = generate_permutation(CITY_COUNT, random.Random(7))

    assert first == second
    assert sorted(first) == CITIES
    with pytest.raises(ValueError, match="至少为 2"):
        generate_permutation(1, random.Random(0))


@pytest.mark.parametrize("seed", range(50))
def test_ordered_crossover_always_returns_permutations(seed: int) -> None:
    first = _random_tour(seed)
    second = _random_tour(seed + 1000)

    for child in ordered_crossover(first, second, random.Random(seed)):
        assert sorted(child) == CITIES


def test_ordered_crossover_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="长度必须相同"):
        ordered_crossover((0, 1), (0, 1, 2), random.Random(0))
    with pytest.raises(ValueError, match="至少需要两个基因"):
        ordered_crossover((0,), (0,), random.Random(0))


def test_inversion_mutation_is_pure_and_reverses_one_segment() -> None:
    class ControlledRandom:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def sample(_population: range, _count: int) -> list[int]:
            return [1, 4]

    gene = tuple(range(6))
    mutated = inversion_mutation(gene, ControlledRandom(), 1.0)

    assert gene == tuple(range(6))
    assert mutated == (0, 4, 3, 2, 1, 5)


def test_inversion_mutation_skips_when_the_probability_is_not_met() -> None:
    gene = (0, 1, 2, 3)

    assert inversion_mutation(gene, random.Random(0), 0.0) == gene


def test_memetic_defaults_reach_the_known_optimum_on_fixed_seeds() -> None:
    distances = [TspGA(GAConfig(seed=seed)).train().fitness for seed in range(5)]

    assert all(distance == pytest.approx(KNOWN_OPTIMAL_LENGTH, abs=1e-12) for distance in distances)


def test_memetic_run_is_reproducible() -> None:
    first = TspGA(GAConfig(seed=11), memetic=True).train()
    second = TspGA(GAConfig(seed=11), memetic=True).train()

    assert first == second


def test_pure_genetic_algorithm_stays_behind_memetic() -> None:
    pure = TspGA(GAConfig(population_size=80, generations=300, seed=0), memetic=False).train()
    memetic = TspGA(GAConfig(seed=0), memetic=True).train()

    gap = (pure.fitness / KNOWN_OPTIMAL_LENGTH - 1) * 100
    assert 0 < gap < 10
    assert memetic.fitness < pure.fitness


def test_experiment_is_reproducible_and_competitive() -> None:
    config = GAConfig(seed=0)
    first = run_experiment(5, config)
    second = run_experiment(5, config)

    assert first == second
    assert first.base_seed == 0
    assert first.optimal_runs == 5
    assert first.runs_within_percent(1) == 5
    assert statistics.mean(first.distances) == first.mean


def test_pure_experiment_reports_the_gap() -> None:
    summary = run_experiment(2, GAConfig(population_size=20, generations=10, seed=0), memetic=False)

    assert summary.mean >= KNOWN_OPTIMAL_LENGTH
    assert summary.worst >= summary.best
    assert len(summary.distances) == 2


def test_unseeded_experiment_reports_a_replayable_base_seed() -> None:
    config = GAConfig(population_size=20, generations=2, seed=None)
    first = run_experiment(2, config)
    replay = run_experiment(2, GAConfig(population_size=20, generations=2, seed=first.base_seed))

    assert 0 <= first.base_seed < 2**32
    assert replay.distances == first.distances
    assert replay.base_seed == first.base_seed


def test_experiment_rejects_zero_runs() -> None:
    with pytest.raises(ValueError, match="至少为 1"):
        run_experiment(0, GAConfig())


def test_experiment_summary_rejects_a_negative_percent() -> None:
    with pytest.raises(ValueError, match="不能为负"):
        ExperimentSummary((1.0,), 1.0).runs_within_percent(-1)
