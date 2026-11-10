import random
from dataclasses import replace

import pytest

from ga_tsp import GAConfig, GeneticAlgorithm, Individual

TARGET = 37
BASE = GAConfig(
    population_size=20,
    generations=20,
    mutation_probability=1.0,
    elite_count=1,
    minimize=False,
    seed=5,
)


def _solver(config: GAConfig = BASE) -> GeneticAlgorithm[int]:
    """在整数基因上最大化 -(x - TARGET)^2 的最小引擎。"""

    def crossover(first: int, second: int, _rng: random.Random) -> tuple[int, int]:
        midpoint = round((first + second) / 2)
        return midpoint, midpoint

    def mutate(value: int, _rng: random.Random, _probability: float) -> int:
        if value < TARGET:
            return value + 1
        if value > TARGET:
            return value - 1
        return value

    return GeneticAlgorithm[int](
        config=config,
        gene_factory=lambda rng: rng.randrange(0, 75),
        fitness_function=lambda value: -float((value - TARGET) ** 2),
        crossover_function=crossover,
        mutation_function=mutate,
    )


def test_maximizes_a_generic_gene_type() -> None:
    best = _solver().train()

    assert best.gene == TARGET
    assert best.fitness == 0


def test_population_size_and_best_history_stay_consistent() -> None:
    solver = _solver(replace(BASE, population_size=40, generations=50, seed=3))
    best = solver.train()

    assert len(solver.population) == 40
    assert len(solver.best_history) == 50
    assert all(
        later >= earlier
        for earlier, later in zip(solver.best_history, solver.best_history[1:], strict=False)
    )
    assert best.fitness == solver.best_history[-1]


def test_training_twice_resets_state_and_repeats_the_seeded_result() -> None:
    solver = _solver()
    first = solver.train()
    first_history = tuple(solver.best_history)

    assert solver.train() == first
    assert tuple(solver.best_history) == first_history


def test_refine_hook_runs_on_every_new_individual() -> None:
    seen: list[int] = []

    def refine(gene: int) -> int:
        seen.append(gene)
        return gene

    solver = _solver()
    solver.refine_function = refine
    solver.train()

    children_per_generation = BASE.population_size - BASE.elite_count
    assert len(seen) == BASE.population_size + BASE.generations * children_per_generation


def test_crossover_must_return_at_least_one_child() -> None:
    solver = _solver()
    solver.crossover_function = lambda first, second, rng: ()

    with pytest.raises(ValueError, match="至少返回一个子代"):
        solver.train()


def test_individual_is_immutable() -> None:
    individual = Individual(1, 2.0)

    with pytest.raises(AttributeError):
        individual.fitness = 3.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_size", 1),
        ("generations", 0),
        ("mutation_probability", -0.1),
        ("mutation_probability", 1.1),
        ("tournament_size", 1),
        ("tournament_size", 81),
        ("elite_count", 80),
        ("elite_count", -1),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: int | float) -> None:
    with pytest.raises(ValueError):
        GAConfig(**{field: value})
