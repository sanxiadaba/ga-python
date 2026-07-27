import random

import pytest

from ga import Ga, GAConfig, GeneticAlgorithm


def test_population_size_and_global_best_history_are_stable() -> None:
    config = GAConfig(population_size=40, generations=50, seed=3)
    solver = Ga(config)
    best = solver.train()

    assert len(solver.population) == config.population_size
    assert len(solver.best_list) == config.generations
    assert len(solver.result_list) == config.generations
    assert len(solver.fitness_list) == config.generations
    assert all(
        later <= earlier
        for earlier, later in zip(solver.fitness_list, solver.fitness_list[1:], strict=False)
    )
    assert best.fitness == solver.fitness_list[-1]
    assert solver.result_list[-1] == best.gene


def test_training_twice_resets_state_and_repeats_seeded_result() -> None:
    solver = Ga(GAConfig(population_size=30, generations=20, seed=11))
    first = solver.train()
    first_history = tuple(solver.fitness_list)
    second = solver.train()

    assert second == first
    assert tuple(solver.fitness_list) == first_history
    assert len(solver.best_list) == 20
    assert len(solver.result_list) == 20


def test_strategy_api_supports_function_maximization() -> None:
    target = 37

    def factory(rng: random.Random) -> int:
        return rng.randrange(0, 75)

    def crossover(first: int, second: int, _rng: random.Random) -> tuple[int, int]:
        midpoint = round((first + second) / 2)
        return midpoint, midpoint

    def mutate(value: int, _rng: random.Random, _probability: float) -> int:
        if value < target:
            return value + 1
        if value > target:
            return value - 1
        return value

    solver = GeneticAlgorithm[int](
        config=GAConfig(
            population_size=20,
            generations=20,
            mutation_probability=1,
            tournament_size=3,
            elite_count=1,
            minimize=False,
            seed=5,
        ),
        gene_factory=factory,
        fitness_function=lambda value: -float((value - target) ** 2),
        crossover_function=crossover,
        mutation_function=mutate,
    )
    best = solver.train()
    assert best.gene == target
    assert best.fitness == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_size", 1),
        ("generations", 0),
        ("mutation_probability", -0.1),
        ("mutation_probability", 1.1),
        ("tournament_size", 1),
        ("elite_count", 80),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: int | float) -> None:
    values = {field: value}
    with pytest.raises(ValueError):
        GAConfig(**values)  # type: ignore[arg-type]
