import random

import pytest

from cross import ordered_crossover
from fitness import tsp_fitness, validate_tour
from gen_gene import generate_permutation
from mutate import inversion_mutation


def test_gene_factory_is_seeded_permutation() -> None:
    first = generate_permutation(25, random.Random(7))
    second = generate_permutation(25, random.Random(7))
    assert first == second
    assert sorted(first) == list(range(25))


def test_ordered_crossover_always_preserves_permutations() -> None:
    first = tuple(range(25))
    second = tuple(reversed(first))
    for seed in range(100):
        children = ordered_crossover(first, second, random.Random(seed))
        assert all(sorted(child) == list(range(25)) for child in children)


def test_inversion_mutation_is_pure_and_changes_an_inclusive_segment() -> None:
    gene = tuple(range(6))

    class ControlledRandom:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def sample(_population: range, _count: int) -> list[int]:
            return [1, 5]

    mutated = inversion_mutation(gene, ControlledRandom(), 1.0)  # type: ignore[arg-type]
    assert gene == tuple(range(6))
    assert mutated == (0, 5, 4, 3, 2, 1)
    assert sorted(mutated) == list(range(6))


def test_tsp_fitness_rejects_invalid_routes() -> None:
    with pytest.raises(ValueError, match="permutation"):
        tsp_fitness((0, 1, 1))
    with pytest.raises(ValueError, match="permutation"):
        validate_tour((0, 2), 2)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_mutation_probability_boundaries_leave_validation_to_config(
    probability: float,
) -> None:
    # The operator is intentionally simple; GAConfig owns policy validation.
    gene = (0, 1, 2)
    result = inversion_mutation(gene, random.Random(0), probability)
    assert sorted(result) == [0, 1, 2]
