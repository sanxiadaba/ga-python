"""Reusable genetic algorithm plus a configured TSP solver."""

from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter
from typing import Generic, TypeVar

from constant import CITY_POS
from cross import ordered_crossover
from fitness import tsp_fitness
from gen_gene import generate_permutation
from individual import Individual
from mutate import inversion_mutation

GeneT = TypeVar("GeneT")
GeneFactory = Callable[[random.Random], GeneT]
FitnessFunction = Callable[[GeneT], float]
CrossoverFunction = Callable[[GeneT, GeneT, random.Random], tuple[GeneT, GeneT]]
MutationFunction = Callable[[GeneT, random.Random, float], GeneT]
ProgressCallback = Callable[[int, Individual[GeneT]], None]
CloneFunction = Callable[[GeneT], GeneT]


@dataclass(frozen=True, slots=True)
class GAConfig:
    population_size: int = 80
    generations: int = 300
    mutation_probability: float = 0.25
    tournament_size: int = 3
    elite_count: int = 2
    minimize: bool = True
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be between 0 and 1")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between 2 and population_size")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be smaller than population_size")


class GeneticAlgorithm(Generic[GeneT]):
    """A deterministic, strategy-driven genetic algorithm."""

    def __init__(
        self,
        *,
        config: GAConfig,
        gene_factory: GeneFactory[GeneT],
        fitness_function: FitnessFunction[GeneT],
        crossover_function: CrossoverFunction[GeneT],
        mutation_function: MutationFunction[GeneT],
        clone_function: CloneFunction[GeneT] = deepcopy,
    ) -> None:
        self.config = config
        self.gene_factory = gene_factory
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.clone_function = clone_function
        self.population: list[Individual[GeneT]] = []
        self.best: Individual[GeneT] | None = None
        self.best_list: list[Individual[GeneT]] = []
        self.result_list: list[GeneT] = []
        self.fitness_list: list[float] = []
        self._rng = random.Random(config.seed)

    @property
    def individual_list(self) -> list[Individual[GeneT]]:
        """Compatibility name for callers of the original project."""
        return self.population

    def _reset(self) -> None:
        self._rng = random.Random(self.config.seed)
        self.population = []
        self.best = None
        self.best_list = []
        self.result_list = []
        self.fitness_list = []

    def _evaluate(self, gene: GeneT) -> Individual[GeneT]:
        stable_gene = self.clone_function(gene)
        return Individual(stable_gene, float(self.fitness_function(stable_gene)))

    def _rank(self, population: list[Individual[GeneT]]) -> list[Individual[GeneT]]:
        return sorted(
            population,
            key=lambda individual: individual.fitness,
            reverse=not self.config.minimize,
        )

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        if self.config.minimize:
            return candidate < incumbent
        return candidate > incumbent

    def _best(self, population: list[Individual[GeneT]]) -> Individual[GeneT]:
        """Return the best individual without paying for a full sort."""
        chooser = min if self.config.minimize else max
        return chooser(population, key=attrgetter("fitness"))

    def _select_parent(self) -> Individual[GeneT]:
        contestants = self._rng.sample(self.population, self.config.tournament_size)
        return self._best(contestants)

    def _next_generation(self) -> None:
        next_population = self._rank(self.population)[: self.config.elite_count]
        while len(next_population) < self.config.population_size:
            first_parent = self._select_parent()
            second_parent = self._select_parent()
            first_gene, second_gene = self.crossover_function(
                self.clone_function(first_parent.gene),
                self.clone_function(second_parent.gene),
                self._rng,
            )
            for gene in (first_gene, second_gene):
                mutated = self.mutation_function(
                    self.clone_function(gene), self._rng, self.config.mutation_probability
                )
                next_population.append(self._evaluate(mutated))
                if len(next_population) == self.config.population_size:
                    break
        self.population = next_population

    def train(self, progress_callback: ProgressCallback[GeneT] | None = None) -> Individual[GeneT]:
        """Reset state, train, and return the best individual found."""
        self._reset()
        self.population = [
            self._evaluate(self.gene_factory(self._rng)) for _ in range(self.config.population_size)
        ]
        self.best = self._best(self.population)

        for generation in range(1, self.config.generations + 1):
            self._next_generation()
            generation_best = self._best(self.population)
            if self._is_better(generation_best.fitness, self.best.fitness):
                self.best = generation_best
            self.best_list.append(self.best)
            self.result_list.append(self.clone_function(self.best.gene))
            self.fitness_list.append(self.best.fitness)
            if progress_callback is not None:
                progress_callback(generation, self.best)

        return self.best


class Ga(GeneticAlgorithm[tuple[int, ...]]):
    """Genetic algorithm configured for the bundled TSP data set."""

    def __init__(self, config: GAConfig | None = None) -> None:
        selected_config = config or GAConfig()
        super().__init__(
            config=selected_config,
            gene_factory=lambda rng: generate_permutation(len(CITY_POS), rng),
            fitness_function=tsp_fitness,
            crossover_function=ordered_crossover,
            mutation_function=inversion_mutation,
            clone_function=lambda gene: gene,
        )
