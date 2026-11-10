"""通用遗传算法引擎。"""

import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from statistics import fmean
from typing import Generic, TypeVar

GeneT = TypeVar("GeneT")

GeneFactory = Callable[[random.Random], GeneT]
FitnessFunction = Callable[[GeneT], float]
CrossoverFunction = Callable[[GeneT, GeneT, random.Random], tuple[GeneT, GeneT]]
MutationFunction = Callable[[GeneT, random.Random, float], GeneT]


@dataclass(frozen=True, slots=True)
class Individual(Generic[GeneT]):
    """一个解和它的得分。"""

    gene: GeneT
    fitness: float


@dataclass(frozen=True, slots=True)
class GAConfig:
    """运行参数，构造时校验。"""

    population_size: int = 30
    generations: int = 30
    mutation_probability: float = 0.25
    tournament_size: int = 3
    elite_count: int = 2
    minimize: bool = True
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("种群规模至少为 2")
        if self.generations < 1:
            raise ValueError("迭代代数至少为 1")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("变异概率必须在 0 到 1 之间")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("锦标赛规模必须在 2 到种群规模之间")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("精英个体数必须小于种群规模")


class GeneticAlgorithm(Generic[GeneT]):
    """遗传算法。同一 seed 结果完全一致。

    传 ``refine_function`` 时，每个新个体先被它改进一次再打分（memetic）；
    不传就是纯遗传算法。
    """

    def __init__(
        self,
        *,
        config: GAConfig,
        gene_factory: GeneFactory[GeneT],
        fitness_function: FitnessFunction[GeneT],
        crossover_function: CrossoverFunction[GeneT],
        mutation_function: MutationFunction[GeneT],
        refine_function: Callable[[GeneT], GeneT] | None = None,
        clone_function: Callable[[GeneT], GeneT] = deepcopy,
    ) -> None:
        self.config = config
        self.gene_factory = gene_factory
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.refine_function = refine_function
        self.clone_function = clone_function
        self.population: list[Individual[GeneT]] = []
        self.best: Individual[GeneT] | None = None
        self.best_history: list[float] = []
        self.mean_history: list[float] = []
        self._rng = random.Random(config.seed)

    def _reset(self) -> None:
        self._rng = random.Random(self.config.seed)
        self.population = []
        self.best = None
        self.best_history = []
        self.mean_history = []

    def _evaluate(self, gene: GeneT) -> Individual[GeneT]:
        if self.refine_function is not None:
            gene = self.refine_function(gene)
        stable = self.clone_function(gene)
        return Individual(stable, float(self.fitness_function(stable)))

    def _rank(self, population: list[Individual[GeneT]]) -> list[Individual[GeneT]]:
        return sorted(
            population,
            key=lambda individual: individual.fitness,
            reverse=not self.config.minimize,
        )

    def _best(self, population: list[Individual[GeneT]]) -> Individual[GeneT]:
        """取最好的个体。"""
        if self.config.minimize:
            return min(population, key=lambda individual: individual.fitness)
        return max(population, key=lambda individual: individual.fitness)

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        if self.config.minimize:
            return candidate < incumbent
        return candidate > incumbent

    def _select_parent(self) -> Individual[GeneT]:
        """锦标赛选择：抽几个不同的个体，取最好的。"""
        population = self.population
        size = len(population)
        picks = [int(self._rng.random() * size)]
        while len(picks) < self.config.tournament_size:
            index = int(self._rng.random() * size)
            if index not in picks:
                picks.append(index)
        best = population[picks[0]]
        for index in picks[1:]:
            candidate = population[index]
            if self._is_better(candidate.fitness, best.fitness):
                best = candidate
        return best

    def _next_generation(self) -> None:
        size = self.config.population_size
        next_population = self._rank(self.population)[: self.config.elite_count]
        while len(next_population) < size:
            first = self._select_parent()
            second = self._select_parent()
            children = self.crossover_function(
                self.clone_function(first.gene),
                self.clone_function(second.gene),
                self._rng,
            )
            if not children:
                raise ValueError("交叉算子至少返回一个子代")
            for gene in children:
                if len(next_population) == size:
                    break
                mutated = self.mutation_function(
                    self.clone_function(gene), self._rng, self.config.mutation_probability
                )
                next_population.append(self._evaluate(mutated))
        self.population = next_population

    def train(
        self, progress_callback: Callable[[int, Individual[GeneT]], None] | None = None
    ) -> Individual[GeneT]:
        """从头训练，返回最好的个体。"""
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
            self.best_history.append(self.best.fitness)
            self.mean_history.append(fmean(individual.fitness for individual in self.population))
            if progress_callback is not None:
                progress_callback(generation, self.best)

        return self.best
