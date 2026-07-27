"""Repeat the TSP experiment and report aggregate quality statistics."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from constant import KNOWN_OPTIMAL_LENGTH
from ga import Ga, GAConfig
from heuristics import best_nearest_neighbor_two_opt


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    distances: tuple[float, ...]
    optimum: float

    @property
    def best(self) -> float:
        return min(self.distances)

    @property
    def worst(self) -> float:
        return max(self.distances)

    @property
    def mean(self) -> float:
        return statistics.mean(self.distances)

    @property
    def median(self) -> float:
        return statistics.median(self.distances)

    @property
    def standard_deviation(self) -> float:
        return statistics.pstdev(self.distances)

    @property
    def mean_gap_percent(self) -> float:
        return (self.mean / self.optimum - 1) * 100

    @property
    def optimal_runs(self) -> int:
        return sum(abs(distance - self.optimum) <= 1e-9 for distance in self.distances)

    def runs_within_percent(self, percent: float) -> int:
        if percent < 0:
            raise ValueError("percent must not be negative")
        limit = self.optimum * (1 + percent / 100)
        return sum(distance <= limit for distance in self.distances)


def run_experiment(runs: int, base_config: GAConfig) -> ExperimentSummary:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    first_seed = base_config.seed or 0
    distances = []
    for offset in range(runs):
        config = GAConfig(
            population_size=base_config.population_size,
            generations=base_config.generations,
            mutation_probability=base_config.mutation_probability,
            tournament_size=base_config.tournament_size,
            elite_count=base_config.elite_count,
            minimize=base_config.minimize,
            seed=first_seed + offset,
        )
        distances.append(Ga(config).train().fitness)
    return ExperimentSummary(tuple(distances), KNOWN_OPTIMAL_LENGTH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeat and summarize the TSP experiment")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--mutation-probability", type=float, default=0.25)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = GAConfig(
            population_size=args.population_size,
            generations=args.generations,
            mutation_probability=args.mutation_probability,
            tournament_size=args.tournament_size,
            elite_count=args.elite_count,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    summary = run_experiment(args.runs, config)
    _, baseline = best_nearest_neighbor_two_opt()
    print(f"runs={len(summary.distances)}")
    print(f"best={summary.best:.12f}")
    print(f"median={summary.median:.12f}")
    print(f"mean={summary.mean:.12f}")
    print(f"standard_deviation={summary.standard_deviation:.12f}")
    print(f"worst={summary.worst:.12f}")
    print(f"known_optimum={summary.optimum:.12f}")
    print(f"mean_gap_percent={summary.mean_gap_percent:.6f}")
    print(f"optimal_runs={summary.optimal_runs}")
    print(f"within_1_percent={summary.runs_within_percent(1)}")
    print(f"within_5_percent={summary.runs_within_percent(5)}")
    print(f"nearest_neighbor_2opt={baseline:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
