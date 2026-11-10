"""Command-line entry point for the TSP genetic algorithm."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from constant import KNOWN_OPTIMAL_LENGTH
from ga import Ga, GAConfig
from heuristics import best_nearest_neighbor_two_opt
from individual import Individual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve the bundled 25-city TSP")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--mutation-probability", type=float, default=0.25)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--show", action="store_true", help="open plot windows")
    parser.add_argument("--no-plot", action="store_true", help="skip image generation")
    parser.add_argument("--progress", action="store_true", help="print every generation")
    return parser


def _progress(generation: int, best: Individual[tuple[int, ...]]) -> None:
    print(f"generation={generation:4d} best_distance={best.fitness:.12f}")


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
    solver = Ga(config)
    best = solver.train(_progress if args.progress else None)
    _, baseline_distance = best_nearest_neighbor_two_opt()
    gap = (best.fitness / KNOWN_OPTIMAL_LENGTH - 1) * 100

    print("best_route=" + ",".join(map(str, best.gene)))
    print(f"best_distance={best.fitness:.12f}")
    print(f"known_optimum={KNOWN_OPTIMAL_LENGTH:.12f}")
    print(f"optimality_gap_percent={gap:.6f}")
    print(f"nearest_neighbor_2opt={baseline_distance:.12f}")

    if not args.no_plot:
        # Imported lazily so `--no-plot` stays usable without matplotlib installed.
        from plot import Plot

        route_path, fitness_path = Plot(solver, args.output_dir).plot_all(show=args.show)
        print(f"route_plot={route_path.resolve()}")
        print(f"fitness_plot={fitness_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
