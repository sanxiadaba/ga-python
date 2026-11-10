"""把路线和适应度曲线存成 png。"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .ga import GeneticAlgorithm
from .tsp import CITY_POS, KNOWN_OPTIMAL_LENGTH

ROUTE_FILENAME = "route.png"
FITNESS_FILENAME = "fitness.png"


def _save(figure: Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    return destination


def _route_figure(route: Sequence[int]) -> Figure:
    closed = (*route, route[0])
    figure, axis = plt.subplots()
    axis.plot(
        [CITY_POS[index][0] for index in closed],
        [CITY_POS[index][1] for index in closed],
        "o-",
        color="tab:red",
        linewidth=1.2,
        markersize=5,
    )
    axis.set(title="TSP route", xlabel="x", ylabel="y")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    return figure


def _fitness_figure(solver: GeneticAlgorithm[tuple[int, ...]]) -> Figure:
    generations = range(1, len(solver.best_history) + 1)
    figure, axis = plt.subplots()
    axis.plot(generations, solver.best_history, color="tab:blue", label="best so far")
    axis.plot(
        generations, solver.mean_history, color="tab:orange", alpha=0.7, label="population mean"
    )
    axis.axhline(
        KNOWN_OPTIMAL_LENGTH, color="tab:green", linestyle="--", linewidth=1, label="known optimum"
    )
    axis.set(title="Tour length by generation", xlabel="Generation", ylabel="Distance")
    axis.legend()
    axis.grid(alpha=0.25)
    return figure


def save_plots(
    solver: GeneticAlgorithm[tuple[int, ...]], output_dir: str | Path, *, show: bool = False
) -> tuple[Path, Path]:
    """存两张图，返回路径。"""
    best = solver.best
    if best is None:
        raise ValueError("需要先训练再绘图")
    directory = Path(output_dir)
    route_figure = _route_figure(best.gene)
    fitness_figure = _fitness_figure(solver)
    try:
        route_path = _save(route_figure, directory / ROUTE_FILENAME)
        fitness_path = _save(fitness_figure, directory / FITNESS_FILENAME)
        if show:
            plt.show()
    finally:
        plt.close(route_figure)
        plt.close(fitness_figure)
    return route_path, fitness_path
