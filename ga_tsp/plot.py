"""Plotting for the bundled TSP experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .constant import CITY_POS
from .ga import Ga


class Plot:
    def __init__(self, ga: Ga, output_dir: str | Path = "output") -> None:
        if ga.best is None:
            raise ValueError("the genetic algorithm must be trained before plotting")
        self.ga = ga
        self.output_dir = Path(output_dir)

    def plot_all(self, *, show: bool = False) -> tuple[Path, Path]:
        """Save both figures and optionally open a blocking GUI window."""
        route_figure = self._tour_figure()
        fitness_figure = self._fitness_figure()
        route_path = self._save(route_figure, "plot_tsp.jpg")
        fitness_path = self._save(fitness_figure, "fitness.jpg")
        if show:
            plt.show()
        self._close(route_figure, fitness_figure)
        return route_path, fitness_path

    def plot_fitness(self, *, show: bool = False) -> Path:
        """Save the best-fitness history and release its figure."""
        figure = self._fitness_figure()
        path = self._save(figure, "fitness.jpg")
        if show:
            plt.show()
        self._close(figure)
        return path

    def plot_tsp(self, *, show: bool = False) -> Path:
        """Save the best route and release its figure."""
        figure = self._tour_figure()
        path = self._save(figure, "plot_tsp.jpg")
        if show:
            plt.show()
        self._close(figure)
        return path

    def _save(self, figure: Figure, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / name
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        return destination

    @staticmethod
    def _close(*figures: Figure) -> None:
        for figure in figures:
            plt.close(figure)

    def _fitness_figure(self) -> Figure:
        figure, axis = plt.subplots()
        axis.plot(range(1, len(self.ga.fitness_list) + 1), self.ga.fitness_list)
        axis.set(title="Best tour length", xlabel="Generation", ylabel="Distance")
        axis.grid(alpha=0.25)
        return figure

    def _tour_figure(self) -> Figure:
        if self.ga.best is None:  # pragma: no cover - __init__ already rejects untrained solvers
            raise ValueError("the genetic algorithm must be trained before plotting")
        route = (*self.ga.best.gene, self.ga.best.gene[0])
        x_values = [CITY_POS[index][0] for index in route]
        y_values = [CITY_POS[index][1] for index in route]
        figure, axis = plt.subplots()
        axis.plot(x_values, y_values, "o-r")
        axis.set(title="TSP route", xlabel="x", ylabel="y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        return figure
