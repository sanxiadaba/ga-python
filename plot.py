"""Plotting for the bundled TSP experiment."""

from pathlib import Path

import matplotlib.pyplot as plt

from constant import CITY_POS
from ga import Ga


class Plot:
    def __init__(self, ga: Ga, output_dir: str | Path = "output") -> None:
        if ga.best is None:
            raise ValueError("the genetic algorithm must be trained before plotting")
        self.ga = ga
        self.output_dir = Path(output_dir)

    def plot_all(self, *, show: bool = False) -> tuple[Path, Path]:
        """Save both figures and optionally open a blocking GUI window."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        route_path = self.plot_tsp()
        fitness_path = self.plot_fitness()
        if show:
            plt.show()
        else:
            plt.close("all")
        return route_path, fitness_path

    def plot_fitness(self) -> Path:
        fig, axis = plt.subplots()
        axis.plot(range(1, len(self.ga.fitness_list) + 1), self.ga.fitness_list)
        axis.set(title="Best tour length", xlabel="Generation", ylabel="Distance")
        axis.grid(alpha=0.25)
        destination = self.output_dir / "fitness.jpg"
        fig.savefig(destination, dpi=160, bbox_inches="tight")
        return destination

    def plot_tsp(self) -> Path:
        if self.ga.best is None:
            raise ValueError("the genetic algorithm must be trained before plotting")
        route = (*self.ga.best.gene, self.ga.best.gene[0])
        x_values = [CITY_POS[index][0] for index in route]
        y_values = [CITY_POS[index][1] for index in route]
        fig, axis = plt.subplots()
        axis.plot(x_values, y_values, "o-r")
        axis.set(title="TSP route", xlabel="x", ylabel="y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        destination = self.output_dir / "plot_tsp.jpg"
        fig.savefig(destination, dpi=160, bbox_inches="tight")
        return destination
