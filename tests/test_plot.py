import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ga_tsp import Ga, GAConfig
from ga_tsp.plot import Plot


def _trained_solver() -> Ga:
    solver = Ga(GAConfig(population_size=10, generations=3))
    solver.train()
    return solver


def test_plot_methods_release_figures(tmp_path: Path) -> None:
    plot = Plot(_trained_solver(), tmp_path)
    for _ in range(3):
        assert plot.plot_tsp().is_file()
        assert plot.plot_fitness().is_file()
    assert plt.get_fignums() == []

    route_path, fitness_path = plot.plot_all()
    assert route_path.is_file()
    assert fitness_path.is_file()
    assert plt.get_fignums() == []


def test_plot_rejects_untrained_algorithm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trained"):
        Plot(Ga(GAConfig(population_size=10, generations=1)), tmp_path)


def test_show_releases_figures_under_a_headless_backend(tmp_path: Path) -> None:
    plot = Plot(_trained_solver(), tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot.plot_tsp(show=True)
        plot.plot_fitness(show=True)
        route_path, fitness_path = plot.plot_all(show=True)
    assert route_path.is_file()
    assert fitness_path.is_file()
    assert plt.get_fignums() == []
