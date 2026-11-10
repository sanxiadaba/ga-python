import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ga_tsp import GAConfig, TspGA
from ga_tsp.cli import main
from ga_tsp.plot import save_plots


def _trained_solver() -> TspGA:
    solver = TspGA(GAConfig(population_size=10, generations=2))
    solver.train()
    return solver


def test_single_run_prints_the_route_and_distance(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--no-plot"]) == 0

    output = capsys.readouterr().out
    assert "最优路线" in output
    assert "最优距离" in output
    assert "基线距离" in output


def test_pure_mode_runs_without_the_refine_step(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--no-plot", "--pure", "--population-size", "20", "--generations", "20"]) == 0

    assert "最优差距" in capsys.readouterr().out


def test_progress_prints_every_generation(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--no-plot", "--progress", "--population-size", "10", "--generations", "2"])

    output = capsys.readouterr().out
    assert "第    1 代" in output
    assert "第    2 代" in output


def test_plots_are_written_to_the_output_directory(tmp_path: Path) -> None:
    assert main(["--output-dir", str(tmp_path)]) == 0

    assert (tmp_path / "route.png").is_file()
    assert (tmp_path / "fitness.png").is_file()


def test_experiment_mode_prints_statistics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--runs", "2"]) == 0

    output = capsys.readouterr().out
    assert "实验次数" in output
    assert "命中最优" in output


def test_invalid_configuration_is_reported_as_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_code:
        main(["--no-plot", "--population-size", "1"])

    assert exit_code.value.code == 2


def test_show_and_no_plot_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exit_code:
        main(["--show", "--no-plot"])

    assert exit_code.value.code == 2


def test_zero_runs_is_rejected_as_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_code:
        main(["--runs", "0"])

    assert exit_code.value.code == 2


def test_module_entry_point_runs() -> None:
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-m", "ga_tsp", "--no-plot", "--generations", "2"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "最优距离" in result.stdout


def test_save_plots_releases_figures(tmp_path: Path) -> None:
    route_path, fitness_path = save_plots(_trained_solver(), tmp_path)

    assert route_path.is_file()
    assert fitness_path.is_file()
    assert plt.get_fignums() == []


def test_save_plots_requires_a_trained_solver(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="先训练"):
        save_plots(TspGA(GAConfig(population_size=10, generations=1)), tmp_path)


def test_show_releases_figures_under_a_headless_backend(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_plots(_trained_solver(), tmp_path, show=True)

    assert plt.get_fignums() == []
