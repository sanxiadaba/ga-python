import os
import subprocess
import sys
from pathlib import Path


def test_main_generates_plots_and_exits(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "main",
            "--generations",
            "5",
            "--population-size",
            "20",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "best_distance=" in result.stdout
    assert (tmp_path / "fitness.jpg").is_file()
    assert (tmp_path / "plot_tsp.jpg").is_file()


def test_experiment_cli_exits() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiment",
            "--runs",
            "2",
            "--generations",
            "5",
            "--population-size",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "runs=2" in result.stdout
    assert "mean_gap_percent=" in result.stdout


def test_main_reports_invalid_configuration_without_traceback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "main", "--population-size", "1", "--no-plot"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "population_size must be at least 2" in result.stderr
    assert "Traceback" not in result.stderr
