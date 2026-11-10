import os
import subprocess
import sys
from pathlib import Path

import pytest

from ga_tsp.experiment import main as experiment_main
from ga_tsp.main import main as cli_main


def test_main_generates_plots_and_exits(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ga_tsp.main",
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
        timeout=60,
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
            "ga_tsp.experiment",
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
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "runs=2" in result.stdout
    assert "mean_gap_percent=" in result.stdout


def test_main_reports_invalid_configuration_without_traceback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ga_tsp.main", "--population-size", "1", "--no-plot"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "population_size must be at least 2" in result.stderr
    assert "Traceback" not in result.stderr


def test_main_prefers_the_console_script_entry_point() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ga_tsp.main import main; raise SystemExit(main(['--no-plot', "
            "'--generations', '2', '--population-size', '10']))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "best_route=" in result.stdout


def test_main_runs_in_process_and_reports_progress(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(["--no-plot", "--progress", "--generations", "2", "--population-size", "10"])
    assert code == 0
    output = capsys.readouterr().out
    assert "generation=   1" in output
    assert "generation=   2" in output


def test_hybrid_flag_never_worsens_the_route(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["--no-plot", "--hybrid", "--generations", "5", "--population-size", "20"]) == 0
    output = capsys.readouterr().out
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    assert "hybrid_route" in values
    assert float(values["hybrid_distance"]) <= float(values["best_distance"])


def test_experiment_runs_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    assert experiment_main(["--runs", "1", "--generations", "2", "--population-size", "10"]) == 0
    output = capsys.readouterr().out
    assert "runs=1" in output
    assert "base_seed=0" in output


def test_show_and_no_plot_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exit_code:
        cli_main(["--show", "--no-plot"])
    assert exit_code.value.code == 2


def test_main_writes_plots_in_process(tmp_path: Path) -> None:
    assert (
        cli_main(
            [
                "--generations",
                "2",
                "--population-size",
                "10",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "plot_tsp.jpg").is_file()
    assert (tmp_path / "fitness.jpg").is_file()


def test_main_reports_invalid_configuration_in_process() -> None:
    with pytest.raises(SystemExit) as exit_code:
        cli_main(["--population-size", "1", "--no-plot"])
    assert exit_code.value.code == 2


def test_experiment_reports_invalid_configuration_in_process() -> None:
    with pytest.raises(SystemExit) as exit_code:
        experiment_main(["--population-size", "1"])
    assert exit_code.value.code == 2
