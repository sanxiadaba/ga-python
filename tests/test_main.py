import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from main import main


def test_prints_the_route_and_the_distance(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main([]) == 0

    output = capsys.readouterr().out
    assert "最优路线" in output
    assert "最优距离" in output
    assert (tmp_path / "output" / "route.png").is_file()
    assert (tmp_path / "output" / "fitness.png").is_file()


def test_no_plot_skips_the_images(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--no-plot"]) == 0

    assert "路线图" not in capsys.readouterr().out


def test_pure_mode_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--no-plot", "--pure", "--generations", "5"]) == 0

    assert "最优差距" in capsys.readouterr().out


def test_show_works_under_a_headless_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["--show"]) == 0

    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "arguments", [["--no-plot", "--population-size", "1"], ["--no-plot", "--generations", "0"]]
)
def test_rejects_bad_numbers(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_code:
        main(arguments)

    assert exit_code.value.code == 2
