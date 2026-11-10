import importlib.resources

import ga_tsp
from ga_tsp import Ga, GAConfig, GeneticAlgorithm, Individual


def test_public_api_is_exported() -> None:
    assert ga_tsp.__version__ == "2.0.0"
    assert ga_tsp.__all__ == ["GAConfig", "Ga", "GeneticAlgorithm", "Individual"]
    assert issubclass(Ga, GeneticAlgorithm)
    assert GAConfig().population_size == 80
    assert Individual(1, 2.0).fitness == 2.0


def test_py_typed_marker_is_shipped() -> None:
    marker = importlib.resources.files("ga_tsp").joinpath("py.typed")
    assert marker.is_file()


def test_package_does_not_shadow_top_level_modules() -> None:
    import inspect

    assert inspect.getmodule(ga_tsp.Ga) is not None
    assert inspect.getmodule(ga_tsp.Ga).__name__ == "ga_tsp.ga"  # type: ignore[union-attr]
