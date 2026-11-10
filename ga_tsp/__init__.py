"""遗传算法小库：通用引擎 + 内置 25 城 TSP 示例。"""

from .ga import GAConfig, GeneticAlgorithm, Individual
from .tsp import KNOWN_OPTIMAL_LENGTH, TspGA

__all__ = ["KNOWN_OPTIMAL_LENGTH", "GAConfig", "GeneticAlgorithm", "Individual", "TspGA"]
__version__ = "3.0.0"
