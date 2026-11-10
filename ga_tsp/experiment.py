"""多种子实验统计。"""

import random
import statistics
from dataclasses import dataclass, replace

from .ga import GAConfig
from .tsp import KNOWN_OPTIMAL_LENGTH, TspGA


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """多次运行的汇总。"""

    distances: tuple[float, ...]
    optimum: float
    base_seed: int = 0

    @property
    def best(self) -> float:
        return min(self.distances)

    @property
    def worst(self) -> float:
        return max(self.distances)

    @property
    def mean(self) -> float:
        return statistics.mean(self.distances)

    @property
    def median(self) -> float:
        return statistics.median(self.distances)

    @property
    def standard_deviation(self) -> float:
        return statistics.pstdev(self.distances)

    @property
    def mean_gap_percent(self) -> float:
        return (self.mean / self.optimum - 1) * 100

    @property
    def optimal_runs(self) -> int:
        return sum(distance <= self.optimum * (1 + 1e-9) for distance in self.distances)

    def runs_within_percent(self, percent: float) -> int:
        if percent < 0:
            raise ValueError("百分比不能为负")
        limit = self.optimum * (1 + percent / 100)
        return sum(distance <= limit for distance in self.distances)


def resolve_base_seed(config: GAConfig) -> int:
    """没给种子就随机抽一个。"""
    if config.seed is not None:
        return config.seed
    return random.SystemRandom().randrange(2**32)


def run_experiment(runs: int, base_config: GAConfig, *, memetic: bool = True) -> ExperimentSummary:
    """用连续种子各跑一次。"""
    if runs < 1:
        raise ValueError("实验次数至少为 1")
    base_seed = resolve_base_seed(base_config)
    distances = tuple(
        TspGA(replace(base_config, seed=base_seed + offset), memetic=memetic).train().fitness
        for offset in range(runs)
    )
    return ExperimentSummary(distances, KNOWN_OPTIMAL_LENGTH, base_seed)
