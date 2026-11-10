"""内置的 25 城 TSP：坐标、距离矩阵、fitness、确定性基线与 TSP 求解器。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import cache
from math import dist

from .ga import GAConfig, GeneticAlgorithm
from .operators import generate_permutation, inversion_mutation, ordered_crossover

CITY_POS: tuple[tuple[float, float], ...] = (
    (0.34373988, 0.28799359),
    (0.59614005, 0.20891475),
    (0.31594294, 0.24015942),
    (0.00445381, 0.27850897),
    (0.72902031, 0.02892819),
    (0.8927462, 0.72593335),
    (0.69496064, 0.49234185),
    (0.13480979, 0.80326909),
    (0.09446444, 0.74511812),
    (0.00322167, 0.83702802),
    (0.13020064, 0.52840454),
    (0.42933579, 0.8439443),
    (0.23382182, 0.40566047),
    (0.76388352, 0.75707397),
    (0.55766312, 0.403058),
    (0.18937066, 0.0122916),
    (0.19229783, 0.33401292),
    (0.10528408, 0.39235952),
    (0.037418, 0.04081298),
    (0.38220903, 0.65365966),
    (0.2893925, 0.3846537),
    (0.38730741, 0.20597485),
    (0.18536116, 0.34750257),
    (0.69820574, 0.16041619),
    (0.31628208, 0.50875925),
)

CITY_COUNT = len(CITY_POS)

# 用独立的混合整数规划模型求到零 MIP gap 的最优解。
KNOWN_OPTIMAL_LENGTH = 4.267835351807
KNOWN_OPTIMAL_TOUR: tuple[int, ...] = (
    0,
    21,
    4,
    23,
    1,
    14,
    6,
    5,
    13,
    11,
    19,
    7,
    9,
    8,
    10,
    24,
    20,
    12,
    16,
    22,
    17,
    3,
    18,
    15,
    2,
)

Point = tuple[float, float]
DistanceMatrix = tuple[tuple[float, ...], ...]


def build_distance_matrix(points: Sequence[Point]) -> DistanceMatrix:
    """预计算所有城市两两之间的欧氏距离。"""
    return tuple(tuple(dist(first, second) for second in points) for first in points)


CITY_DISTANCES = build_distance_matrix(CITY_POS)


def route_length(route: Sequence[int], distance_matrix: DistanceMatrix = CITY_DISTANCES) -> float:
    """闭合路线的总长度。"""
    if not route:
        raise ValueError("路线至少需要一个城市")
    total = 0.0
    previous = route[-1]
    for city in route:
        total += distance_matrix[previous][city]
        previous = city
    return total


@cache
def _required_cities(city_count: int) -> frozenset[int]:
    return frozenset(range(city_count))


def validate_tour(gene: Sequence[int], city_count: int = CITY_COUNT) -> None:
    """拒绝丢城市或重复城市的路线。"""
    if len(gene) != city_count or set(gene) != _required_cities(city_count):
        raise ValueError(f"路线必须是 0..{city_count - 1} 的一个排列")


def tsp_fitness(gene: Sequence[int], distance_matrix: DistanceMatrix = CITY_DISTANCES) -> float:
    """闭合 TSP 路线的长度，顺便把住排列约束这个关口。"""
    validate_tour(gene, len(distance_matrix))
    return route_length(gene, distance_matrix)


def nearest_neighbor_tour(
    start: int, distance_matrix: DistanceMatrix = CITY_DISTANCES
) -> tuple[int, ...]:
    """从 start 出发的最近邻路线。"""
    if not 0 <= start < len(distance_matrix):
        raise ValueError("起点超出城市范围")
    route = [start]
    remaining = set(range(len(distance_matrix))) - {start}
    while remaining:
        previous = route[-1]
        distances = distance_matrix[previous]
        next_city = min(remaining, key=lambda city: (distances[city], city))
        route.append(next_city)
        remaining.remove(next_city)
    return tuple(route)


def two_opt(
    route: Sequence[int], distance_matrix: DistanceMatrix = CITY_DISTANCES
) -> tuple[tuple[int, ...], float]:
    """用 2-opt 局部搜索改进闭合路线，直到没有可改进的边对。

    每次只用四条边的长度差判断是否改进，不再重算整条路线。
    """
    best = list(route)
    count = len(best)
    improved = True
    while improved:
        improved = False
        for head in range(count - 1):
            first, second = best[head], best[head + 1]
            distances = distance_matrix[first]
            base = distances[second]
            for tail in range(head + 2, count):
                if head == 0 and tail == count - 1:
                    continue
                tail_city = best[tail]
                closing_city = best[(tail + 1) % count]
                gain = (
                    distances[tail_city]
                    + distance_matrix[second][closing_city]
                    - base
                    - distance_matrix[tail_city][closing_city]
                )
                if gain < -1e-12:
                    best[head + 1 : tail + 1] = reversed(best[head + 1 : tail + 1])
                    second = best[head + 1]
                    base = distances[second]
                    improved = True
    return tuple(best), route_length(best, distance_matrix)


def baseline_tour(
    distance_matrix: DistanceMatrix = CITY_DISTANCES,
) -> tuple[tuple[int, ...], float]:
    """对每个起点做最近邻 + 2-opt，返回最好的一条；完全确定性。"""
    return min(
        (
            two_opt(nearest_neighbor_tour(start, distance_matrix), distance_matrix)
            for start in range(len(distance_matrix))
        ),
        key=lambda result: result[1],
    )


def _identity(gene: tuple[int, ...]) -> tuple[int, ...]:
    return gene


def _refine(gene: tuple[int, ...]) -> tuple[int, ...]:
    return two_opt(gene)[0]


class TspGA(GeneticAlgorithm[tuple[int, ...]]):
    """求解内置 25 城 TSP 的遗传算法。

    ``memetic=True``（默认）会在每个新个体评估前跑一次 2-opt，
    这样几十毫秒就能稳定拿到已知最优解；``memetic=False`` 是纯遗传算法，
    留给对照实验用。
    """

    def __init__(self, config: GAConfig | None = None, *, memetic: bool = True) -> None:
        super().__init__(
            config=config or GAConfig(),
            gene_factory=self._gene_factory,
            fitness_function=tsp_fitness,
            crossover_function=ordered_crossover,
            mutation_function=inversion_mutation,
            refine_function=_refine if memetic else None,
            clone_function=_identity,
        )

    @staticmethod
    def _gene_factory(rng: random.Random) -> tuple[int, ...]:
        return generate_permutation(CITY_COUNT, rng)
