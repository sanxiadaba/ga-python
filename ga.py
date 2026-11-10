"""遗传算法求解 25 城 TSP。

每一代：留下最好的几条 → 挑两条交叉 → 偶尔倒转一段 → 组成新一代。
"""

import random
from dataclasses import dataclass
from statistics import fmean

from tsp import CITY_COUNT, route_length

Route = tuple[int, ...]


@dataclass
class Result:
    """一次运行的结果。"""

    route: Route
    distance: float
    best_history: list[float]
    mean_history: list[float]


def _child(keep: Route, fill: Route, start: int, end: int) -> Route:
    """保留 keep 的一段，其余按 fill 的顺序补齐。"""
    child = [0] * CITY_COUNT
    child[start : end + 1] = keep[start : end + 1]
    used = set(keep[start : end + 1])
    position = end + 1
    for value in fill[end + 1 :]:
        if value not in used:
            child[position] = value
            position += 1
    for value in fill[: end + 1]:
        if value not in used:
            if position == CITY_COUNT:
                position = 0
            child[position] = value
            position += 1
    return tuple(child)


def ordered_crossover(first: Route, second: Route, rng: random.Random) -> tuple[Route, Route]:
    """顺序交叉：两个父代各交出一段，子代仍是每个城市各出现一次。"""
    start, end = sorted(rng.sample(range(CITY_COUNT), 2))
    return _child(first, second, start, end), _child(second, first, start, end)


def inversion_mutation(route: Route, rng: random.Random, probability: float) -> Route:
    """按概率把路线里的一段顺序倒过来。"""
    if rng.random() >= probability:
        return route
    start, end = sorted(rng.sample(range(CITY_COUNT), 2))
    return route[:start] + route[start : end + 1][::-1] + route[end + 1 :]


def random_route(rng: random.Random) -> Route:
    route = list(range(CITY_COUNT))
    rng.shuffle(route)
    return tuple(route)


def _evaluate(route: Route) -> tuple[Route, float]:
    return route, route_length(route)


def solve(
    *,
    population_size: int = 80,
    generations: int = 300,
    mutation_probability: float = 0.25,
    tournament_size: int = 3,
    elite_count: int = 2,
    seed: int = 0,
) -> Result:
    """跑一遍遗传算法。"""
    rng = random.Random(seed)
    population = [_evaluate(random_route(rng)) for _ in range(population_size)]
    best = min(population, key=lambda item: item[1])
    best_history: list[float] = []
    mean_history: list[float] = []

    def pick_parent() -> Route:
        contestants = rng.sample(range(len(population)), tournament_size)
        return min((population[index] for index in contestants), key=lambda item: item[1])[0]

    for _ in range(generations):
        population.sort(key=lambda item: item[1])
        next_population = population[:elite_count]
        while len(next_population) < population_size:
            for route in ordered_crossover(pick_parent(), pick_parent(), rng):
                if len(next_population) == population_size:
                    break
                next_population.append(
                    _evaluate(inversion_mutation(route, rng, mutation_probability))
                )
        population = next_population

        current = min(population, key=lambda item: item[1])
        if current[1] < best[1]:
            best = current
        best_history.append(best[1])
        mean_history.append(fmean(item[1] for item in population))

    return Result(best[0], best[1], best_history, mean_history)
