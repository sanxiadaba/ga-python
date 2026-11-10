"""遗传算法：随机生成一堆路线，每代挑好的、交叉、偶尔倒转一段、淘汰差的。"""

import random
from dataclasses import dataclass
from statistics import fmean

from tsp import CITY_COUNT, route_length, two_opt


@dataclass
class Result:
    """一次运行的结果。"""

    route: tuple[int, ...]
    distance: float
    best_history: list[float]
    mean_history: list[float]


def _child(keep: tuple[int, ...], fill: tuple[int, ...], start: int, end: int) -> tuple[int, ...]:
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


def solve(
    *,
    population_size: int = 30,
    generations: int = 30,
    mutation_probability: float = 0.25,
    tournament_size: int = 3,
    elite_count: int = 2,
    seed: int = 0,
    polish: bool = True,
) -> Result:
    """跑一遍遗传算法。polish=True 时每个新路线先做一次 2-opt 再比较。"""
    rng = random.Random(seed)

    def evaluate(route: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
        if polish:
            route = two_opt(route)
        return route, route_length(route)

    def random_route() -> tuple[int, ...]:
        route = list(range(CITY_COUNT))
        rng.shuffle(route)
        return tuple(route)

    def crossover(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        start, end = sorted(rng.sample(range(CITY_COUNT), 2))
        return _child(first, second, start, end), _child(second, first, start, end)

    def mutate(route: tuple[int, ...]) -> tuple[int, ...]:
        if rng.random() >= mutation_probability:
            return route
        start, end = sorted(rng.sample(range(CITY_COUNT), 2))
        return route[:start] + route[start : end + 1][::-1] + route[end + 1 :]

    population = [evaluate(random_route()) for _ in range(population_size)]
    best = min(population, key=lambda item: item[1])
    best_history: list[float] = []
    mean_history: list[float] = []

    def pick_parent() -> tuple[int, ...]:
        contestants = rng.sample(range(len(population)), tournament_size)
        return min((population[index] for index in contestants), key=lambda item: item[1])[0]

    for _ in range(generations):
        population.sort(key=lambda item: item[1])
        next_population = population[:elite_count]
        while len(next_population) < population_size:
            for route in crossover(pick_parent(), pick_parent()):
                if len(next_population) == population_size:
                    break
                next_population.append(evaluate(mutate(route)))
        population = next_population

        current = min(population, key=lambda item: item[1])
        if current[1] < best[1]:
            best = current
        best_history.append(best[1])
        mean_history.append(fmean(item[1] for item in population))

    return Result(best[0], best[1], best_history, mean_history)
