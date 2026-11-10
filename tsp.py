"""25 个城市、城市间距离，以及把路线理顺的 2-opt。"""

from math import dist

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

# 独立用整数规划验证过的最优解。
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

DISTANCES = tuple(tuple(dist(first, second) for second in CITY_POS) for first in CITY_POS)


def route_length(route: tuple[int, ...]) -> float:
    """走完一圈的总距离。"""
    total = 0.0
    previous = route[-1]
    for city in route:
        total += DISTANCES[previous][city]
        previous = city
    return total


def two_opt(route: tuple[int, ...]) -> tuple[int, ...]:
    """把交叉的两条边换开，直到换不动为止。"""
    best = list(route)
    count = len(best)
    improved = True
    while improved:
        improved = False
        for head in range(count - 1):
            first, second = best[head], best[head + 1]
            distances = DISTANCES[first]
            base = distances[second]
            for tail in range(head + 2, count):
                if head == 0 and tail == count - 1:
                    continue
                tail_city = best[tail]
                closing_city = best[(tail + 1) % count]
                gain = (
                    distances[tail_city]
                    + DISTANCES[second][closing_city]
                    - base
                    - DISTANCES[tail_city][closing_city]
                )
                if gain < -1e-12:
                    best[head + 1 : tail + 1] = reversed(best[head + 1 : tail + 1])
                    second = best[head + 1]
                    base = distances[second]
                    improved = True
    return tuple(best)
