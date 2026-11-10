"""TSP 用的三个算子。排列约束由 fitness 把关，这里不做校验。"""

import random
from collections.abc import Sequence


def generate_permutation(length: int, rng: random.Random) -> tuple[int, ...]:
    """生成 0..length-1 的随机排列。"""
    if length < 2:
        raise ValueError("城市数至少为 2")
    gene = list(range(length))
    rng.shuffle(gene)
    return tuple(gene)


def _ordered_child(
    segment_parent: Sequence[int], fill_parent: Sequence[int], start: int, end: int
) -> tuple[int, ...]:
    """保留一段，其余按另一个父代的顺序补齐。"""
    length = len(segment_parent)
    child = [0] * length
    child[start : end + 1] = segment_parent[start : end + 1]
    used = set(segment_parent[start : end + 1])
    position = end + 1
    for value in fill_parent[end + 1 :]:
        if value not in used:
            child[position] = value
            position += 1
    for value in fill_parent[: end + 1]:
        if value not in used:
            if position == length:
                position = 0
            child[position] = value
            position += 1
    return tuple(child)


def ordered_crossover(
    first: Sequence[int], second: Sequence[int], rng: random.Random
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """顺序交叉：两个父代交换一段，子代仍是排列。"""
    if len(first) != len(second):
        raise ValueError("两个父代的长度必须相同")
    if len(first) < 2:
        raise ValueError("父代至少需要两个基因")
    start, end = sorted(rng.sample(range(len(first)), 2))
    return (
        _ordered_child(first, second, start, end),
        _ordered_child(second, first, start, end),
    )


def inversion_mutation(
    gene: Sequence[int], rng: random.Random, probability: float
) -> tuple[int, ...]:
    """按概率反转一段。"""
    values = tuple(gene)
    if len(values) < 2 or rng.random() >= probability:
        return values
    start, end = sorted(rng.sample(range(len(values)), 2))
    return values[:start] + values[start : end + 1][::-1] + values[end + 1 :]
