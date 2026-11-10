"""画两张图：最短路线，和每一代的距离变化。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ga import Result
from tsp import CITY_POS, KNOWN_OPTIMAL_LENGTH

# 有中文字体就写中文，没有就退回默认字体。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

ROUTE_FILENAME = "route.png"
FITNESS_FILENAME = "fitness.png"


def _route_figure(route: tuple[int, ...]) -> Figure:
    closed = (*route, route[0])
    figure, axis = plt.subplots()
    axis.plot(
        [CITY_POS[index][0] for index in closed],
        [CITY_POS[index][1] for index in closed],
        "o-",
        color="tab:red",
        linewidth=1.2,
        markersize=5,
    )
    axis.set(title="最短路线")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    return figure


def _fitness_figure(result: Result) -> Figure:
    generations = range(1, len(result.best_history) + 1)
    figure, axis = plt.subplots()
    axis.plot(generations, result.best_history, color="tab:blue", label="历史最优")
    axis.plot(generations, result.mean_history, color="tab:orange", alpha=0.7, label="每代平均")
    axis.axhline(
        KNOWN_OPTIMAL_LENGTH, color="tab:green", linestyle="--", linewidth=1, label="标准答案"
    )
    axis.set(title="每一代的最短距离", xlabel="代")
    axis.legend()
    axis.grid(alpha=0.25)
    return figure


def save_plots(result: Result, output_dir: Path, *, show: bool = False) -> tuple[Path, Path]:
    """存两张图，返回路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    route_figure = _route_figure(result.route)
    fitness_figure = _fitness_figure(result)
    route_path = output_dir / ROUTE_FILENAME
    fitness_path = output_dir / FITNESS_FILENAME
    try:
        route_figure.savefig(route_path, dpi=160, bbox_inches="tight")
        fitness_figure.savefig(fitness_path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(route_figure)
        plt.close(fitness_figure)
    return route_path, fitness_path
