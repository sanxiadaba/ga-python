"""跑一次：python main.py"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from unicodedata import east_asian_width

from ga import solve
from tsp import KNOWN_OPTIMAL_LENGTH

OUTPUT_DIR = Path("output")
LABEL_WIDTH = 12


def _display_width(text: str) -> int:
    return sum(2 if east_asian_width(char) in "WF" else 1 for char in text)


def _row(label: str, value: str) -> str:
    return f"{label}{' ' * max(1, LABEL_WIDTH - _display_width(label))}{value}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用遗传算法求 25 城 TSP 的最短路线")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--population-size", type=int, default=80, help="每代多少条路线")
    parser.add_argument("--generations", type=int, default=300, help="迭代多少代")
    parser.add_argument("--no-plot", action="store_true", help="不画图")
    parser.add_argument("--show", action="store_true", help="存完图打开窗口")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.population_size < 2:
        parser.error("每代至少要 2 条路线")
    if args.generations < 1:
        parser.error("至少要迭代 1 代")

    started = time.perf_counter()
    result = solve(
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started

    gap = (result.distance / KNOWN_OPTIMAL_LENGTH - 1) * 100
    lines = [
        _row("最优路线", ",".join(map(str, result.route))),
        _row("最优距离", f"{result.distance:.12f}"),
        _row("已知最优", f"{KNOWN_OPTIMAL_LENGTH:.12f}"),
        _row("最优差距", f"{gap:.6f}%"),
        _row("耗时", f"{elapsed:.3f} 秒"),
    ]

    if not args.no_plot:
        try:
            from plot import save_plots
        except ModuleNotFoundError as error:
            raise SystemExit("缺少 matplotlib，先运行：pip install matplotlib") from error
        route_path, fitness_path = save_plots(result, OUTPUT_DIR, show=args.show)
        lines.append(_row("路线图", str(route_path.resolve())))
        lines.append(_row("适应度图", str(fitness_path.resolve())))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
