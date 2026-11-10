"""ga-tsp 命令行。"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from unicodedata import east_asian_width

from .experiment import ExperimentSummary, run_experiment
from .ga import GAConfig, Individual
from .tsp import KNOWN_OPTIMAL_LENGTH, TspGA, baseline_tour

# 纯 GA 要跑更多代才看得出效果。
MEMETIC_DEFAULTS = {"population_size": 30, "generations": 30}
PURE_DEFAULTS = {"population_size": 80, "generations": 300}
LABEL_WIDTH = 12


def _display_width(text: str) -> int:
    return sum(2 if east_asian_width(char) in "WF" else 1 for char in text)


def _row(label: str, value: str) -> str:
    return f"{label}{' ' * max(1, LABEL_WIDTH - _display_width(label))}{value}"


def _run_count(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("实验次数至少为 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ga-tsp",
        description="用遗传算法求解内置的 25 城 TSP，并和确定性基线对比。",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--population-size", type=int, default=None, help="种群规模")
    parser.add_argument("--generations", type=int, default=None, help="迭代代数")
    parser.add_argument(
        "--mutation-probability", type=float, default=0.25, help="每个子代的变异概率"
    )
    parser.add_argument("--tournament-size", type=int, default=3, help="锦标赛规模")
    parser.add_argument("--elite-count", type=int, default=2, help="每代保留的精英数")
    parser.add_argument(
        "--pure",
        action="store_true",
        help="关掉 2-opt，跑纯遗传算法",
    )
    parser.add_argument(
        "--runs", type=_run_count, default=None, help="跑 N 个连续种子并输出统计，此时不画图"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="图片输出目录")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--show", action="store_true", help="保存后打开图片窗口")
    output_group.add_argument("--no-plot", action="store_true", help="不生成图片")
    parser.add_argument("--progress", action="store_true", help="每代打印一次当前最优")
    return parser


def _build_config(
    parser: argparse.ArgumentParser, args: argparse.Namespace, memetic: bool
) -> GAConfig:
    defaults = MEMETIC_DEFAULTS if memetic else PURE_DEFAULTS
    population_size = (
        defaults["population_size"] if args.population_size is None else args.population_size
    )
    generations = defaults["generations"] if args.generations is None else args.generations
    try:
        return GAConfig(
            population_size=population_size,
            generations=generations,
            mutation_probability=args.mutation_probability,
            tournament_size=args.tournament_size,
            elite_count=args.elite_count,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))


def _progress(generation: int, best: Individual[tuple[int, ...]]) -> None:
    print(f"第 {generation:4d} 代  最优 {best.fitness:.12f}")


def _print_single(args: argparse.Namespace, config: GAConfig, memetic: bool) -> int:
    solver = TspGA(config, memetic=memetic)
    started = time.perf_counter()
    best = solver.train(_progress if args.progress else None)
    elapsed = time.perf_counter() - started
    _, baseline_distance = baseline_tour()
    gap = (best.fitness / KNOWN_OPTIMAL_LENGTH - 1) * 100
    lines = [
        _row("最优路线", ",".join(map(str, best.gene))),
        _row("最优距离", f"{best.fitness:.12f}"),
        _row("已知最优", f"{KNOWN_OPTIMAL_LENGTH:.12f}"),
        _row("最优差距", f"{gap:.6f}%"),
        _row("基线距离", f"{baseline_distance:.12f}"),
        _row("耗时", f"{elapsed:.3f} 秒"),
    ]
    if not args.no_plot:
        # 延迟导入，--no-plot 时不加载 matplotlib。
        from .plot import save_plots

        route_path, fitness_path = save_plots(solver, args.output_dir, show=args.show)
        lines.append(_row("路线图", str(route_path.resolve())))
        lines.append(_row("适应度图", str(fitness_path.resolve())))
    print("\n".join(lines))
    return 0


def _print_experiment(args: argparse.Namespace, config: GAConfig, memetic: bool, runs: int) -> int:
    started = time.perf_counter()
    summary: ExperimentSummary = run_experiment(runs, config, memetic=memetic)
    elapsed = time.perf_counter() - started
    _, baseline_distance = baseline_tour()
    count = len(summary.distances)
    mode = "memetic" if memetic else "纯遗传算法"
    lines = [
        _row(
            "实验次数",
            f"{count}（种子 {summary.base_seed}..{summary.base_seed + count - 1}，{mode}）",
        ),
        _row("最好", f"{summary.best:.12f}"),
        _row("中位", f"{summary.median:.12f}"),
        _row("平均", f"{summary.mean:.12f}"),
        _row("标准差", f"{summary.standard_deviation:.12f}"),
        _row("最差", f"{summary.worst:.12f}"),
        _row("平均差距", f"{summary.mean_gap_percent:.6f}%"),
        _row("命中最优", f"{summary.optimal_runs}/{count}"),
        _row("差距 5% 内", f"{summary.runs_within_percent(5)}/{count}"),
        _row("基线距离", f"{baseline_distance:.12f}"),
        _row("耗时", f"{elapsed:.3f} 秒"),
    ]
    print("\n".join(lines))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    memetic = not args.pure
    config = _build_config(parser, args, memetic)
    if args.runs is None:
        return _print_single(args, config, memetic)
    return _print_experiment(args, config, memetic, args.runs)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
