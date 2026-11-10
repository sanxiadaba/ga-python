# 可复现的 TSP 遗传算法

[English](README.md) | 简体中文

本项目用遗传算法（GA）求解一个固定的 25 城市欧氏旅行商问题。它以带类型标注的 `ga_tsp`
包发布，包含确定性运行、不变量测试、确定性启发式基线、多种子实验统计，以及可复用到其他
优化问题的通用策略 API。

## 定位

这是一个**遗传算法工程示例**，不是一个拿来解 TSP 的工具。25 个城市上，确定性的
最近邻 + 2-opt 基线半秒内就命中最优解，而 GA 跑 300 代仍差约 2.4%。这个差距是刻意的：GA
里不掺局部搜索，才能量出它自己的搜索质量；基线的作用是让差距可见，而不是把它藏起来。
想看差距如何被抹平，可以用可选的 `--hybrid`。

## 环境要求

- Python 3.10 到 3.14（CI 在 Ubuntu 与 Windows 上逐版本跑全量测试）
- Matplotlib 3.8 或更高

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

开发与测试环境：

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
```

## 运行求解器

```powershell
.venv\Scripts\ga-tsp
```

命令默认使用固定种子，打印路线与总长度，与已独立验证的最优解和最近邻 + 2-opt 基线对比，
并把图写入 `output/`；保存完成后正常退出，加 `--show` 可打开交互窗口。
`--no-plot` 完全不导入绘图模块，所以即使机器上没有 Matplotlib 也能跑求解部分。

常用参数：

```powershell
.venv\Scripts\ga-tsp --seed 4 --generations 300 --population-size 80
.venv\Scripts\ga-tsp --no-plot
.venv\Scripts\ga-tsp --progress
.venv\Scripts\ga-tsp --hybrid
```

`--hybrid` 是可选的对照：把 GA 的结果再用 2-opt 收尾，额外打印 `hybrid_distance=` 与
`hybrid_optimality_gap_percent=`，而原来的 `best_distance=` 保持不动。也就是说默认的 GA
路径和下文所有数字都不受影响。

不安装也可以：在仓库根目录用 `python -m ga_tsp.main`。

## 复现实验

单次 GA 结果不足以评价随机算法，需要跑连续种子并统计分布：

```powershell
.venv\Scripts\ga-tsp-experiment --runs 30
```

内置路线有已证明的最优值 `4.267835351807`：一个独立的混合整数线性规划模型以零 MIP 间隙
求解，确定性的最近邻 + 2-opt 基线同样到达该值。

默认参数下种子 0–19 的结果：

| 指标 | 重写前 | 当前实现 |
|---|---:|---:|
| 最优 | 4.387226 | 4.267835 |
| 中位数 | 4.759353 | 4.369096 |
| 均值 | 4.782647 | 4.371189 |
| 均值相对最优的差距 | 12.06% | 2.42% |
| 最差 | 5.544973 | 4.589487 |
| 命中最优次数 | 0/50 | 4/20 |

两列的运行次数不同：左侧是原项目 50 次的审计记录，右侧是修复后实现的 20 种子回归样本。
右侧是本代码 `ga-tsp-experiment --runs 20 --seed 0` 的实测输出；本表早期草稿引用过一组略好
的均值与最差值，但当前实现复现不出来，因此那两行是重新测的，不是照抄的。想在自己的机器上
得到新报告，直接跑上面的命令。

每次报告里的基线那行都是确定性的：耗时约半秒，每次打印同一个最优值。报告还会打印
`base_seed=<n>`，所以未指定种子的运行（`GAConfig(seed=None)`）可以用这个值作为 `--seed`
重新回放。

## 算法要点

- 顺序交叉（OX）保证子代仍是合法排列。
- 触发时的反转变异覆盖闭区间，且至少改动两个位置。
- 锦标赛的参赛个体无放回抽样。
- 精英保留让记录的最优适应度单调不增（最小化问题）。
- 每代种群规模恒等于配置值。
- 基因与评估后的适应度一起存为不可变个体。
- 每次 `train()` 都会重置全部状态并重建带种子的随机数生成器。
- 策略回调收到的是克隆后的基因；TSP 的不可变元组采用零拷贝克隆策略。

交叉前、变异前、评估前各克隆一次是刻意设计：用户提供的算子可能原地修改输入，而
`Individual` 绝不能与种群共享可变状态。路线在适应度边界被校验为排列，算子出错时会立刻
报错，而不是把垃圾解算出分数。

两两城市距离只预计算一次。GA 与 2-opt 基线刻意分离，这样评价的是 GA 本身的质量，而不是
被局部搜索掩盖后的质量。

## 用这个引擎解别的问题

`GeneticAlgorithm` 接受基因生成、适应度、交叉、变异四个策略。下面的例子在整数基因上最大化
`-(x - 37)^2`：

```python
import random

from ga_tsp import GAConfig, GeneticAlgorithm

target = 37


def crossover(first: int, second: int, rng: random.Random) -> tuple[int, int]:
    midpoint = round((first + second) / 2)
    return midpoint, rng.choice((first, second))


def mutate(value: int, rng: random.Random, probability: float) -> int:
    if rng.random() >= probability or value == target:
        return value
    return value + 1 if value < target else value - 1


solver = GeneticAlgorithm[int](
    config=GAConfig(minimize=False, mutation_probability=1, seed=0),
    gene_factory=lambda rng: rng.randrange(0, 75),
    fitness_function=lambda value: -float((value - target) ** 2),
    crossover_function=crossover,
    mutation_function=mutate,
)
print(solver.train())
```

## 目录结构

- `ga_tsp/ga.py`：通用 GA 引擎与配置好的 TSP 求解器
- `ga_tsp/cross.py`、`ga_tsp/mutate.py`、`ga_tsp/gen_gene.py`：遗传算子
- `ga_tsp/fitness.py`、`ga_tsp/utility.py`：带校验的路线适应度与几何工具
- `ga_tsp/heuristics.py`：最近邻与 2-opt 基线
- `ga_tsp/experiment.py`：多次运行的统计
- `ga_tsp/main.py`、`ga_tsp/plot.py`：命令行应用与绘图
- `ga_tsp/py.typed`：PEP 561 标记，下游类型检查器能看到标注
- `tests/`：单元、集成与回归测试

## 测试与静态检查

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\mypy
.venv\Scripts\pytest --cov
```

测试覆盖算子不变量、非法配置、种群规模恒定、最优历史单调、重复训练、通用函数最大化、已
验证的最优解、多种子可复现性、CLI 退出行为、绘图生成与 figure 释放、包元数据。语句与分支
覆盖率门禁为 95%（当前 100%），`mypy --strict` 覆盖整个包。GitHub Actions 在 Python
3.10–3.14、Ubuntu 与 Windows 上跑同一套检查，另有一个 job 构建 wheel 并用其中的两个命令行
入口实跑。

## 从 1.x 升级

1.x 把模块平铺在仓库顶层（`ga.py`、`main.py`、`plot.py` 等）。2.0 把它们收进 `ga_tsp`
包并删除了顶层模块，因此安装后不会再覆盖 `main`、`plot`、`utility` 这类通用名字。

| 1.x | 2.0 |
|---|---|
| `from ga import GAConfig, GeneticAlgorithm` | `from ga_tsp import GAConfig, GeneticAlgorithm` |
| `from heuristics import best_nearest_neighbor_two_opt` | `from ga_tsp.heuristics import best_nearest_neighbor_two_opt` |
| `python -m main` | `python -m ga_tsp.main` |
| `ga-tsp`、`ga-tsp-experiment` | 不变 |

完整变更见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。

最初的学习参考项目：<https://github.com/zifeiyu0531/ga-tsp>。
