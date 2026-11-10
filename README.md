# 遗传算法求解 25 城 TSP

一个可复用的遗传算法引擎，外加一个内置的 25 城 TSP 例子。默认 0.1 秒内稳定给出已知最优解；
`--pure` 关掉局部搜索，能看清纯遗传算法和最优解之间差多少。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

WSL / Linux 把 `.venv\Scripts\` 换成 `.venv/bin/`。只需要运行时依赖的话去掉 `[dev]`。

## 跑

```powershell
.venv\Scripts\ga-tsp
```

```
最优路线    1,14,6,5,13,11,19,7,9,8,10,24,20,12,16,22,17,3,18,15,2,0,21,4,23
最优距离    4.267835351807
已知最优    4.267835351807
最优差距    0.000000%
基线距离    4.267835351807
耗时        0.072 秒
路线图      ...\output\route.png
适应度图    ...\output\fitness.png
```

| 命令 | 说明 |
|---|---|
| `ga-tsp` | 默认跑法：每个新个体先 2-opt 再打分 |
| `ga-tsp --pure` | 纯遗传算法，不做局部改进 |
| `ga-tsp --runs 30` | 30 个连续种子的统计，不画图 |
| `ga-tsp --pure --runs 30` | 纯遗传算法的统计 |
| `ga-tsp --no-plot` | 只打印，不碰 matplotlib |
| `ga-tsp --show` | 存完图再打开窗口 |
| `ga-tsp --progress` | 每代打印一次当前最优 |

`--seed`、`--population-size`、`--generations`、`--mutation-probability`、`--tournament-size`、
`--elite-count`、`--output-dir` 都可以改。两种跑法的默认规模不一样：默认 30×30，`--pure` 80×300，
因为纯遗传算法需要更多代数才看得出效果。

没装包时也能直接跑：`python -m ga_tsp`。

## 实测

本机 Windows / Python 3.11，每个数都是由下面的命令产出的：

| 跑法 | 单次耗时 | 平均差距 | 命中最优 | 最差 |
|---|---:|---:|---:|---:|
| 默认（带 2-opt），30 次 | 0.07 s | 0% | 30/30 | 4.267835 |
| `--pure`（纯 GA），30 次 | 0.20 s | 3.36% | 4/30 | 4.578020 |
| 基线（最近邻 + 2-opt） | 0.005 s | 0% | 确定性 | 4.267835 |

已知最优 `4.267835351807` 是用独立的混合整数规划模型求到零 MIP gap 得到的，
`tests/test_tsp.py` 会核对它和内置路线一致。

## 为什么默认让 2-opt 参与

反转变异（inversion）本身就是一次 2-opt 邻域移动。纯遗传算法跑了 300 代之后，最优解已经是
2-opt 局部最优，所以“跑完再补 2-opt”不会让结果变好哪怕一点。默认改为每代对新个体做一次
2-opt 精炼，等于把它变成模因算法：交叉负责提供好起点，2-opt 负责爬坡。这是同一个搜索预算下
唯一能稳定命中最优的组合。

25 城这个规模，真正干活的是 2-opt。想要更快，把 `--generations` 降到 10 也能 25/25 命中最优；
默认留了富余，只是为了让适应度曲线有东西可看。

## 当成库用

`GeneticAlgorithm` 与 TSP 无关，换目标函数只需要给四个策略：造初始解、交叉、变异、打分。
这里在整数基因上最大化 `-(x - 37)^2`：

```python
import random

from ga_tsp import GAConfig, GeneticAlgorithm

TARGET = 37


def crossover(first: int, second: int, rng: random.Random) -> tuple[int, int]:
    midpoint = round((first + second) / 2)
    return midpoint, rng.choice((first, second))


def mutate(value: int, rng: random.Random, probability: float) -> int:
    if rng.random() >= probability or value == TARGET:
        return value
    return value + 1 if value < TARGET else value - 1


solver = GeneticAlgorithm[int](
    config=GAConfig(minimize=False, mutation_probability=1, seed=0),
    gene_factory=lambda rng: rng.randrange(0, 75),
    fitness_function=lambda value: -float((value - TARGET) ** 2),
    crossover_function=crossover,
    mutation_function=mutate,
)
print(solver.train())
```

可选地传 `refine_function=...`，就是上面那种“每个新个体做一次局部改进”的模因模式。
`train(progress_callback)` 可以拿到每代进度；`solver.best_history` 和 `solver.mean_history`
记录每代的历史最优和种群平均。给定同一个 `seed`，两次 `train()` 的结果完全一致。

## 目录

```
ga_tsp/ga.py          引擎：GAConfig、Individual、GeneticAlgorithm
ga_tsp/operators.py   初始种群、顺序交叉、反转变异
ga_tsp/tsp.py         25 城数据、距离矩阵、fitness、最近邻 + 2-opt、TspGA
ga_tsp/experiment.py  多种子统计
ga_tsp/cli.py         ga-tsp 命令
ga_tsp/plot.py        路线图和适应度曲线
tests/                引擎、TSP、命令行的测试
```

## 本地检查

这个库没有远程 CI，全部在本地跑。Windows 和 WSL 各来一遍：

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\mypy
.venv\Scripts\python -m pytest --cov
```

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
.venv/bin/python -m pytest --cov
```

要求 `mypy --strict` 和分支覆盖率 95% 都过。

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。
