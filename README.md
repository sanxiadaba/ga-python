# 遗传算法求解 25 城 TSP

内置一个 25 城 TSP，用遗传算法求最短路线，默认 0.1 秒内找到已知最优解。
`--pure` 关掉 2-opt，可以看到纯遗传算法差多少。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

WSL / Linux 把 `.venv\Scripts\` 换成 `.venv/bin/`。

## 跑

```powershell
.venv\Scripts\ga-tsp
```

```
最优路线    1,14,6,...,4,23
最优距离    4.267835351807
已知最优    4.267835351807
最优差距    0.000000%
基线距离    4.267835351807
耗时        0.072 秒
路线图      ...\output\route.png
适应度图    ...\output\fitness.png
```

常用参数：`--pure` 纯遗传算法，`--runs 30` 多种子统计（不画图），`--no-plot` 不画图，
`--show` 存完开窗口，`--progress` 每代打印。其余看 `ga-tsp --help`。
没装包时用 `python -m ga_tsp`。

## 实测

| 跑法 | 单次耗时 | 平均差距 | 命中最优 |
|---|---:|---:|---:|
| 默认（每个新解先做一次 2-opt），30 次 | 0.07 秒 | 0 | 30/30 |
| `--pure`（纯 GA），30 次 | 0.20 秒 | 3.36% | 4/30 |
| 基线（每个起点做最近邻 + 2-opt） | 0.005 秒 | 0 | 确定性 |

反转变异本身就是一次 2-opt，纯 GA 跑完已经是 2-opt 局部最优，再补也好不了。
让 2-opt 在每代介入才能稳定命中最优。

## 当库用

`GeneticAlgorithm` 和 TSP 无关，换目标只要给四个函数。下面最大化 `-(x - 37)^2`：

```python
from ga_tsp import GAConfig, GeneticAlgorithm


def mutate(x: int, rng, probability: float) -> int:
    if rng.random() >= probability or x == 37:
        return x
    return x + 1 if x < 37 else x - 1


solver = GeneticAlgorithm[int](
    config=GAConfig(minimize=False, mutation_probability=1, seed=0),
    gene_factory=lambda rng: rng.randrange(75),
    fitness_function=lambda x: -float((x - 37) ** 2),
    crossover_function=lambda a, b, rng: (round((a + b) / 2),) * 2,
    mutation_function=mutate,
)
print(solver.train())  # Individual(gene=37, ...)
```

`best_history` / `mean_history` 是每代的历史最优和种群平均；传 `refine_function`
就是上面说的 memetic。

## 检查

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\mypy
.venv\Scripts\python -m pytest --cov
```
