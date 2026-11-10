# Reproducible Genetic Algorithm for TSP

English | [简体中文](README.zh-CN.md)

This project uses a genetic algorithm (GA) to solve a fixed 25-city Euclidean traveling
salesperson problem. It ships as a typed package (`ga_tsp`) with deterministic runs, invariant
tests, a deterministic heuristic baseline, aggregate experiment reporting, and a generic
strategy API for other optimization problems.

## Scope

This is a worked example of *GA engineering*, not a TSP solver to reach for. On 25 cities the
deterministic nearest-neighbor + 2-opt baseline reaches the optimum in about half a second,
while the GA stops roughly 2.4% short after 300 generations. That gap is deliberate: the GA is
kept free of local search so its own search quality can be measured, and the baseline exists to
make the gap visible rather than hide it. `--hybrid` exists if you want to see the gap close.

## Requirements

- Python 3.10 to 3.14 (CI runs the full suite on Ubuntu and Windows for each)
- Matplotlib 3.8 or newer

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

For development and tests:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run the TSP solver

```powershell
.venv\Scripts\ga-tsp
```

The command uses a fixed seed by default, prints the route and its distance, compares it with
the independently verified optimum and the nearest-neighbor + 2-opt baseline, and writes plots
to `output/`. It exits normally after saving; add `--show` to open interactive plot windows.
`--no-plot` skips the plotting import entirely, so the solver still runs where Matplotlib is
missing.

Useful options:

```powershell
.venv\Scripts\ga-tsp --seed 4 --generations 300 --population-size 80
.venv\Scripts\ga-tsp --no-plot
.venv\Scripts\ga-tsp --progress
.venv\Scripts\ga-tsp --hybrid
```

`--hybrid` is an opt-in comparison. It finishes the GA route with 2-opt and prints
`hybrid_distance=` and `hybrid_optimality_gap_percent=` next to the untouched `best_distance=`,
so the default GA path and every number quoted below stay unchanged.

Without installing, run `python -m ga_tsp.main` from the repository root.

## Repeat the experiment

A single GA result is not enough to assess a stochastic algorithm. Run multiple consecutive
seeds and report the distribution:

```powershell
.venv\Scripts\ga-tsp-experiment --runs 30
```

The bundled route has a proven optimum of `4.267835351807`. An independent mixed-integer linear
programming model solved it to zero MIP gap. The deterministic nearest-neighbor + 2-opt baseline
also reaches that value.

For seeds 0-19 with the default GA settings:

| Metric | Before the rewrite | Current implementation |
|---|---:|---:|
| Best | 4.387226 | 4.267835 |
| Median | 4.759353 | 4.369096 |
| Mean | 4.782647 | 4.371189 |
| Mean gap from optimum | 12.06% | 2.42% |
| Worst | 5.544973 | 4.589487 |
| Optimal runs | 0/50 | 4/20 |

The run counts differ because the left column records the original 50-run audit and the right
column is the fixed implementation's 20-seed regression sample. The right column is the measured
output of `ga-tsp-experiment --runs 20 --seed 0` on this code; an earlier draft of this table
quoted a slightly better mean and worst that the current implementation does not reproduce, so
those two rows were refreshed rather than carried over. Use the experiment command to produce a
fresh report on your machine.

The baseline line in every report is deterministic, costs about half a second, and prints the
same optimum on every run. The report also prints `base_seed=<n>`, so an unseeded run
(`GAConfig(seed=None)`) can be replayed by passing that value back as `--seed`.

## Algorithm

- Ordered crossover preserves valid permutations.
- Inclusive inversion mutation always changes at least two positions when triggered.
- Tournament contestants are sampled without replacement.
- Elites preserve the best routes and make the recorded best fitness monotonic.
- The configured population size remains constant across generations.
- Genes and evaluated fitness values are stored together as immutable individuals.
- Every `train()` call resets all state and recreates its seeded random generator.
- Strategy callbacks receive cloned genes; immutable TSP tuples use a zero-copy clone strategy.

The clones at crossover, mutation, and evaluation are deliberate: a user-supplied operator may
mutate its input in place, and an `Individual` must never share mutable state with the
population. Routes are validated as permutations at the fitness boundary, so a broken operator
fails loudly instead of scoring garbage.

Pairwise city distances are precomputed once. The GA is intentionally kept separate from the
2-opt baseline so its quality can be evaluated rather than hidden by local search.

## Use the engine for another objective

`GeneticAlgorithm` accepts strategies for gene creation, fitness, crossover, and mutation. This
example maximizes `-(x - 37)^2` over integer genes:

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

## Package layout

- `ga_tsp/ga.py`: generic GA engine and the configured TSP solver
- `ga_tsp/cross.py`, `ga_tsp/mutate.py`, `ga_tsp/gen_gene.py`: genetic operators
- `ga_tsp/fitness.py`, `ga_tsp/utility.py`: validated route fitness and geometry helpers
- `ga_tsp/heuristics.py`: nearest-neighbor and 2-opt baselines
- `ga_tsp/experiment.py`: repeated-run statistics
- `ga_tsp/main.py`, `ga_tsp/plot.py`: command-line application and plotting
- `ga_tsp/py.typed`: PEP 561 marker, so downstream type checkers see the annotations
- `tests/`: unit, integration, and regression tests

## Test and lint

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\mypy
.venv\Scripts\pytest --cov
```

The suite covers operator invariants, invalid configuration, fixed population size, monotonic
best history, repeated training, generic function maximization, the verified optimum, multi-seed
reproducibility, CLI exit behavior, plot generation and figure cleanup, and package metadata.
Statement and branch coverage is gated at 95% (currently 100%) and `mypy --strict` runs over the
package. GitHub Actions runs the same checks on Python 3.10 through 3.14 on Ubuntu and Windows,
plus a job that builds a wheel and runs both console scripts from it.

## Upgrading from 1.x

1.x exposed flat top-level modules (`ga.py`, `main.py`, `plot.py`, ...). 2.0 moves them into the
`ga_tsp` package and removes the flat modules, so the distribution no longer shadows generic
names such as `main`, `plot`, or `utility` in `site-packages`.

| 1.x | 2.0 |
|---|---|
| `from ga import GAConfig, GeneticAlgorithm` | `from ga_tsp import GAConfig, GeneticAlgorithm` |
| `from heuristics import best_nearest_neighbor_two_opt` | `from ga_tsp.heuristics import best_nearest_neighbor_two_opt` |
| `python -m main` | `python -m ga_tsp.main` |
| `ga-tsp`, `ga-tsp-experiment` | unchanged |

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## License

Apache-2.0. See [LICENSE](LICENSE).

The original learning reference is <https://github.com/zifeiyu0531/ga-tsp>.
