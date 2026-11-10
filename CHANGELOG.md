# Changelog

## 2.0.0 - 2026-11-10

### Changed (breaking)

- Modules moved from the repository root into the `ga_tsp` package. `from ga import ...` becomes
  `from ga_tsp import ...`, and `python -m main` becomes `python -m ga_tsp.main`. The flat modules
  are gone, so the distribution no longer shadows generic names such as `main`, `plot`, or
  `utility` in `site-packages`.
- The distribution now ships `py.typed` (PEP 561) so downstream type checkers can see the
  annotations.
- Build metadata uses the PEP 639 license form and requires `setuptools>=77`.

### Added

- `--hybrid` opt-in comparison that finishes the GA route with 2-opt, printed next to the
  untouched GA result.
- `base_seed=<n>` in experiment reports, so an unseeded run (`GAConfig(seed=None)`) can be
  replayed exactly.
- Statement/branch coverage gate at 95% (currently 100%) and `mypy --strict` over the package.
- CI now covers Python 3.10-3.14 on Ubuntu and Windows, runs `ruff format --check`, and has a
  separate job that builds a wheel and runs both console scripts from it.
- Chinese README (`README.zh-CN.md`) and this changelog.

### Fixed

- `--no-plot` no longer imports Matplotlib, so the solver runs where it is missing and startup
  drops from ~3.0 s to ~2.1 s.
- `--show` and `--no-plot` are mutually exclusive instead of silently ignoring one of them.
- Figures are closed after each plotting call instead of accumulating in the global pyplot state.
- The README experiment table was refreshed with measured values; its mean and worst rows had
  drifted from what the code produces.
- `experiment.run_experiment` uses `dataclasses.replace`, so a new `GAConfig` field can no longer
  be dropped silently when building per-seed configs.

## 1.0.0 - 2026-07-27

- Reproducible rewrite of the original learning project: generic strategy API, invariant tests,
  independently verified optimum, deterministic nearest-neighbor + 2-opt baseline, aggregate
  experiment reporting, and a documented before/after comparison.
