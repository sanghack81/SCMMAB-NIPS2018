# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-27

Modernization of the original NeurIPS-2018 release so it runs cleanly on current
scientific-Python stacks, plus a unit-test suite. No changes to the published
algorithms — POMIS results are unchanged (cross-checked against the brute-force
oracle).

### Changed
- Require **Python >= 3.11** (the floor imposed by the current `numpy`/`scipy`/`networkx` stack); migrated packaging from `distutils` to `setuptools` (`find_packages`, `install_requires`).
- Modernized type hints to PEP 585 / PEP 604 builtins (`dict`/`list`/`frozenset`/…, `X | None`).
- Figure scripts now fall back to mathtext (with a warning) when LaTeX is unavailable instead of crashing at render time; `seaborn.set` → `seaborn.set_theme`.

### Fixed
- `rand_argmax` no longer crashes on all-NaN input — the `is np.nan` guard never fired; it now uses `np.isnan`.
- Import `beta` from the public `numpy.random` instead of the private `numpy.random.mtrand`.
- Corrected implicit-`Optional` annotations on `CausalDiagram.__init__` and `StructuralCausalModel.query`.

### Added
- A `pytest` unit-test suite (99 tests) under `tests/`, installable via the `[test]` extra. It cross-checks the POMIS algorithm against a brute-force oracle, verifies the paper's optimality claim, and guards the bandit algorithms (including a regression test for the `rand_argmax` fix).
- Test-running instructions in the README.

### Removed
- Dead code: the unused `theta` initialization in `thompson_sampling`, an unused loop variable, and redundant `lambda`s.

## [0.1.0] - 2021-09-12

- Original code release accompanying the NeurIPS-2018 paper *Structural Causal Bandits: Where to Intervene?* (Lee & Bareinboim).

[0.2.0]: https://github.com/sanghack81/SCMMAB-NIPS2018/compare/c15bfa6...v0.2.0
