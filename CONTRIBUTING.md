# Contributing

Thanks for your interest in contributing!

## Quick start

```bash
git clone https://github.com/simply-mihir/traffic-demand-prediction.git
cd traffic-demand-prediction
pip install -r requirements.txt
pip install pre-commit && pre-commit install   # optional: auto-format on commit
make test                                       # verify everything works
```

## Development workflow

1. **Fork** the repo and create a feature branch: `git checkout -b feat/my-improvement`
2. Place `train.csv`, `test.csv`, `sample_submission.csv` in `data/` (git-ignored).
3. Make your changes — code in `src/`, notebooks in `notebooks/`.
4. Run `make test` to verify the pipeline still works.
5. Run `make lint` to check code style (requires `ruff` and `mypy`).
6. Commit, push, and open a **Pull Request**.

## Code style

- Python 3.9+, type hints encouraged.
- Formatting: `ruff format` (runs automatically via pre-commit).
- Linting: `ruff check` — no errors allowed on CI.

## Project structure

- `src/solution.py` — single-model pipeline (the entry point).
- `src/feature_engineering.py` — shared feature builder.
- `notebooks/` — experiments (EDA, ensemble, ablation, visualization).
- `tests/` — CI smoke test.

## Reporting issues

Open a GitHub issue with:
- What you expected vs. what happened.
- Steps to reproduce.
- Your Python version and OS.
