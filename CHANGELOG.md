# Changelog

All notable iterations of this project.

## v4.0 — Full Portfolio Release (current)

### Deployment
- **Streamlit interactive demo** with glass-morphism UI, multi-accent color palette
  (coral, electric blue, mint, amber, lavender, rose), and aurora-gradient background.
- Deployed to **HuggingFace Spaces** and **Streamlit Cloud** (dual hosting).
- App serves **precomputed aggregated profiles** (2.7 MB) instead of raw data —
  no competition dataset exposed publicly.
- UptimeRobot keyword monitoring keeps both deployments awake.

### Analysis notebooks
- **Error analysis** — residual breakdown by hour, road type, weather, and geohash
  region; identifies peak-hour under-prediction as the primary failure mode.
- **Ablation study** — quantifies each feature group's marginal R² contribution;
  confirms geohash×time target encodings as the dominant signal.
- **Geospatial visualization** — 5 dark-themed spatial plots: demand heatmap,
  hourly evolution (6 panels), K-means clusters, spatial-spillover comparison,
  decoded location grid.
- **Model comparison** — head-to-head R² of all 5 model families, OOF prediction
  correlation matrix, exhaustive search over all 63 model subsets.
- **Validation integrity** — side-by-side comparison of random CV (R²=0.99) vs
  forward holdout (R²=0.76), quantifying the 23-point temporal leakage gap.

### Engineering
- **CI/CD** — GitHub Actions smoke test (synthetic data → pipeline → validation)
  with green build badge.
- **Docker** — containerized execution via Dockerfile.
- **Makefile** — one-command targets: `make train`, `make test`, `make lint`.
- **Pre-commit** — ruff linting + auto-formatting on every commit.
- **Data validation** — schema checks (types, ranges, uniqueness, format) via
  `tests/validate_data.py`.
- **Type hints + docstrings** on all key functions in `src/`.

### Documentation
- **APPROACH.md** — 176-line comprehensive methodology covering all components.
- **CONTRIBUTING.md** — fork/branch/test/PR development workflow.
- **Issue templates** — structured bug report and feature request forms.
- **INTERVIEW_PREP.md** — resume bullets, interview Q&A, STAR stories, tech stack.
- **README** — 310 lines, 10 badges, 11 embedded visualizations, dual demo links.

### Visual assets
- 3 SVG diagrams: pipeline architecture, results chart, feature importance.
- 14 PNG charts: geospatial (5), error analysis (4), ablation (1), model
  comparison (2), validation comparison (1), exhaustive combo search (1).

## v3.0 — Stacked Ensemble
- **5-model stacked ensemble** (LightGBM + XGBoost + CatBoost + HGBR + ExtraTrees)
  with a Ridge meta-learner over out-of-fold predictions.
- Added **spatial-neighbour demand** feature (k-nearest geohash neighbours).
- Accuracy: **97.85**

## v2.0 — Single GBM + Feature Engineering
- Rich feature set: geohash×time target encodings, denoised day-48 profile,
  cyclical time, K-means geo-clusters.
- Single LightGBM (fallback: HistGradientBoosting), 3 seeds × 5 folds, log space.
- Accuracy: ~90

## v1.0 — Baseline
- Persistence baseline (predict from the reference day's same-slot demand).
- Accuracy: ~80

## v0.1 — Initial EDA
- Identified the spatio-temporal structure (not independent rows).
- Geohash decoded to lat/lon; confirmed time-of-day profile as dominant signal.
