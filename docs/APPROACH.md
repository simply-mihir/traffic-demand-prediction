# Traffic Demand Prediction — Project Description

## The Problem

Predicting normalized traffic demand across a city's road network, aggregated into **geohash-encoded spatial cells** over **15-minute time buckets**. Given historical demand for ~1,250 locations across a reference day plus the early hours of the following day, the task is to forecast the remaining daytime demand for 41,778 location-time pairs. The evaluation metric is `accuracy = max(0, 100 × R²)`.

---

## What Makes This Hard

This is not a standard tabular regression problem — it is a **spatio-temporal time-series forecasting** task disguised as one. The rows are not independent: they form a continuous timeline across a fixed set of recurring locations. The test set is the immediate temporal continuation of the training set (day 49 daytime, following a training window of day 48 + day 49 early morning). This means that the dominant signal is each location's **time-of-day demand profile**, not the supplied context attributes (road type, weather, temperature), which carry comparatively little predictive power. Recognizing this structure — and building the solution around it rather than treating rows as i.i.d. — is the key insight that separates a baseline score of ~50 from a competitive score of ~97.

---

## Data Pipeline

**Source data:** 77,299 training records and 41,778 test records, each identified by a geohash6 location code, a sequential day index, and a timestamp (`H:M` on a 15-minute grid), with additional context columns (RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather). The target `demand` is a continuous value normalized to [0, 1]. The dataset originates from the Grab AI for S.E.A. 2019 Traffic Management challenge, with anonymised geohash coordinates (adjacency preserved) and synthetic attribute columns added.

**Data quality validation:** A dedicated schema-checking script (`tests/validate_data.py`) runs column-level assertions (types, ranges, uniqueness, format patterns) before any training, catching structural issues early. It verifies geohash length, demand bounds, timestamp format, and index integrity across train, test, and sample submission files.

---

## Feature Engineering

All features are built **jointly on train and test** to ensure consistent encoding. The feature matrix comprises 42 columns across five groups:

**Spatial features** — The 6-character geohash strings are decoded to latitude/longitude using a custom base-32 geocoder (no external geohash library dependency). Region prefixes at precision 4 and 5 are extracted as categorical features. A **K-means clustering** (k=30) on the decoded coordinates groups nearby locations into spatial clusters, enabling the model to share demand patterns across adjacent cells.

**Temporal features** — Minutes-of-day, hour, minute, and 15-minute slot index are extracted from the `H:M` timestamp strings. **Cyclical sin/cos encodings** of both the time-of-day and the hour ensure the model treats 23:45 and 00:00 as adjacent rather than maximally distant. Day index and day-of-week (`day % 7`) capture weekly periodicity.

**Demand profile features (the core signal)** — **Leak-free K-fold target encodings** compute the smoothed mean demand at multiple granularities: per-geohash, geohash×hour, geohash×slot, region(gh4)×hour, region(gh5)×hour, RoadType, Weather, geohash×RoadType, RoadType×hour, and Weather×hour. Out-of-fold computation on 5 folds with Bayesian smoothing (shrinkage toward the global mean) prevents target leakage while giving the model access to each location's historical demand rhythm. A **denoised reference-day time-of-day profile** per geohash (rolling-smoothed across neighboring time slots) and its lag/lead neighbours capture the shape of the daily demand curve. A per-geohash **recent demand level** from the latest available observations provides a recency signal.

**Spatial spillover features** — For each location, the mean demand of its **k=6 nearest geohash neighbours** at the same time slot is computed from the reference day. Since geohash adjacency is preserved (confirmed by the dataset FAQ and validated by the smooth spatial gradient in the neighbour-demand visualization), this captures local demand diffusion between adjacent road segments.

**Context features** — NumberofLanes and Temperature as numeric values; RoadType, Weather, LargeVehicles, and Landmarks as categorical codes. These are included but contribute modestly relative to the spatio-temporal features (confirmed by the ablation study).

---

## Modeling

**Single-model pipeline (`src/solution.py`):** A gradient-boosted decision tree regressor trained in **log space** (`log1p` transform of demand, exponentiated at prediction time) to handle the heavy right skew of the target distribution. The model is **LightGBM** when available, with an automatic fallback cascade to XGBoost → CatBoost → scikit-learn's HistGradientBoostingRegressor, so the pipeline runs in any environment without modification. Predictions are clipped to [0, 1] and **bagged over 3 random seeds × 5 folds** for stability.

**Stacked ensemble (`notebooks/02_stacked_ensemble.ipynb`):** The headline model. Five diverse model families are trained independently with **out-of-fold prediction** (5-fold):
- **LightGBM** — gradient boosting with native categorical support for geohash, RoadType, Weather, etc.
- **XGBoost** — histogram-based gradient boosting with categorical feature handling.
- **CatBoost** — ordered boosting with native categorical encoding, particularly effective on high-cardinality features like geohash.
- **HistGradientBoostingRegressor** (scikit-learn) — fast histogram-based gradient boosting, no external dependency.
- **ExtraTrees** (scikit-learn) — extremely randomized trees (bagging family, structurally different from boosting).

Each model's out-of-fold predictions on the training set and averaged predictions on the test set form a 5-column meta-feature matrix. A **Ridge regression meta-learner** with non-negative coefficients combines these into the final prediction, learning the optimal weighting of each model family. This two-level stacking achieves **97.85 accuracy** on the competition leaderboard.

---

## Validation Strategy

A **forward holdout** is used throughout: the model is trained on the reference day (day 48) and evaluated on the held-out next-day records (day 49 early slots). This mimics the true forecasting task — predicting future demand from past observations — and avoids the optimistic bias of random K-fold cross-validation on time-series data. Random-CV R² (~0.97) is reported for reference but never used for model selection; the forward-holdout R² (~0.76) is the trusted metric. The consistent gap between the two (and the observation that forward-holdout scores historically understated the leaderboard by ~13 points due to the night-vs-daytime regime difference) is itself an analytical finding documented in the approach.

---

## Analysis & Visualization

**Exploratory data analysis (`notebooks/01_eda.ipynb`):** Target distribution (right-skewed, near-zero-inflated), temporal coverage (train = 1 full day + early next morning; test = daytime continuation), and the discovery that demand follows location-specific daily profiles.

**Geospatial visualization (`notebooks/06_geospatial_visualization.ipynb`):** Five dark-themed spatial plots generated with matplotlib:
- Decoded geohash grid showing the urban road structure.
- Demand hotspot heatmap (PowerNorm-scaled) revealing stable high-demand clusters.
- Hourly spatial evolution (6-panel grid) showing how hotspots activate across the day.
- K-means geo-cluster map validating the spatial feature.
- Own-demand vs. neighbour-demand side-by-side confirming smooth spatial spillover (geohash adjacency preserved).

**Error analysis (`notebooks/04_error_analysis.ipynb`):** Residual analysis by hour of day (errors peak at high-demand hours due to under-prediction of spikes), by road type (rare types have higher MAE), by weather condition, and by geohash region (top-10 worst regions by MAE). Residual distribution shows near-zero bias with a right tail.

**Ablation study (`notebooks/05_ablation_study.ipynb`):** Quantifies each feature group's marginal contribution by training with progressively richer feature sets. Confirms that geohash×time target encodings provide the single largest lift, while context attributes add a modest but real improvement.

**Feature importance (`assets/feature_importance.svg`):** Real permutation importance from the trained model, showing the reference-day demand profile and geohash×time encodings dominate — confirming the spatio-temporal framing is justified.

---

## Interactive Demo (Streamlit)

A **Streamlit web application** (`app/streamlit_app.py`) deployed at `traffic-demand-prediction-simply-mihir.streamlit.app` provides an interactive demand prediction interface. Users select a geohash location, time of day, and context attributes (road type, weather, temperature, lanes); the app returns a demand prediction with the profile resolution level, decoded lat/lon coordinates, a 24-hour demand bar chart for that location, a map pin, and detailed location statistics.

The app loads **precomputed aggregated profiles** (2.7 MB of mean demand by geohash×slot, geohash×hour, and geohash-level statistics) rather than the raw training data, so no competition dataset is exposed publicly. The profiles are committed to the repository; the raw data stays local and git-ignored.

**UI design:** Custom CSS glass-morphism theme with a multi-accent color palette (coral, electric blue, mint, amber, lavender, rose), aurora-gradient background, translucent glass cards with hover animations, gradient-text metrics, glowing section dots, and responsive layout. Inter typeface. No emojis. Streamlit branding hidden.

---

## Engineering & DevOps

**CI/CD** — A GitHub Actions workflow (`.github/workflows/ci.yml`) runs a smoke test on every push: generates a tiny synthetic dataset, runs the full `solution.py` pipeline, and validates the output submission (correct shape, columns, value ranges). A green **CI passing** badge renders in the README.

**Containerization** — A `Dockerfile` builds a lightweight Python 3.11 image that installs dependencies and runs the smoke test, enabling reproducible execution in any environment.

**Build automation** — A `Makefile` provides one-command targets: `make train` (run the pipeline), `make ensemble` (execute the stacked-ensemble notebook), `make test` (CI smoke test), `make lint` (ruff + mypy), `make clean`.

**Code quality** — A `.pre-commit-config.yaml` with ruff (linting + auto-formatting) and standard hooks (trailing whitespace, end-of-file, YAML validation, large-file guard) runs automatically on every commit. Type hints and docstrings are added to the core functions in the source modules.

**Data validation** — `tests/validate_data.py` performs schema checks (column names, types, value ranges, uniqueness, format) on all input CSVs before training, catching data-integrity issues early.

**Collaboration infrastructure** — `CONTRIBUTING.md` documents the development workflow (fork, branch, test, PR). GitHub issue templates (`.github/ISSUE_TEMPLATE/`) provide structured bug-report and feature-request forms. `CHANGELOG.md` records the iteration history from baseline through the stacked ensemble.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Core ML** | Python 3.9+, pandas, NumPy, scikit-learn |
| **Gradient boosting** | LightGBM, XGBoost, CatBoost |
| **Stacking** | scikit-learn Ridge (meta-learner), K-Fold OOF predictions |
| **Feature engineering** | Custom geohash base-32 decoder, K-means clustering, cyclical encoding, Bayesian-smoothed target encoding |
| **Visualization** | matplotlib (dark-themed spatial heatmaps, ablation charts, error analysis) |
| **Interactive demo** | Streamlit (glass-morphism CSS, precomputed profile serving) |
| **CI/CD** | GitHub Actions (smoke test on push) |
| **Containerization** | Docker (Python 3.11-slim) |
| **Code quality** | ruff (lint + format), mypy (type checking), pre-commit hooks |
| **Build automation** | GNU Make |
| **Data validation** | Custom schema-checking script (pandas-based assertions) |
| **Version control** | Git, GitHub (MIT license, issue templates, CONTRIBUTING guide) |

---

## Repository Structure

```
traffic-demand-prediction/
├── README.md                               # Visual landing page (badges, SVG diagrams, embedded charts)
├── LICENSE                                  # MIT
├── Makefile                                 # One-command: make train / make test / make lint
├── Dockerfile                               # Reproducible container
├── CHANGELOG.md                             # Iteration history (v0.1 → v3.0)
├── CONTRIBUTING.md                          # Development workflow guide
├── requirements.txt                         # Core + optional + dev dependencies
├── .gitignore                               # Excludes data/*.csv, submission files, caches
├── .pre-commit-config.yaml                  # ruff + hooks (auto-format on commit)
├── .github/
│   ├── workflows/ci.yml                     # CI smoke test → green build badge
│   └── ISSUE_TEMPLATE/                      # Bug report + feature request templates
├── app/
│   ├── streamlit_app.py                     # Interactive demo (glass-morphism UI)
│   ├── data/profiles_*.csv|json             # Precomputed profiles (no raw data exposed)
│   └── .streamlit/config.toml               # Dark theme config
├── src/
│   ├── solution.py                          # End-to-end single-model pipeline
│   └── feature_engineering.py               # Shared feature builder (42 features)
├── notebooks/
│   ├── 01_eda.ipynb                         # Exploratory data analysis
│   ├── 02_stacked_ensemble.ipynb            # 5-model stacked ensemble (headline)
│   ├── 03_recursive_forecast.ipynb          # Autoregressive lag experiment
│   ├── 04_error_analysis.ipynb              # Residual analysis by hour/road/weather
│   ├── 05_ablation_study.ipynb              # Feature-group contribution study
│   └── 06_geospatial_visualization.ipynb    # Demand heatmaps, clusters, spillover
├── tests/
│   ├── make_synth_and_check.py              # CI smoke test script
│   └── validate_data.py                     # Schema & quality validation
├── docs/
│   └── APPROACH.md                          # Full methodology write-up
├── assets/
│   ├── pipeline.svg                         # Architecture diagram
│   ├── results.svg                          # Performance chart
│   ├── feature_importance.svg               # Permutation importance chart
│   └── *.png                                # Generated visualization outputs
└── data/
    └── .gitkeep                             # Local data directory (git-ignored)
```

---

## Results

| Approach | Accuracy |
|----------|:--------:|
| Persistence baseline (copy reference day) | ~80 |
| Single GBM (LightGBM, 3×5 bagged) | ~90 |
| **Stacked ensemble (LightGBM + XGBoost + CatBoost + HGBR + ExtraTrees + Ridge)** | **~97** |
