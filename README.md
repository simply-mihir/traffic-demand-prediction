<div align="center">

# 🚦 Traffic Demand Prediction

### Spatio-temporal forecasting of normalized traffic demand on a 15-minute geohash grid

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-9ACD32)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-EB0F00)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-gradient%20boosting-FFCC00)](https://catboost.ai/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![CI](https://github.com/simply-mihir/traffic-demand-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/simply-mihir/traffic-demand-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Predicting travel demand to help understand urban traffic patterns and alleviate congestion.*

</div>

---

## 📌 Overview

This repository tackles a **travel-demand forecasting** problem: given a city's historical
demand aggregated into **geohash locations** over **15-minute buckets**, predict the
normalized `demand` (a value in `[0, 1]`) for a held-out future window.

The data is a **spatio-temporal time series**, not a table of independent rows — and the
solution is built around that fact. The headline pipeline is a **stacked ensemble of five
gradient-boosted / tree models** on top of rich geohash×time features.

> **Evaluation metric:** `accuracy = max(0, 100 × R²(actual, predicted))`

<div align="center">
<img src="assets/pipeline.svg" alt="Solution pipeline" width="100%"/>
</div>

---

## 🗂️ Dataset

| File | Rows × Cols | Description |
|------|-------------|-------------|
| `train.csv` | 77,299 × 11 | Historical demand with context features |
| `test.csv`  | 41,778 × 10 | Records to forecast (target hidden) |
| `sample_submission.csv` | 5 × 2 | Submission format (`Index`, `demand`) |

**Columns**

| Column | Meaning |
|--------|---------|
| `Index` | Unique row identifier |
| `geohash` | Geocoded location (decodes to lat/lon; adjacency preserved) |
| `day` | Sequential day index (not a calendar date) |
| `timestamp` | Time of day, `H:M`, on a 15-minute grid |
| `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks` | Road context |
| `Temperature`, `Weather` | Environmental context |
| `demand` | **Target** — normalized traffic demand in `[0, 1]` |

**Structure that matters:** training spans one full reference day plus the early hours of
the next; the test set continues that next day through the daytime. The same geohash
locations recur across train and test, and demand follows a smooth, location-specific
**time-of-day profile** — the dominant predictive signal.

---

## 🧠 Approach

### 1 · Feature engineering (`src/feature_engineering.py`)
Built **jointly on train + test** so every encoding is consistent.

- **Spatial** — geohash decoded to latitude/longitude, region prefixes, and **K-means
  geo-clusters** grouping nearby locations.
- **Temporal** — minutes-of-day, hour/minute, **cyclical sin/cos** encodings, day index,
  day-of-week.
- **Demand profile (core signal)** — **leak-free K-fold target encodings** of mean demand
  at multiple granularities (`geohash`, `geohash×hour`, `geohash×slot`, region×hour, …),
  a **denoised reference-day time-of-day profile**, and a per-geohash recent level.
- **Spatial spillover** — mean demand of each location's *k* nearest geohash neighbours
  at the same time slot (geohash adjacency is preserved), capturing local diffusion.
- **Context** — road and weather attributes (numeric + categorical).

<div align="center">
<img src="assets/feature_importance.svg" alt="Feature importances" width="80%"/>
</div>

The chart above is the model's **permutation importance** — the reference-day time-of-day
profile and geohash×time demand encodings carry the signal, confirming the spatio-temporal framing.

### 2 · Models
- **`src/solution.py`** — single end-to-end pipeline: gradient-boosted trees
  (**LightGBM**, automatic fallback to scikit-learn **HistGradientBoosting**), trained in
  **log space**, predictions clipped to `[0, 1]`, **bagged over 3 seeds × 5 folds**.
- **`notebooks/02_stacked_ensemble.ipynb`** — the headline model: **out-of-fold stacking**
  of **LightGBM + XGBoost + CatBoost + HistGradientBoosting + ExtraTrees**, combined by a
  **Ridge meta-learner**. Boosters use `geohash` and other categoricals **natively**.

### 3 · Validation
A **forward holdout** (train on the reference day, predict the held-out next-day records)
is used for model selection — a random K-fold split is optimistic for time-series data and
is reported only for reference.

---

## 📊 Results

<div align="center">
<img src="assets/results.svg" alt="Results by approach" width="90%"/>
</div>

| Approach | Accuracy |
|----------|:--------:|
| Persistence baseline | ~80 |
| Single GBM (`solution.py`) | ~90 |
| **Stacked ensemble** (`02_stacked_ensemble.ipynb`) | **97.85** |

Validation uses a **forward holdout** (train on the reference day, predict the held-out
next-day records) so reported numbers reflect true forecasting performance rather than an
optimistic random split. See [`docs/APPROACH.md`](docs/APPROACH.md) for the full
methodology, EDA, and ablations.

---

## 🚀 Quickstart

```bash
# 1. clone & install
git clone https://github.com/USERNAME/traffic-demand-prediction.git
cd traffic-demand-prediction
pip install -r requirements.txt          # LightGBM/XGBoost/CatBoost optional; HGBR fallback works

# 2. place the data
#    put train.csv, test.csv, sample_submission.csv in ./data/

# 3a. run the single-model pipeline
python src/solution.py                    # writes submission.csv

# 3b. or run the stacked ensemble
jupyter notebook notebooks/02_stacked_ensemble.ipynb
```

Both produce a `submission.csv` of shape **(41778, 2)** with columns `Index, demand`,
predictions clipped to `[0, 1]`.

---

## 📁 Repository structure

```
traffic-demand-prediction/
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt                # dependencies
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI smoke-test (build badge)
├── src/
│   ├── solution.py                 # end-to-end single-model pipeline (LightGBM → HGBR)
│   └── feature_engineering.py      # shared feature builder (encodings, profile, clusters)
├── notebooks/
│   ├── 01_eda.ipynb                # exploratory data analysis
│   ├── 02_stacked_ensemble.ipynb   # 5-model stacked ensemble (headline)
│   └── 03_recursive_forecast.ipynb # autoregressive lag-based experiment
├── docs/
│   └── APPROACH.md                 # full methodology, EDA & experiments
├── assets/
│   ├── pipeline.svg                # architecture diagram
│   ├── results.svg                 # results chart
│   └── feature_importance.svg      # permutation-importance chart
└── data/                           # place train.csv / test.csv here (git-ignored)
    └── .gitkeep
```

---

## 🛠️ Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` · `XGBoost` · `CatBoost` · `Jupyter`

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">
<sub>Built for a traffic-demand forecasting challenge · spatio-temporal ML</sub>
</div>
