# Traffic Demand Prediction — Approach & Methodology

## 1. Problem framing
The task is to predict normalized traffic `demand` (target in [0, 1]) for a set of
test records identified by `Index`, given location (`geohash`), `day`, `timestamp`,
and several context attributes (`RoadType`, `NumberofLanes`, `LargeVehicles`,
`Landmarks`, `Temperature`, `Weather`). Scoring is `max(0, 100 * R²)`.

After exploratory analysis the data is best understood as a **spatio-temporal
time series**, not a set of independent rows:
- `timestamp` is `H:M` on a 15-minute grid; `day` is a sequential integer.
- Training spans day 48 (a full day, all 96 fifteen-minute slots) plus the early
  hours of day 49; the test set is day 49 from 02:15 to 13:45.
- The same `geohash` locations recur across train and test (the location set is
  effectively shared), and the test time-slots are a continuation of the series
  rather than a random sample.

This framing drives every modeling decision below.

## 2. Exploratory data analysis
Key findings that shaped the solution:
- **Demand is highly skewed and near-zero-inflated**: median ≈ 0.048, with a long
  right tail toward 1.0. This motivated training in **log space** (`log1p`) and
  clipping predictions to the observed `[0, 1]` range.
- **Strong time-of-day structure**: demand follows a smooth daily profile per
  location, rising from early morning into the daytime. This is the single most
  predictive signal.
- **Geohash decodes to a coherent spatial region** (decoded lat/lon form a compact
  bounding box), so latitude/longitude and geohash prefixes carry real spatial
  information, and nearby locations behave similarly.
- **Context attributes add little marginal signal**: within a fixed
  (geohash, time) group, `demand` correlates weakly with `Temperature`/`Weather`/
  `RoadType`. They are included but are not the primary drivers.

## 3. Feature engineering
All features are built jointly on train and test so encodings are consistent.

**Spatial**
- `geohash` decoded to **latitude / longitude** (custom base-32 decoder).
- Region **prefixes** (`gh4`, `gh5`) and a full geohash code as categoricals.
- **K-means geo-clusters** on (lat, lon) to group nearby locations (used by the
  ensemble notebook).

**Temporal**
- Minutes-of-day, hour, minute.
- **Cyclical encodings** (sin/cos of time-of-day and of hour) so the model treats
  23:45 and 00:00 as adjacent.
- Day index and day-of-week (`day % 7`).

**Demand profile (the core signal)**
- **Leak-free K-fold target encodings** of mean demand at several granularities:
  `geohash`, `geohash×hour`, `geohash×slot`, `gh4`, `gh4×hour`, `gh5×hour`,
  plus `RoadType`, `Weather`, and `RoadType×hour`, `Weather×hour`. Out-of-fold
  computation prevents target leakage; smoothing shrinks low-count groups toward
  the global mean.
- **Denoised day-48 time-of-day profile** per geohash (rolling-smoothed across
  neighbouring slots) and neighbouring-slot values as lag-style features.
- A per-geohash **recent level** feature from the latest available day.
**Spatial spillover**
- Mean demand of each location's *k* nearest geohash neighbours at the same time slot
  (computed from the reference day; geohash adjacency is preserved). Captures local
  diffusion of demand between adjacent areas.
**Context**
- `NumberofLanes`, `Temperature` (numeric); `LargeVehicles`, `Landmarks`,
  `RoadType`, `Weather` (categorical / binary-normalized).

## 4. Modeling
**Primary model.** Gradient-boosted decision trees. The provided `solution.py`
uses **LightGBM** when available and falls back automatically to scikit-learn's
**HistGradientBoostingRegressor** so it runs in any environment. Training is in
log space; predictions are exponentiated and clipped to [0, 1]; the model is
**bagged over 3 seeds × 5 folds** and averaged for stability.

**Hard stacked ensemble (see `hard_ensemble.ipynb`).** For maximum accuracy we
stack five model families with out-of-fold predictions combined by a Ridge
meta-learner:
- LightGBM, XGBoost, CatBoost (gradient boosting; `geohash` and other categoricals
  used **natively**),
- HistGradientBoosting and ExtraTrees (scikit-learn).
Each model is trained 5-fold; its out-of-fold predictions feed a non-negative
Ridge meta-model, and the stacked test prediction is the final output.

## 5. Validation strategy
Because the test is a future window, we validate with a **forward holdout**:
train on day 48 and predict the held-out day-49 records. We monitor R² on this
holdout rather than a random split, since a random K-fold split is optimistic
for time-series data. We also report random-CV R² for reference but do not select
on it.

## 6. Tools
- Python 3, pandas, numpy
- scikit-learn (HistGradientBoosting, ExtraTrees, KFold, Ridge, KMeans, metrics)
- LightGBM, XGBoost, CatBoost (gradient-boosted trees; categorical support)
- Custom geohash base-32 decoder (no external geohash dependency required)

## 7. How to run
```
# place train.csv, test.csv, sample_submission.csv beside the scripts
pip install lightgbm xgboost catboost   # optional; HGBR fallback works without them
python solution.py                       # writes submission.csv  (single strong model)
# or, for the stacked ensemble, run hard_ensemble.ipynb top to bottom
```
Both write a `submission.csv` of shape (41778, 2) with columns `Index, demand`,
predictions clipped to [0, 1].

## 8. Files in this archive
- `APPROACH.md` — this document.
- `solution.py` — end-to-end single-model pipeline (LightGBM → HGBR fallback).
- `hard_ensemble.ipynb` — 5-model stacked ensemble (LightGBM + XGBoost + CatBoost
  + HGBR + ExtraTrees, Ridge meta-learner).
- `feature_engineering.py` — the shared feature-building module used by the
  ensemble (geohash decode, encodings, profile, clustering).
- `requirements.txt` — dependencies.
