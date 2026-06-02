# Changelog

All notable iterations of this project.

## v3.0 — Stacked Ensemble (current)
- **5-model stacked ensemble** (LightGBM + XGBoost + CatBoost + HGBR + ExtraTrees)
  with a Ridge meta-learner over out-of-fold predictions.
- Added **spatial-neighbour demand** feature (k-nearest geohash neighbours).
- Added geospatial visualization notebook.
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
