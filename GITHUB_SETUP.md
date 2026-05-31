# GitHub Setup Guide

Everything you need to publish this repository.

## 1. Create the repo & push

```bash
cd traffic-demand-prediction
git init
git add .
git commit -m "Traffic demand prediction: features + stacked ensemble"
git branch -M main
git remote add origin https://github.com/<your-username>/traffic-demand-prediction.git
git push -u origin main
```

## 2. About section (paste into the repo's ⚙️ "About" box)

**Description**
> Spatio-temporal forecasting of normalized traffic demand on a 15-minute geohash grid. Feature-rich geohash×time encodings + a stacked ensemble of LightGBM, XGBoost, CatBoost, HistGradientBoosting and ExtraTrees.

**Website** *(optional)* — link to your notebook/report if hosted.

## 3. Topics / tags (add via the "About" gear → Topics)

```
machine-learning
time-series
forecasting
demand-prediction
gradient-boosting
lightgbm
xgboost
catboost
scikit-learn
ensemble-learning
feature-engineering
geohash
spatio-temporal
traffic-prediction
data-science
python
```

## 4. Suggested repo settings
- ✅ Include the README (already the landing page)
- ✅ Add the MIT license (already in `LICENSE`)
- 📌 Pin the repo on your profile
- 🖼️ Social preview: upload `assets/pipeline.svg` (exported to PNG) under
  Settings → Social preview for a nice link card.

## 5. Recommended commit hygiene
The `.gitignore` already excludes `data/*.csv` and `submission*.csv`, so you won't
accidentally commit the dataset or output files. Keep the data in `data/` locally.
