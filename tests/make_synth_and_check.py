"""CI helper: generate a tiny synthetic dataset, run the pipeline, validate output.
Kept dependency-light (pandas, numpy, scikit-learn) so CI installs fast."""
import os, subprocess, sys
import numpy as np, pandas as pd

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(0)
B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
geos = ["".join(rng.choice(list(B32), 6)) for _ in range(20)]

def make(n, day, slots):
    rows = []
    slots = list(slots)
    for _ in range(n):
        s = int(rng.choice(slots))
        rows.append(dict(
            geohash=rng.choice(geos), day=day,
            timestamp=f"{s//4}:{(s%4)*15}",
            RoadType=rng.choice(["A", "B", "C"]), NumberofLanes=int(rng.integers(1, 6)),
            LargeVehicles=rng.choice(["allowed", "not allowed"]),
            Landmarks=rng.choice(["yes", "no"]),
            Temperature=round(float(rng.normal(20, 5)), 1),
            Weather=rng.choice(["clear", "rain", "clouds"])))
    return pd.DataFrame(rows)

tr = pd.concat([make(800, 48, range(96)), make(80, 49, range(9))], ignore_index=True)
tr["demand"] = rng.uniform(0, 1, len(tr)).round(4)
tr.insert(0, "Index", range(len(tr)))
te = make(200, 49, range(9, 56)); te.insert(0, "Index", range(len(te)))
tr.to_csv("data/train.csv", index=False)
te.to_csv("data/test.csv", index=False)
pd.DataFrame({"Index": te["Index"][:5], "demand": [0]*5}).to_csv("data/sample_submission.csv", index=False)
print("synthetic data ready:", tr.shape, te.shape)

subprocess.run([sys.executable, "src/solution.py"], check=True)

s = pd.read_csv("submission.csv")
assert list(s.columns) == ["Index", "demand"], s.columns
assert s["demand"].notna().all(), "NaNs in predictions"
assert s["demand"].between(0, 1).all(), "predictions outside [0,1]"
print("submission OK:", s.shape)
