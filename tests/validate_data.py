"""Data quality validation — run before training to catch schema issues early.

Usage:
    python tests/validate_data.py              # validates data/*.csv
    python tests/validate_data.py path/to/dir  # validates csv files in a custom dir
"""
import sys, os
import pandas as pd
import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"

EXPECTED_TRAIN_COLS = [
    "Index", "geohash", "day", "timestamp", "demand",
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    "Temperature", "Weather"
]
EXPECTED_TEST_COLS = [c for c in EXPECTED_TRAIN_COLS if c != "demand"]

def check(cond: bool, msg: str) -> None:
    status = "✅" if cond else "❌"
    print(f"  {status} {msg}")
    if not cond:
        raise AssertionError(msg)

def validate_train(path: str) -> None:
    print(f"\n— Validating {path}")
    df = pd.read_csv(path)
    check(list(df.columns) == EXPECTED_TRAIN_COLS, f"columns match expected schema")
    check(len(df) > 0, f"non-empty ({len(df):,} rows)")
    check(df["Index"].is_unique, "Index is unique")
    check(df["Index"].is_monotonic_increasing, "Index is sorted ascending")
    check(df["geohash"].str.len().eq(6).all(), "all geohashes are length 6")
    check(df["demand"].between(0, 1).all(), "demand in [0, 1]")
    check(df["demand"].notna().all(), "no NaN in demand")
    check(df["day"].dtype in [np.int64, np.int32, int], "day is integer")
    check(df["timestamp"].str.contains(":").all(), "timestamp has H:M format")
    print(f"  📊 {len(df):,} rows × {len(df.columns)} cols  |  "
          f"days {df['day'].min()}–{df['day'].max()}  |  "
          f"geohashes {df['geohash'].nunique():,}")

def validate_test(path: str) -> None:
    print(f"\n— Validating {path}")
    df = pd.read_csv(path)
    check(set(EXPECTED_TEST_COLS).issubset(set(df.columns)), "columns match expected schema")
    check("demand" not in df.columns, "demand column is absent (target hidden)")
    check(len(df) > 0, f"non-empty ({len(df):,} rows)")
    check(df["Index"].is_unique, "Index is unique")
    check(df["geohash"].str.len().eq(6).all(), "all geohashes are length 6")
    print(f"  📊 {len(df):,} rows × {len(df.columns)} cols")

def validate_sample(path: str) -> None:
    print(f"\n— Validating {path}")
    df = pd.read_csv(path)
    check(list(df.columns) == ["Index", "demand"], "columns are [Index, demand]")
    check(len(df) > 0, "non-empty")
    print(f"  📊 {len(df)} rows (submission format reference)")

if __name__ == "__main__":
    print("=" * 50)
    print("Data Quality Validation")
    print("=" * 50)
    ok = True
    for name, fn in [("train.csv", validate_train), ("test.csv", validate_test),
                      ("sample_submission.csv", validate_sample)]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            try:
                fn(path)
            except AssertionError:
                ok = False
        else:
            print(f"\n  ⚠️  {path} not found — skipping")
    print("\n" + ("✅ All checks passed!" if ok else "❌ Some checks failed."))
