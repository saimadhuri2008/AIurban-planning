#!/usr/bin/env python3
"""
validate.py
Robust validation & quick plots for temporal parquet files.
Works whether the parquet has 'h3_index'/'timestamp' as index or as columns.
Saves:
 - sample_plot.png
 - temporal_h3_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
log = logging.getLogger("temporal_validate")

# ---- CONFIG ----
PARQUET_PATH = Path("data/raw/temporal/temporal_30d_combined.parquet")
OUT_DIR = Path("data/processed/temporal_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PARQUET_PATH.exists():
    log.error("Parquet not found: %s", PARQUET_PATH)
    sys.exit(1)

log.info("Loading parquet: %s", PARQUET_PATH)
df = pd.read_parquet(PARQUET_PATH)

# If parquet was saved with MultiIndex, bring indices back as columns
if isinstance(df.index, pd.MultiIndex):
    log.info("Detected MultiIndex: converting index levels to columns")
    df = df.reset_index()
else:
    # if index is single datetime index, ensure timestamp column present
    if "timestamp" not in df.columns and df.index.inferred_type == "datetime64":
        df = df.reset_index().rename(columns={"index": "timestamp"})

# Ensure timestamp is datetime and sorted
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
else:
    log.error("No 'timestamp' column found after reset. Columns: %s", df.columns.tolist())
    sys.exit(1)

# Basic sanity checks
n_rows, n_cols = df.shape
log.info("Loaded dataframe. Rows: %d Columns: %d", n_rows, n_cols)

if "h3_index" not in df.columns:
    # maybe it was saved with a different name; try to infer
    candidates = [c for c in df.columns if c.lower().startswith("h3") or "h3" in c.lower()]
    if candidates:
        df = df.rename(columns={candidates[0]: "h3_index"})
        log.info("Renamed %s -> h3_index", candidates[0])
    else:
        log.error("No h3_index column found. Columns available: %s", df.columns.tolist())
        sys.exit(1)

# Print head and basic info
log.info("\n%s", df.head().to_string())
log.info("Unique H3 cells: %d", df["h3_index"].nunique())
log.info("Time range: %s -> %s", df["timestamp"].min(), df["timestamp"].max())

# Choose a sample H3 index (median-frequency one)
h3_counts = df["h3_index"].value_counts()
sample_h3 = h3_counts.index[0]  # most common (should be all equal)
log.info("Using sample h3_index: %s (rows=%d)", sample_h3, int(h3_counts.iloc[0]))

# Prepare sample timeseries for plotting
sample = df[df["h3_index"] == sample_h3].sort_values("timestamp")
if "aqi" not in sample.columns:
    log.warning("No 'aqi' column found. Available columns: %s", sample.columns.tolist())
else:
    plt.figure(figsize=(12, 4))
    plt.plot(sample["timestamp"], sample["aqi"], marker=".", linewidth=0.8)
    plt.title(f"AQI time series for H3: {sample_h3}")
    plt.xlabel("timestamp")
    plt.ylabel("AQI")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plot_path = OUT_DIR / "sample_aqi_timeseries.png"
    plt.savefig(plot_path, dpi=150)
    log.info("Saved plot: %s", plot_path)
    plt.close()

# Summary statistics per H3 cell (mean, min, max for AQI and temp if present)
agg_metrics = {}
metrics = []
if "aqi" in df.columns:
    metrics.append("aqi")
if "temp_c" in df.columns:
    metrics.append("temp_c")

if metrics:
    summary = df.groupby("h3_index")[metrics].agg(["mean", "min", "max", "std"])
    # flatten columns
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    csv_path = OUT_DIR / "temporal_h3_summary.csv"
    summary.to_csv(csv_path, index=False)
    log.info("Wrote summary CSV: %s (rows=%d, cols=%d)", csv_path, summary.shape[0], summary.shape[1])
else:
    log.warning("No AQI or temperature metrics found to summarize.")

log.info("Validation finished successfully.")
