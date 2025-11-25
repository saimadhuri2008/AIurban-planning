#!/usr/bin/env python3
"""
traffic_congestion.py
Generates a large synthetic traffic congestion dataset with correct ward mapping.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("traffic_congestion")

# ---------- FIXED GLOBAL PATHS ----------
H3_GEOJSON = Path(r"C:\AIurban-planning\outputs\h3_cells.geojson")
WARDS_GEOJSON = Path(r"C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")

# ----------------------------------------------------------
# Load wards
# ----------------------------------------------------------
def load_wards(path: Path):
    gdf = gpd.read_file(path).to_crs(4326)
    ward_col = next((c for c in gdf.columns if "ward" in c.lower()), "ward_id")
    if ward_col != "ward_id":
        gdf = gdf.rename(columns={ward_col: "ward_id"})
    gdf["ward_id"] = gdf["ward_id"].astype(str)
    return gdf[["ward_id", "geometry"]]

# ----------------------------------------------------------
# Load H3 cells from temporal output
# ----------------------------------------------------------
def load_h3_cells(path: Path):
    h3g = gpd.read_file(path).to_crs(4326)
    if "h3_id" not in h3g.columns:
        raise ValueError("h3_id missing in H3 geojson")
    return h3g[["h3_id", "centroid_lat", "centroid_lon", "geometry"]]

# ----------------------------------------------------------
# Create point centroids and spatial join
# ----------------------------------------------------------
def build_h3_to_ward_mapping(h3g: gpd.GeoDataFrame, wards: gpd.GeoDataFrame):
    pts = gpd.GeoDataFrame(
        {
            "h3_id": h3g["h3_id"],
            "geometry": [Point(lon, lat) for lon, lat in zip(h3g["centroid_lon"], h3g["centroid_lat"])]
        },
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(pts, wards, how="left", predicate="within")

    joined["ward_id"] = joined["ward_id"].fillna("ward_unknown")
    mapping = dict(zip(joined["h3_id"], joined["ward_id"]))
    return mapping

# ----------------------------------------------------------
# Synthetic traffic generator
# ----------------------------------------------------------
def generate_traffic(h3_ids, timestamps):
    n_cells = len(h3_ids)
    n_steps = len(timestamps)

    rng = np.random.default_rng(20251120)

    avg_speed = np.zeros((n_steps, n_cells))
    max_speed = np.zeros((n_steps, n_cells))
    cong = np.zeros((n_steps, n_cells))
    tti = np.zeros((n_steps, n_cells))
    veh = np.zeros((n_steps, n_cells))

    hours = np.linspace(0, 2 * np.pi, n_steps)
    diurnal = (np.sin(hours - np.pi/2) + 1) / 2

    for t in range(n_steps):
        traffic_factor = diurnal[t]

        avg_speed[t] = 45 - traffic_factor * 25 + rng.normal(0, 3, n_cells)
        max_speed[t] = avg_speed[t] + rng.normal(4, 2, n_cells)

        cong[t] = np.clip(traffic_factor * 1.1 + rng.normal(0, 0.1, n_cells), 0, 1)
        tti[t] = 1 + (traffic_factor * 1.5)

        veh[t] = (400 * traffic_factor) + rng.normal(0, 30, n_cells)
        veh[t] = np.clip(veh[t], 10, None)

    return avg_speed, max_speed, cong, tti, veh

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading wards...")
    wards = load_wards(WARDS_GEOJSON)

    log.info("Loading H3 grid...")
    h3g = load_h3_cells(H3_GEOJSON)
    h3_ids = list(h3g["h3_id"])

    log.info("Building H3 → ward mapping...")
    h3_to_ward = build_h3_to_ward_mapping(h3g, wards)

    # ------------------- TIME INDEX -------------------
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    timestamps = pd.date_range(start, end, freq=args.freq)

    # ------------------- TRAFFIC -------------------
    avg_speed, max_speed, cong, tti, veh = generate_traffic(h3_ids, timestamps)

    # ------------------- BUILD DF -------------------
    rows = []
    for t_idx, ts in enumerate(timestamps):
        for c_idx, h in enumerate(h3_ids):
            rows.append({
                "timestamp": ts,
                "h3_id": h,
                "ward_id": h3_to_ward[h],          # FIXED
                "avg_speed": float(avg_speed[t_idx][c_idx]),
                "max_speed": float(max_speed[t_idx][c_idx]),
                "congestion_index": float(cong[t_idx][c_idx]),
                "travel_time_index": float(tti[t_idx][c_idx]),
                "vehicle_count": float(veh[t_idx][c_idx])
            })

    df = pd.DataFrame(rows)
    log.info("Created %d records", len(df))

    # Save CSV
    csv_path = outdir / "traffic_congestion.csv"
    df.to_csv(csv_path, index=False)

    # Save GeoJSON (join with H3 geometry)
    merged = df.merge(h3g[["h3_id", "geometry"]], on="h3_id", how="left")
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    geo_path = outdir / "traffic_congestion.geojson"
    gdf.to_file(geo_path, driver="GeoJSON")

    log.info("WROTE: %s", csv_path)
    log.info("WROTE: %s", geo_path)
    log.info("Done.")

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--freq", type=str, default="1H")
    p.add_argument("--out", type=str, default="./outputs")
    args = p.parse_args()
    main(args)
