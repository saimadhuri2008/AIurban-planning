#!/usr/bin/env python3
"""
mobility_od.py

Generate OD flows (origin_h3, destination_h3, od_flow, time_of_day, purpose)
Large synthetic dataset suitable for mobility modelling / transport demand estimation.

Outputs:
 - od_flows_{days}d.parquet
 - od_flows_summary_by_h3.csv

Usage:
 python mobility_od.py --days 7 --h3-res 8 --k-ring 6 --out outputs
"""

import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Polygon, Point

try:
    import h3
except Exception:
    h3 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("mobility_od")

RNG_SEED = 20251119
rng = np.random.RandomState(RNG_SEED)


def h3_boundary_poly(h):
    b = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(pt[1], pt[0]) for pt in b])


def generate_h3_cells(center_lat=12.9716, center_lon=77.5946, res=8, k_ring=6):
    if h3 is None:
        return [f"dummy_{i}" for i in range(max(200, k_ring*20))]
    center = h3.geo_to_h3(center_lat, center_lon, res)
    cells = list(h3.k_ring(center, k_ring))
    return list(cells)


def safe_write_geojson(gdf: gpd.GeoDataFrame, path: Path):
    if "geometry" not in gdf.columns:
        raise ValueError("No geometry column")
    gdf = gdf.set_geometry("geometry")
    gdf.to_file(str(path), driver="GeoJSON")


def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    days = args.days
    h3_res = args.h3_res
    k_ring = args.k_ring

    # build H3 set around city center
    center_lat, center_lon = 12.9716, 77.5946
    h3_cells = generate_h3_cells(center_lat=center_lat, center_lon=center_lon, res=h3_res, k_ring=k_ring)
    n = len(h3_cells)
    log.info("Using %d H3 cells for OD generation", n)

    # make H3 geo static file (polygons)
    h3_rows = []
    for h in h3_cells:
        poly = h3_boundary_poly(h) if h3 is not None else None
        lat, lon = h3.h3_to_geo(h) if h3 is not None else (None, None)
        h3_rows.append({"h3_id": h, "geometry": poly, "centroid_lat": lat, "centroid_lon": lon})
    h3_gdf = gpd.GeoDataFrame(h3_rows, geometry="geometry", crs="EPSG:4326")
    safe_write_geojson(h3_gdf, Path(outdir) / "mobility_h3_static.geojson")

    # OD generation parameters
    time_bins_per_day = 24  # hourly
    total_periods = days * time_bins_per_day

    # Grow a gravity-like popularity for each cell (some cells are stronger origins/destinations)
    pop = np.clip(1 + np.linspace(0, 5, n) + rng.normal(0, 0.5, n), 0.1, None)
    # normalize to ~city scale
    pop = pop / pop.sum()

    # purposes distribution
    purposes = ["work", "school", "shopping", "leisure", "other"]
    purpose_probs = [0.45, 0.15, 0.15, 0.15, 0.10]

    rows = []
    for t in range(total_periods):
        tod = t % time_bins_per_day  # hour of day
        # base OD trips this hour
        hourly_total = int(10000 * (0.5 + 0.8 * (1 if 7 <= tod <= 9 or 17 <= tod <= 19 else 0.4)))  # peaks
        # sample origins and destinations by pop * spatial gradient
        origins = rng.choice(range(n), size=hourly_total, p=pop)
        destinations = rng.choice(range(n), size=hourly_total, p=pop[::-1])  # slight difference
        for o, d in zip(origins, destinations):
            # small chance of same cell trip
            if o == d and rng.rand() > 0.6:
                continue
            flow = 1
            purpose = rng.choice(purposes, p=purpose_probs)
            rows.append({
                "timestamp_hour": pd.Timestamp.now().floor("H") - pd.Timedelta(days=days) + pd.Timedelta(hours=t),
                "origin_h3": h3_cells[o],
                "destination_h3": h3_cells[d],
                "od_flow": flow,
                "time_of_day_hour": int(tod),
                "purpose": purpose
            })
        # flush in chunks to avoid memory blow for large days
        if len(rows) > 200000:
            df_chunk = pd.DataFrame(rows)
            path_chunk = Path(outdir) / f"od_chunk_{t}.parquet"
            df_chunk.to_parquet(path_chunk, index=False)
            log.info("WROTE chunk: %s rows=%d", path_chunk, df_chunk.shape[0])
            rows = []

    # final write
    if rows:
        df_final = pd.DataFrame(rows)
    else:
        # load all chunks and concat
        parquet_files = sorted(Path(outdir).glob("od_chunk_*.parquet"))
        dfs = [pd.read_parquet(p) for p in parquet_files]
        df_final = pd.concat(dfs, ignore_index=True)

    out_path = Path(outdir) / f"od_flows_{days}d.parquet"
    df_final.to_parquet(out_path, index=False)
    log.info("WROTE OD flows: %s rows=%d", out_path, df_final.shape[0])

    # summary: aggregate OD flows by origin/destination totals
    summary = df_final.groupby("origin_h3")["od_flow"].sum().reset_index().rename(columns={"od_flow": "origin_total_flow"})
    summary2 = df_final.groupby("destination_h3")["od_flow"].sum().reset_index().rename(columns={"od_flow": "dest_total_flow"})
    sum_df = summary.merge(summary2, left_on="origin_h3", right_on="destination_h3", how="outer"
                           ).fillna(0)
    sum_df.to_csv(Path(outdir) / "od_flows_summary_by_h3.csv", index=False)
    log.info("WROTE summary: %s", Path(outdir) / "od_flows_summary_by_h3.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", "--outdir", dest="outdir", default="./outputs")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--h3-res", type=int, default=8, dest="h3_res")
    p.add_argument("--k-ring", type=int, default=6)
    args = p.parse_args()
    main(args)
