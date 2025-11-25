#!/usr/bin/env python3
"""
env_climate_synthetic.py

Generate 3 synthetic datasets (NDVI, LST, Flood risk) per H3 and aggregate to wards.
If real rasters are provided (ndvi_raster, lst_raster, dem_raster) the script will try to use them.

Outputs (per variant):
 - ndvi_{variant}_h3.geojson  + .csv
 - ndvi_{variant}_wards.csv
 - lst_{variant}_h3.geojson   + .csv
 - lst_{variant}_wards.csv
 - flood_{variant}_h3.geojson + .csv
 - flood_{variant}_wards.csv
"""
import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, mapping
import h3
import sys
from datetime import datetime

RNG_SEED = 1234
rng = np.random.RandomState(RNG_SEED)

# ----------------------
# Utilities
# ----------------------
def echo(s):
    print(s); sys.stdout.flush()

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def h3_polygon(h):
    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(p[1], p[0]) for p in boundary])

def h3_centroid_latlon(h):
    lat, lon = h3.h3_to_geo(h)
    return lat, lon

# ----------------------
# Build H3 grid covering wards
# ----------------------
def build_h3_grid_for_wards(wards_gdf, h3_res):
    # compute ward union bbox and then find H3 cells covering wards
    echo("Building H3 index set covering wards...")
    # union all ward polygons
    union = wards_gdf.unary_union
    # sample a fine grid of points across bbox and select h3 cells covering union
    minx, miny, maxx, maxy = union.bounds
    # create points at regular lat/lon increments ~ approximate spacing by resolution
    # but simpler: get H3 cells covering polygon by iterating boundaries of wards
    h3_set = set()
    for geom in wards_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        # get polygon boundary coordinates -> add h3 for centroid of each vertex
        try:
            lonlats = list(geom.exterior.coords)
        except Exception:
            continue
        for lon, lat in lonlats:
            try:
                h3idx = h3.geo_to_h3(lat, lon, h3_res)
                h3_set.add(h3idx)
            except Exception:
                pass
    # expand by 2 rings to cover interiors
    initial = set(h3_set)
    for h in list(initial):
        h3_set.update(h3.k_ring(h, 2))
    # filter by testing cell centroid within wards union
    final = []
    for h in h3_set:
        lat, lon = h3.h3_to_geo(h)
        pt = Point(lon, lat)
        if union.contains(pt):
            final.append(h)
    echo(f"Selected {len(final)} H3 cells within wards.")
    return final

# ----------------------
# Synthetic NDVI generation per H3
# ----------------------
def synth_ndvi_for_h3(h3_cells, variant="base"):
    rows = []
    for i,h in enumerate(h3_cells):
        lat, lon = h3.h3_to_geo(h)
        # base NDVI pattern: higher NDVI in parks (use sinusoidal + noise) and lower in core city
        base = 0.45 + 0.25 * math.sin(lat*3.0) - 0.2 * math.exp(-((lon-77.6)**2 + (lat-12.95)**2)*5)
        # variant tweaks
        if variant == "base":
            trend = rng.uniform(-0.005, 0.005)  # slight change per year
            noise_scale = 0.05
        elif variant == "urban_hotspot":
            base = base - 0.15 * math.exp(-((lon-77.6)**2 + (lat-12.95)**2)*10)  # reduce green in hotspot center
            trend = rng.uniform(-0.01, -0.002)
            noise_scale = 0.07
        elif variant == "climate_trend":
            base = base - 0.05 * math.cos((lat+lon)*2.0)
            trend = rng.uniform(-0.02, 0.0)
            noise_scale = 0.06
        else:
            trend = rng.uniform(-0.005, 0.005)
            noise_scale = 0.05

        # simulate yearly NDVI stats 2013-2023 to compute mean/min/max and trend
        yearly = []
        for year in range(2013, 2024):
            year_noise = rng.normal(loc=0.0, scale=noise_scale)
            val = max(0.0, min(0.95, base + (year - 2013) * trend + year_noise))
            yearly.append(val)
        ndvi_mean = float(np.mean(yearly))
        ndvi_min = float(np.min(yearly))
        ndvi_max = float(np.max(yearly))
        # compute linear trend (slope per year) with numpy polyfit
        yrs = np.arange(2013, 2024)
        slope = np.polyfit(yrs, yearly, 1)[0]
        rows.append({"h3_id": h, "ndvi_mean": ndvi_mean, "ndvi_min": ndvi_min, "ndvi_max": ndvi_max, "trend_2013_2023": float(slope)})
    return pd.DataFrame(rows)

# ----------------------
# Synthetic LST generation per H3
# ----------------------
def synth_lst_for_h3(h3_cells, variant="base"):
    rows = []
    # baseline rural day/night temps approx
    for i,h in enumerate(h3_cells):
        lat, lon = h3.h3_to_geo(h)
        # base temps (Celsius)
        rural_day = 30 + 3.0 * math.cos(lat*5.0)
        rural_night = 22 + 2.0 * math.sin(lon*3.0)
        # urban increase near center
        distance_center = math.hypot(lon-77.6, lat-12.95)
        urban_warming = 4.0 * math.exp(-distance_center*6.0)
        if variant == "base":
            lst_day = rural_day + urban_warming + rng.normal(0, 1.2)
            lst_night = rural_night + urban_warming*0.8 + rng.normal(0, 0.9)
        elif variant == "urban_hotspot":
            lst_day = rural_day + urban_warming*1.5 + rng.normal(0, 1.6)
            lst_night = rural_night + urban_warming*1.2 + rng.normal(0, 1.1)
        elif variant == "climate_trend":
            lst_day = rural_day + urban_warming + 1.5 + rng.normal(0,1.4)
            lst_night = rural_night + urban_warming*0.9 + 1.0 + rng.normal(0,1.0)
        else:
            lst_day = rural_day + urban_warming + rng.normal(0,1.2)
            lst_night = rural_night + urban_warming*0.8 + rng.normal(0,0.9)
        # heat_island_index = day - rural baseline day (approx)
        heat_island_index = float((lst_day - rural_day))
        # temperature anomaly relative to long-term baseline (we simulate small anomalies)
        temp_anomaly = float(rng.normal(loc=0.5 if variant=="climate_trend" else 0.1, scale=0.5))
        rows.append({"h3_id": h, "lst_day": round(float(lst_day), 3), "lst_night": round(float(lst_night),3),
                     "heat_island_index": round(heat_island_index,3), "temperature_anomaly": round(temp_anomaly,3)})
    return pd.DataFrame(rows)

# ----------------------
# Synthetic DEM -> flood risk per H3
# ----------------------
def synth_flood_for_h3(h3_cells, variant="base"):
    # create elevation surface: gently varying with lat/lon, lower near certain drain lines
    elev = {}
    for h in h3_cells:
        lat, lon = h3.h3_to_geo(h)
        # base elevation in meters
        base_elev = 900 + 100*math.sin(lat*1.5) - 80*math.exp(-((lon-77.55)**2 + (lat-12.92)**2)*30)
        # variant tweaks
        if variant == "base":
            noise = rng.normal(0, 8)
        elif variant == "urban_hotspot":
            base_elev -= 6 * math.exp(-((lon-77.6)**2 + (lat-12.95)**2)*20)  # lower small depression
            noise = rng.normal(0, 12)
        elif variant == "climate_trend":
            base_elev -= 1.0  # slight sea-level effect
            noise = rng.normal(0, 10)
        else:
            noise = rng.normal(0, 8)
        elev[h] = float(base_elev + noise)

    # compute simple slope approximate: slope = max elevation difference to neighbors / distance
    # and a simplified flow accumulation: for each cell, flow to lowest neighbor, accumulate counts
    # Build neighbor map
    neighbors = {}
    for h in h3_cells:
        neighbors[h] = [n for n in h3.k_ring(h,1) if n in elev and n!=h]

    # compute slope
    slope_map = {}
    for h in h3_cells:
        lat, lon = h3.h3_to_geo(h)
        max_diff = 0.0
        for n in neighbors.get(h, []):
            lat2, lon2 = h3.h3_to_geo(n)
            d = haversine_m(lon, lat, lon2, lat2)
            if d == 0: continue
            diff = abs(elev[h] - elev[n]) / d * 100  # percent
            if diff > max_diff: max_diff = diff
        slope_map[h] = float(max_diff)

    # flow accumulation: naive approach
    flow = {h: 1.0 for h in h3_cells}  # start with unit area
    # do iterative downhill draining
    for _ in range(3):  # few passes to propagate
        for h in h3_cells:
            # find neighbor with lowest elevation
            nbrs = neighbors.get(h, [])
            if not nbrs:
                continue
            lowest = min(nbrs, key=lambda n: elev[n])
            if elev[lowest] < elev[h]:
                # send fraction of h's flow to lowest neighbor
                flow[lowest] += flow[h]*0.6
                # optionally reduce current cell
                flow[h] *= 0.4

    # compute flood risk index combining low elevation, high flow, low slope
    rows = []
    elev_values = list(elev.values())
    elev_min, elev_max = min(elev_values), max(elev_values)
    for h in h3_cells:
        norm_elev = (elev[h] - elev_min) / (elev_max - elev_min + 1e-9)
        # risk increases when elevation low -> invert
        elev_risk = 1.0 - norm_elev
        # high flow increases risk
        flow_val = float(flow.get(h, 0.0))
        # normalize flow roughly
        # compute rough percentiles
        rows.append({"h3_id": h,
                     "elevation_m": round(float(elev[h]),3),
                     "slope": round(float(slope_map[h]),3),
                     "flow_accumulation": round(flow_val,3),
                     "flood_risk_index": round(float(0.6*elev_risk + 0.3*(flow_val/ (1+flow_val)) + 0.1*(1.0/(1.0+slope_map[h]))),3)})
    return pd.DataFrame(rows)

# ----------------------
# Aggregation to wards and saving
# ----------------------
# ----------------------
# Aggregation to wards and saving  (FIXED VERSION)
# ----------------------
def aggregate_and_save(h3_df, wards_gdf, out_prefix, outdir):
    """
    h3_df: dataframe with column h3_id and numeric fields
    wards_gdf: GeoDataFrame of wards (EPSG:4326)
    out_prefix: base name, e.g. 'ndvi_base'
    """

    outdir = Path(outdir)

    # Construct GeoDataFrame for H3 polygons
    geo_rows = []
    for _, r in h3_df.iterrows():
        h = r["h3_id"]
        poly = h3_polygon(h)
        row = dict(r)
        row["geometry"] = poly
        geo_rows.append(row)

    # Create H3 GeoDataFrame
    h3_gdf = gpd.GeoDataFrame(geo_rows, geometry="geometry", crs="EPSG:4326")

    # ----- FIX 1: Convert centroid to numeric instead of geometry -----
    # Reproject for accurate centroid
    h3_gdf_proj = h3_gdf.to_crs(3857)
    centroids = h3_gdf_proj.geometry.centroid
    h3_gdf["centroid_x"] = centroids.x
    h3_gdf["centroid_y"] = centroids.y

    # NO SECOND GEOMETRY COLUMN ANYMORE ✔

    # Spatial join for wards
    if wards_gdf is None or wards_gdf.empty:
        h3_gdf["ward_id"] = None
        wards_agg = pd.DataFrame()
    else:
        wards = wards_gdf.to_crs(4326)

        # Create point layer for join
        h3_pts = gpd.GeoDataFrame(
            h3_gdf[["h3_id", "centroid_x", "centroid_y"]].copy(),
            geometry=gpd.points_from_xy(h3_gdf["centroid_x"], h3_gdf["centroid_y"]),
            crs="EPSG:4326"
        )

        joined = gpd.sjoin(h3_pts, wards, how="left", predicate="within")

        # Determine ward ID column
        ward_cols = [c for c in joined.columns if "ward" in c.lower()]
        ward_key = ward_cols[0] if ward_cols else wards.columns[0]

        joined["ward_id"] = joined[ward_key].astype(str)

        # Attach ward_id back
        h3_gdf = h3_gdf.merge(joined[["h3_id", "ward_id"]], on="h3_id", how="left")

        # Ward-level aggregation
        numeric_cols = [c for c in h3_df.columns if c != "h3_id"]
        wards_agg = h3_gdf.dissolve(by="ward_id", aggfunc="mean")[numeric_cols]
        wards_agg = wards_agg.reset_index()

    # ----- SAVE OUTPUTS -----

    out_geo = Path(outdir) / f"{out_prefix}_h3.geojson"
    out_csv = Path(outdir) / f"{out_prefix}_h3.csv"

    # GeoJSON (only one geometry column now)
    h3_gdf.to_file(out_geo, driver="GeoJSON")

    # CSV
    h3_gdf.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    # Wards CSV
    if wards_agg is not None and len(wards_agg) > 0:
        wards_csv = Path(outdir) / f"{out_prefix}_wards.csv"
        wards_agg.to_csv(wards_csv, index=False)
    else:
        wards_csv = None

    return str(out_geo), str(out_csv), str(wards_csv) if wards_csv else None
    

# ----------------------
# Main runner
# ----------------------
def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # wards
    wards_path = Path(args.wards) if args.wards else Path("/mnt/data/wards_master_enriched.geojson")
    wards_gdf = None
    if wards_path.exists():
        wards_gdf = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
        echo(f"Loaded wards: {len(wards_gdf)} features from {wards_path}")
    else:
        echo("No wards file provided or found at default; H3->ward linking will be synthetic.")

    h3_res = int(args.h3_res)
    variants = args.variants if args.variants else ["base","urban_hotspot","climate_trend"]
    echo(f"H3 resolution: {h3_res}")

    # build H3 cell set from wards if available, else create H3 set from bbox approx around Bangalore center
    if wards_gdf is not None:
        h3_cells = build_h3_grid_for_wards(wards_gdf, h3_res)
    else:
        # fallback: sample around Bengaluru center (lat~12.97, lon~77.59)
        echo("No wards -> sampling H3 cells around default bbox (Bengaluru).")
        center_lat, center_lon = 12.97, 77.59
        # take a small radius of cells (k_ring of center cell)
        center_h = h3.geo_to_h3(center_lat, center_lon, h3_res)
        h3_cells = list(h3.k_ring(center_h, 6))  # ~ area
    echo(f"Total H3 cells to generate: {len(h3_cells)}")

    # for each variant generate ndvi/lst/flood
    for variant in variants:
        echo(f"Generating synthetic data for variant: {variant}")
        ndvi_df = synth_ndvi_for_h3(h3_cells, variant=variant)
        lst_df  = synth_lst_for_h3(h3_cells, variant=variant)
        flood_df= synth_flood_for_h3(h3_cells, variant=variant)

        # merge h3-level tables for saving (if you want them separate keep separate)
        # Save NDVI
        ndvi_prefix = f"ndvi_{variant}"
        ndvi_geo, ndvi_csv, ndvi_wards = aggregate_and_save(ndvi_df, wards_gdf, ndvi_prefix, outdir)
        echo(f"Saved NDVI: {ndvi_geo}, {ndvi_csv}, wards: {ndvi_wards}")

        # Save LST
        lst_prefix = f"lst_{variant}"
        lst_geo, lst_csv, lst_wards = aggregate_and_save(lst_df, wards_gdf, lst_prefix, outdir)
        echo(f"Saved LST: {lst_geo}, {lst_csv}, wards: {lst_wards}")

        # Save Flood
        flood_prefix = f"flood_{variant}"
        flood_geo, flood_csv, flood_wards = aggregate_and_save(flood_df, wards_gdf, flood_prefix, outdir)
        echo(f"Saved Flood: {flood_geo}, {flood_csv}, wards: {flood_wards}")

    echo("All variants generated. Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wards", default=None, help="C:\data\geo\wards\wards_master_enriched.geojson")
    p.add_argument("--h3_res", default=8, type=int)
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--variants", nargs="+", default=["base","urban_hotspot","climate_trend"], help="list of variants to generate")
    args = p.parse_args()
    main(args)
