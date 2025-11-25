#!/usr/bin/env python3
"""
economy_pipeline.py

Produces economy datasets linked to wards and H3.

Outputs:
 - economy_h3_{variant}.geojson  and .csv  (per-H3)
 - economy_wards_{variant}.csv   (ward aggregates)

Defaults (use uploaded files if present):
 - wards: /mnt/data/wards_master_enriched.geojson
 - bescom CSV (optional real data): /mnt/data/BESCOM_Category_wise_installations_and_Consumption_upto_Mar_2022.csv
 - electricity CSV (optional): /mnt/data/electricity.csv

Usage:
 python src/data_pipeline/economy_pipeline.py --outdir ./outputs --h3_res 8 --variants base hotspot growth
"""
import argparse
from pathlib import Path
import math, json, sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import h3
from datetime import datetime

RNG_SEED = 2025
rng = np.random.RandomState(RNG_SEED)

# --- helpers ---
def h3_poly(h):
    b = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(p[1], p[0]) for p in b])

def h3_centroid(h):
    lat, lon = h3.h3_to_geo(h)
    return lat, lon

def build_h3_cells_for_wards(wards_gdf, res):
    # produce a covering set of H3 cells intersecting wards
    union = wards_gdf.unary_union
    # sample boundary vertices -> get their h3 and expand
    hset = set()
    for geom in wards_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.exterior.coords)
        except Exception:
            continue
        for lon, lat in coords:
            try:
                hset.add(h3.geo_to_h3(lat, lon, res))
            except Exception:
                continue
    # expand
    initial = list(hset)
    for h in initial:
        hset.update(h3.k_ring(h, 2))
    # filter cells whose centroid lies in union
    cells = [c for c in hset if union.contains(Point(h3.h3_to_geo(c)[1], h3.h3_to_geo(c)[0]))]
    return cells

def safe_write_geojson(gdf, path):
    # remove extra geometry columns
    geom_cols = [c for c in gdf.columns if gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            gdf = gdf.drop(columns=[gc])
    gdf = gdf.set_geometry("geometry")
    gdf.to_file(path, driver="GeoJSON")

# --- synthetic generators ---
def generate_economy_h3(cells, variant="base"):
    rows = []
    for i,h in enumerate(cells):
        lat, lon = h3.h3_to_geo(h)
        # base densities per km2 (synthetic)
        base_it = 50 * math.exp(-((lat-12.97)**2 + (lon-77.59)**2)*20) + 5  # hotspots near center
        base_non_it = 200 * (1 - math.exp(-((lat-12.97)**2 + (lon-77.59)**2)*5)) + 20
        industrial = max(0, int(rng.poisson(0.2 + 3*math.exp(-((lon-77.55)**2+(lat-12.95)**2)*10))))
        commercial = max(0, int(rng.poisson(1.5 + 5*math.exp(-((lon-77.58)**2+(lat-12.96)**2)*8))))
        informal = max(0, int((base_non_it*0.1) * (1.0 + rng.normal(0,0.2))))
        nightlight = float( (base_it*0.3 + base_non_it*0.7)/10.0 + rng.normal(0,0.5) )
        # variant tweaks
        if variant == "hotspot":
            base_it *= 1.6; nightlight *= 1.4
        elif variant == "growth":
            base_it *= 1.2; base_non_it *= 1.1; nightlight *= 1.1
        rows.append({
            "h3_id": h,
            "it_job_density": max(0.0, float(base_it + rng.normal(0, 5))),
            "non_it_job_density": max(0.0, float(base_non_it + rng.normal(0,10))),
            "informal_sector_jobs": int(max(0, informal)),
            "industrial_units_count": int(industrial),
            "commercial_units_count": int(commercial),
            "night_light_intensity": round(max(0.0, nightlight),3)
        })
    return pd.DataFrame(rows)

# --- try to extract from real CSVs if present (best-effort) ---
def try_extract_from_bescom(bescom_path):
    try:
        df = pd.read_csv(bescom_path)
        return df
    except Exception:
        return None

def try_extract_from_electricity(elec_path):
    try:
        df = pd.read_csv(elec_path)
        return df
    except Exception:
        return None

# --- main ---
def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    wards_default = Path("/mnt/data/wards_master_enriched.geojson")
    wards_path = Path(args.wards) if args.wards else wards_default
    if not wards_path.exists():
        print("[WARN] Wards file not found at", wards_path, "-> aborting.")
        sys.exit(1)
    wards = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
    h3_res = int(args.h3_res)
    variants = args.variants if args.variants else ["base","hotspot","growth"]

    # build H3 cells covering wards
    cells = build_h3_cells_for_wards(wards, h3_res)
    if len(cells) == 0:
        # fallback: build k_ring around city center
        center_h = h3.geo_to_h3(12.97, 77.59, h3_res)
        cells = list(h3.k_ring(center_h, 6))

    print(f"Generating economy datasets for {len(cells)} H3 cells, H3 res {h3_res} ...")

    # attempt real data extract
    bescom_df = None
    if args.bescom_csv and Path(args.bescom_csv).exists():
        bescom_df = try_extract_from_bescom(Path(args.bescom_csv))
    elif Path("/mnt/data/BESCOM_Category_wise_installations_and_Consumption_upto_Mar_2022.csv").exists():
        bescom_df = try_extract_from_bescom(Path("/mnt/data/BESCOM_Category_wise_installations_and_Consumption_upto_Mar_2022.csv"))

    elec_df = None
    if args.electricity_csv and Path(args.electricity_csv).exists():
        elec_df = try_extract_from_electricity(Path(args.electricity_csv))
    elif Path("/mnt/data/electricity.csv").exists():
        elec_df = try_extract_from_electricity(Path("/mnt/data/electricity.csv"))

    # If real dataframes are available, you can implement mapping logic here.
    # For now we use synthetic generator but if bescom/elec present we slightly bias nightlight
    for var in variants:
        h3_df = generate_economy_h3(cells, variant=var)
        # if electricity data present, bump night_light_intensity using a small bias
        if elec_df is not None:
            bias = 0.5
            h3_df["night_light_intensity"] = h3_df["night_light_intensity"] * (1.0 + bias*0.1)
        # build GeoDataFrame
        geo_rows = []
        for _, r in h3_df.iterrows():
            geo_rows.append({"h3_id": r["h3_id"], **{k:v for k,v in r.items() if k!="h3_id"}, "geometry": h3_poly(r["h3_id"])})
        h3_gdf = gpd.GeoDataFrame(geo_rows, geometry="geometry", crs="EPSG:4326")

        # spatial join to wards via centroid
        h3_proj = h3_gdf.to_crs(epsg=3857)   # project to metric CRS
        h3_pts = gpd.GeoDataFrame(
        {"h3_id": h3_gdf["h3_id"]},
        geometry=h3_proj.geometry.centroid.to_crs(epsg=4326),  # convert back to lat/lon
        crs="EPSG:4326"
        )

        joined = gpd.sjoin(h3_pts, wards[[wards.columns[0],"geometry"]], how="left", predicate="within")
        ward_key = [c for c in joined.columns if "ward" in c.lower()]
        ward_col = ward_key[0] if ward_key else wards.columns[0]
        joined = joined.rename(columns={ward_col: "ward_id"}) if ward_col != "ward_id" else joined
        h3_gdf = h3_gdf.merge(joined[["h3_id","ward_id"]], on="h3_id", how="left")

        # ward aggregates
        numeric_cols = ["it_job_density","non_it_job_density","informal_sector_jobs","industrial_units_count","commercial_units_count","night_light_intensity"]
        wards_agg = h3_gdf.dropna(subset=["ward_id"]).dissolve(by="ward_id", aggfunc="mean")[numeric_cols]
        wards_agg = wards_agg.reset_index()

        # save outputs
        out_geo = outdir / f"economy_h3_{var}.geojson"
        out_csv = outdir / f"economy_h3_{var}.csv"
        out_ward_csv = outdir / f"economy_wards_{var}.csv"

        safe_write_geojson(h3_gdf, out_geo)
        h3_gdf.drop(columns=["geometry"]).to_csv(out_csv, index=False)
        wards_agg.to_csv(out_ward_csv, index=False)

        print("Saved:", out_geo, out_csv, out_ward_csv)

    print("Economy pipeline done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wards", help="path to wards geojson (default /mnt/data/wards_master_enriched.geojson)")
    p.add_argument("--bescom_csv", help="optional BESCOM CSV path")
    p.add_argument("--electricity_csv", help="optional electricity CSV path")
    p.add_argument("--h3_res", type=int, default=8)
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--variants", nargs="+", default=["base","hotspot","growth"])
    args = p.parse_args()
    main(args)
