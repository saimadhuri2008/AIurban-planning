#!/usr/bin/env python3
"""
income_housing_pipeline.py

Produces income & housing datasets linked to wards and H3.

Outputs:
 - income_housing_h3_{variant}.geojson  and .csv
 - income_housing_wards_{variant}.csv

Defaults:
 - wards: /mnt/data/wards_master_enriched.geojson
 - census: /mnt/data/bangalore-ward-level-census-2011.csv

Usage:
 python src/data_pipeline/income_housing_pipeline.py --outdir ./outputs --h3_res 8 --variants base gentrification inequality
"""
import argparse
from pathlib import Path
import sys, math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import h3

RNG_SEED = 3030
rng = np.random.RandomState(RNG_SEED)

# --- helpers ---
def h3_poly(h):
    b = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(p[1], p[0]) for p in b])

def build_cells(wards_gdf, res):
    union = wards_gdf.unary_union
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
            except:
                pass
    for h in list(hset):
        hset.update(h3.k_ring(h,2))
    cells = [c for c in hset if union.contains(Point(h3.h3_to_geo(c)[1], h3.h3_to_geo(c)[0]))]
    return cells

def safe_write_geojson(gdf, path):
    geom_cols = [c for c in gdf.columns if gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            gdf = gdf.drop(columns=[gc])
    gdf = gdf.set_geometry("geometry")
    gdf.to_file(path, driver="GeoJSON")

# --- synthetic generator for income & housing ---
def generate_income_housing(cells, variant="base"):
    rows=[]
    for h in cells:
        lat, lon = h3.h3_to_geo(h)
        # base income distribution (percent)
        # lower/middle/upper percentages sum to 1
        center_dist = math.exp(-((lat-12.97)**2 + (lon-77.59)**2)*25)
        lower = max(0.05, 0.6*(1-center_dist) + rng.normal(0,0.05))
        upper = max(0.02, 0.05 + 0.3*center_dist + rng.normal(0,0.03))
        middle = max(0.1, 1.0 - lower - upper)
        # rental price per month (INR)
        base_rent = 8000 + 30000*center_dist + rng.normal(0,500)
        # property value per sq.m (INR)
        base_prop = 20000 + 120000*center_dist + rng.normal(0,2000)
        # slum percentage higher in periphery inversely proportional to center_dist
        slum_pct = min(0.9, max(0.0, 0.3*(1-center_dist) + rng.normal(0,0.05)))
        # housing type probabilities influenced by center
        if variant == "gentrification":
            base_rent *= 1.25; base_prop *= 1.5; upper *= 1.2; lower *= 0.8
        elif variant == "inequality":
            base_rent *= 1.05; base_prop *= 1.1; upper *= 1.3; lower *= 0.7
        # normalize income percentages
        tot = lower + middle + upper
        lower /= tot; middle /= tot; upper /= tot

        # decide dominant housing type
        if slum_pct > 0.35:
            housing = "slum"
        elif center_dist > 0.6:
            housing = "apartment"
        else:
            housing = "independent"

        rows.append({
            "h3_id": h,
            "income_lower_pct": round(float(lower),3),
            "income_middle_pct": round(float(middle),3),
            "income_upper_pct": round(float(upper),3),
            "rental_prices": round(float(max(1000, base_rent)),2),
            "property_values": round(float(max(1000, base_prop)),2),
            "slum_population_percentage": round(float(slum_pct),3),
            "housing_type": housing
        })
    return pd.DataFrame(rows)

# --- try to read census if available for ward-level anchoring ---
def try_read_census(cpath):
    try:
        df = pd.read_csv(cpath)
        return df
    except:
        return None

# --- main ---
def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    wards_path = Path(args.wards) if args.wards else Path("/mnt/data/wards_master_enriched.geojson")
    if not wards_path.exists():
        print("[ERR] wards file not found:", wards_path); sys.exit(1)
    wards = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
    h3_res = int(args.h3_res)
    variants = args.variants if args.variants else ["base","gentrification","inequality"]

    cells = build_cells(wards, h3_res)
    if len(cells) == 0:
        center_h = h3.geo_to_h3(12.97, 77.59, h3_res)
        cells = list(h3.k_ring(center_h,6))
    print("Generating income & housing for", len(cells), "H3 cells")

    # optionally read census to bias ward aggregates
    census_df = None
    if args.census_csv and Path(args.census_csv).exists():
        census_df = try_read_census(Path(args.census_csv))
    elif Path("/mnt/data/bangalore-ward-level-census-2011.csv").exists():
        census_df = try_read_census(Path("/mnt/data/bangalore-ward-level-census-2011.csv"))

    for var in variants:
        h3_df = generate_income_housing(cells, variant=var)
        # convert to GeoDataFrame
        geo_rows=[]
        for _,r in h3_df.iterrows():
            geo_rows.append({"h3_id": r["h3_id"], **{k:v for k,v in r.items() if k!="h3_id"}, "geometry": h3_poly(r["h3_id"])})
        h3_gdf = gpd.GeoDataFrame(geo_rows, geometry="geometry", crs="EPSG:4326")

        # attach ward via centroid sjoin
        # Reproject to a metric CRS for correct centroid
        h3_proj = h3_gdf.to_crs(3857)
        centroids = h3_proj.geometry.centroid

# Put back into EPSG:4326 for spatial join with wards
        h3_pts = gpd.GeoDataFrame( 
        {"h3_id": h3_gdf["h3_id"]},
        geometry=gpd.points_from_xy(centroids.x, centroids.y),
        crs="EPSG:3857"
        ).to_crs(4326)

        joined = gpd.sjoin(h3_pts, wards[[wards.columns[0],"geometry"]], how="left", predicate="within")
        ward_key = [c for c in joined.columns if "ward" in c.lower()]
        ward_col = ward_key[0] if ward_key else wards.columns[0]
        joined = joined.rename(columns={ward_col: "ward_id"}) if ward_col != "ward_id" else joined
        h3_gdf = h3_gdf.merge(joined[["h3_id","ward_id"]], on="h3_id", how="left")

        # ward aggregates: for numeric columns compute means; for housing_type compute mode
        numeric_cols = ["rental_prices","property_values","slum_population_percentage","income_lower_pct","income_middle_pct","income_upper_pct"]
        wards_mean = h3_gdf.dropna(subset=["ward_id"]).dissolve(by="ward_id", aggfunc="mean")[numeric_cols].reset_index()

        # housing type mode per ward
        types = h3_gdf.dropna(subset=["ward_id"]).groupby("ward_id")["housing_type"].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else "")
        wards_mean["housing_type"] = wards_mean["ward_id"].map(types)

        # save
        out_geo = outdir / f"income_housing_h3_{var}.geojson"
        out_csv = outdir / f"income_housing_h3_{var}.csv"
        out_wards = outdir / f"income_housing_wards_{var}.csv"

        safe_write_geojson(h3_gdf, out_geo)
        h3_gdf.drop(columns=["geometry"]).to_csv(out_csv, index=False)
        wards_mean.to_csv(out_wards, index=False)

        print("Saved:", out_geo, out_csv, out_wards)

    print("Income & Housing pipeline done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wards", help="path to wards geojson (default /mnt/data/wards_master_enriched.geojson)")
    p.add_argument("--census_csv", help="optional census CSV path (default /mnt/data/bangalore-ward-level-census-2011.csv)")
    p.add_argument("--h3_res", type=int, default=8)
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--variants", nargs="+", default=["base","gentrification","inequality"])
    args = p.parse_args()
    main(args)
