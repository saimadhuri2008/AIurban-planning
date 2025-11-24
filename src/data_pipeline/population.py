#!/usr/bin/env python3
"""
population_h3_pipeline.py

Produces:
 - population_h3.geojson  (one feature per H3 cell with population_h3, population_density_h3, ward_id, household_count, avg_household_size)
 - population_h3.csv
 - population_wards.csv   (ward-level original + aggregated fields)

Usage:
 python src/data_pipeline/population_h3_pipeline.py \
   --wards "/path/to/wards_master_enriched.geojson" \
   --census_csv "/path/to/bangalore-ward-level-census-2011.csv" \
   --h3_res 8 \
   --outdir "./outputs"
"""

import argparse
from pathlib import Path
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, shape, Polygon
import h3


# -----------------------
# Utilities
# -----------------------
def find_column(df_cols, keywords):
    """Return first column name in df_cols that contains any keyword (case-insensitive)."""
    for c in df_cols:
        lower = str(c).lower()
        for kw in keywords:
            if kw in lower:
                return c
    return None


def ward_id_candidates(gdf):
    for c in gdf.columns:
        if c.lower() in ("ward_id", "ward", "wardno", "ward_no", "wardcode", "ward_code"):
            return c
    # fallback to first column
    return gdf.columns[0]


def polygon_to_h3_set(poly, res):
    """
    Use h3.polyfill on the GeoJSON mapping of a polygon (lat/lon).
    poly: shapely polygon (assumed lon/lat order)
    returns set of h3 indices
    """
    geom = mapping(poly)
    # h3.polyfill expects geo_json style: coordinates as [ [ [lon, lat], ... ] ]? It expects lat/lon order inside.
    # Create geo_json with coordinates in [lat, lon] pairs for the library:
    def coords_to_latlon(coords):
        # coords: list of (lon, lat) pairs (shapely uses (x=lon,y=lat))
        return [(y, x) for (x, y) in coords]

    if geom["type"] == "Polygon":
        # single polygon
        shell = coords_to_latlon(geom["coordinates"][0])
        holes = [coords_to_latlon(r) for r in geom["coordinates"][1:]] if len(geom["coordinates"]) > 1 else []
        geo_json = {"type": "Polygon", "coordinates": [shell] + holes}
    elif geom["type"] == "MultiPolygon":
        # flatten into one combined polygon by polyfill over each component
        cells = set()
        for poly_coords in geom["coordinates"]:
            shell = coords_to_latlon(poly_coords[0])
            holes = [coords_to_latlon(r) for r in poly_coords[1:]] if len(poly_coords) > 1 else []
            gj = {"type": "Polygon", "coordinates": [shell] + holes}
            cells.update(h3.polyfill(gj, res, geo_json_conformant=True))
        return cells
    else:
        return set()

    return set(h3.polyfill(geo_json, res, geo_json_conformant=True))


def h3_to_polygon(h):
    """Return shapely Polygon for an H3 index (lon/lat order)."""
    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    # boundary is list of (lat, lon) pairs -> convert to (lon, lat)
    coords = [(p[1], p[0]) for p in boundary]
    return Polygon(coords)


def compute_area_km2(geom):
    """
    Compute area in km^2 for a shapely geometry.
    We reproject to EPSG:3857 (WebMercator) for an approximate area in m^2, then convert to km^2.
    """
    if geom is None or geom.is_empty:
        return 0.0
    g = gpd.GeoSeries([geom], crs="EPSG:4326")
    try:
        g = g.to_crs(epsg=3857)
    except Exception:
        # if reprojection fails, return 0
        return 0.0
    area_m2 = float(g.iloc[0].area)
    return area_m2 / 1e6


# -----------------------
# Main pipeline
# -----------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use the uploaded CSV path by default if none provided
    default_census = Path("C:\\Users\\jbhuv\\Downloads\\bangalore-ward-level-census-2011.csv")
    census_csv = Path(args.census_csv) if args.census_csv else default_census
    if not census_csv.exists():
        raise FileNotFoundError(f"Census CSV not found: {census_csv}")

    default_wards = Path("C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
    wards_path = Path(args.wards) if args.wards else default_wards
    if not wards_path.exists():
        raise FileNotFoundError(f"Wards GeoJSON not found: {wards_path}")

    h3_res = int(args.h3_res)
    print(f"Reading wards from: {wards_path}")
    wards = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
    print(f"Loaded {len(wards)} ward polygons.")

    print(f"Reading census CSV from: {census_csv}")
    census = pd.read_csv(census_csv, dtype=str)  # read as string to detect columns robustly
    print("Census columns:", list(census.columns)[:20])

    # Attempt to detect ward id/name and population/household columns
    population_col = find_column(census.columns, ["population", "total_population", "total pop", "persons", "persons_total", "pop_total"])
    household_col = find_column(census.columns, ["household", "households", "house hold", "hh", "house_hold"])
    ward_col_csv = find_column(census.columns, ["ward", "wardno", "ward_no", "ward_id", "wardcode", "ward_code", "ward name", "wardname"])

    if population_col is None:
        warnings.warn("No population column detected in census CSV. Please provide a CSV with ward population. Script will fill NaNs with 0.")
    else:
        print(f"Detected population column in CSV: {population_col}")

    if household_col is None:
        print("No household column detected in census CSV. Household counts will be estimated where needed.")
    else:
        print(f"Detected household column in CSV: {household_col}")

    if ward_col_csv is None:
        print("No ward id/name column detected in CSV. Attempting positional matching (order).")
    else:
        print(f"Detected ward id column in CSV: {ward_col_csv}")

    # Normalize numeric fields in census
    def to_numeric_safe(s):
        if pd.isna(s):
            return np.nan
        # remove commas and whitespace
        ss = str(s).replace(",", "").strip()
        try:
            return float(ss)
        except:
            return np.nan

    census_numeric = census.copy()
    if population_col:
        census_numeric["population_total_raw"] = census_numeric[population_col].apply(to_numeric_safe)
    else:
        census_numeric["population_total_raw"] = np.nan

    if household_col:
        census_numeric["household_count_raw"] = census_numeric[household_col].apply(to_numeric_safe)
    else:
        census_numeric["household_count_raw"] = np.nan

    # Build mapping from wards GeoJSON to census rows
    ward_id_col = ward_id_candidates(wards)
    print("Ward id column in wards GeoJSON used:", ward_id_col)

    # Create output ward dataframe with ward_id and geometry
    wards_df = wards[[ward_id_col, "geometry"]].copy()
    wards_df = wards_df.rename(columns={ward_id_col: "ward_id"})
    wards_df["ward_id"] = wards_df["ward_id"].astype(str)

    # Map census rows to wards
    # Strategy:
    # 1) If ward_col_csv exists, try to map by string equality (after normalization)
    # 2) else if counts match number of wards, positional mapping
    # 3) else try numeric ward id match
    census_numeric["ward_key_csv"] = census_numeric[ward_col_csv].astype(str).str.strip() if ward_col_csv else None

    mapping_idx = {}
    if ward_col_csv:
        # try exact matches: normalize both sides
        csv_keys = census_numeric["ward_key_csv"].astype(str).str.lower().str.replace(r"[^0-9a-z]+", "", regex=True)
        ward_keys = wards_df["ward_id"].astype(str).str.lower().str.replace(r"[^0-9a-z]+", "", regex=True)
        # build dict from csv keys to index
        csv_map = {}
        for i, k in enumerate(csv_keys):
            if k not in csv_map:
                csv_map[k] = i  # first occurrence
        # attempt mapping
        mapped = []
        for j, wk in enumerate(ward_keys):
            if wk in csv_map:
                mapping_idx[j] = csv_map[wk]
        if len(mapping_idx) == 0:
            # try numeric matches: extract digits
            csv_nums = census_numeric[ward_col_csv].astype(str).str.extract(r"(\d+)", expand=False)
            ward_nums = wards_df["ward_id"].astype(str).str.extract(r"(\d+)", expand=False)
            for j, wn in enumerate(ward_nums):
                if pd.isna(wn):
                    continue
                candidates = census_numeric[csv_nums == wn]
                if len(candidates) > 0:
                    mapping_idx[j] = candidates.index[0]
    # If mapping still empty and row counts equal, positional mapping
    if len(mapping_idx) == 0 and len(census_numeric) == len(wards_df):
        print("Applying positional mapping census row -> wards (same number of records).")
        for j in range(len(wards_df)):
            mapping_idx[j] = j

    if len(mapping_idx) == 0:
        warnings.warn("Could not robustly map census CSV rows to wards. Assigning population to wards by order of rows where possible; unmatched wards get NaN.")
        # attempt best-effort positional mapping for min(lengths)
        for j in range(min(len(wards_df), len(census_numeric))):
            mapping_idx[j] = j

    # Build ward-level DataFrame with population_total and household_count
    ward_rows = []
    for j, wrow in wards_df.reset_index(drop=True).iterrows():
        cidx = mapping_idx.get(j, None)
        if cidx is not None:
            pop = float(census_numeric.at[cidx, "population_total_raw"]) if not pd.isna(census_numeric.at[cidx, "population_total_raw"]) else np.nan
            hh  = float(census_numeric.at[cidx, "household_count_raw"]) if not pd.isna(census_numeric.at[cidx, "household_count_raw"]) else np.nan
            ward_rows.append({
                "ward_id": str(wrow["ward_id"]),
                "geometry": wrow["geometry"],
                "population_total": pop,
                "household_count": hh
            })
        else:
            ward_rows.append({
                "ward_id": str(wrow["ward_id"]),
                "geometry": wrow["geometry"],
                "population_total": np.nan,
                "household_count": np.nan
            })

    ward_pop_gdf = gpd.GeoDataFrame(ward_rows, geometry="geometry", crs="EPSG:4326")

    # If population_total missing for some wards, warn. You can choose to generate synthetic values if needed.
    missing_pop = ward_pop_gdf["population_total"].isna().sum()
    if missing_pop > 0:
        warnings.warn(f"{missing_pop} wards have missing population_total. You may provide a better census CSV to avoid synthetic data.")

    # Fill household_count if missing by estimating avg_household_size (default 4.0)
    default_avg_household_size = float(args.default_hh_size)
    ward_pop_gdf["avg_household_size"] = ward_pop_gdf.apply(
        lambda r: (r["population_total"] / r["household_count"]) if (not pd.isna(r["population_total"]) and not pd.isna(r["household_count"]) and r["household_count"]>0) else default_avg_household_size,
        axis=1
    )
    # If household_count missing but population present, estimate households:
    ward_pop_gdf["household_count"] = ward_pop_gdf.apply(
        lambda r: r["household_count"] if (not pd.isna(r["household_count"])) else (r["population_total"] / r["avg_household_size"] if not pd.isna(r["population_total"]) else np.nan),
        axis=1
    )

    # Ensure population_total numeric (fill missing with 0 optionally)
    if args.fill_missing_pop_with_zero:
        ward_pop_gdf["population_total"] = ward_pop_gdf["population_total"].fillna(0.0)

    # Now for each ward, compute H3 cells and area-weighted population allocation
    print("Computing H3 cells per ward and allocating population (area-weighted). This can take some time for many wards.")
    h3_rows = []
    total_wards = len(ward_pop_gdf)
    for idx, w in ward_pop_gdf.iterrows():
        ward_id = str(w["ward_id"])
        poly = w["geometry"]
        if poly is None or poly.is_empty:
            continue
        cells = polygon_to_h3_set(poly, h3_res)
        if not cells:
            continue
        # For each cell compute intersection area with ward polygon
        # Build polygons for each cell, intersect and compute area_km2
        areas = []
        cell_polys = {}
        for c in cells:
            poly_c = h3_to_polygon(c)
            inter = poly.intersection(poly_c)
            if inter is None or inter.is_empty:
                a_km2 = 0.0
            else:
                a_km2 = compute_area_km2(inter)
            if a_km2 > 0.0:
                areas.append((c, a_km2, inter))
                cell_polys[c] = (poly_c, inter)
        if not areas:
            continue
        # total area for normalization
        total_area = sum([a for (_, a, _) in areas])
        if total_area <= 0:
            continue
        # population to allocate
        pop_total = float(w["population_total"]) if not pd.isna(w["population_total"]) else 0.0
        hh_count = float(w["household_count"]) if not pd.isna(w["household_count"]) else np.nan
        avg_hh = float(w["avg_household_size"]) if not pd.isna(w["avg_household_size"]) else default_avg_household_size

        for (c, a_km2, inter_geom) in areas:
            frac = a_km2 / total_area
            pop_cell = pop_total * frac
            # household count per cell = fraction * ward household_count
            hh_cell = hh_count * frac if not np.isnan(hh_count) else (pop_cell / avg_hh if avg_hh>0 else np.nan)
            # density = pop_cell / a_km2 (people per km2); if a_km2 zero skip
            density = pop_cell / a_km2 if a_km2 > 0 else 0.0
            h3_rows.append({
                "ward_id": ward_id,
                "h3_id": c,
                "population_h3": pop_cell,
                "population_density_h3": density,
                "household_count_h3": hh_cell,
                "avg_household_size_h3": (pop_cell / hh_cell) if (hh_cell and hh_cell>0) else avg_hh,
                "h3_area_km2": a_km2
            })

    if len(h3_rows) == 0:
        raise RuntimeError("No H3 rows computed (possible geometry/polyfill issue).")

    h3_df = pd.DataFrame(h3_rows)
    # round numeric columns
    for col in ["population_h3", "population_density_h3", "household_count_h3", "avg_household_size_h3", "h3_area_km2"]:
        if col in h3_df.columns:
            h3_df[col] = h3_df[col].astype(float)

    # Create GeoDataFrame of H3 polygons
    print("Building H3 polygon GeoDataFrame for output...")
    geo_features = []
    for _, r in h3_df.iterrows():
        h = r["h3_id"]
        poly = h3_to_polygon(h)
        props = r.to_dict()
        props.pop("h3_id", None)
        geo_features.append({"h3_id": h, **props, "geometry": poly})
    h3_gdf = gpd.GeoDataFrame(geo_features, geometry="geometry", crs="EPSG:4326")

    # Save outputs
    out_h3_geo = outdir / f"population_h3_res{h3_res}.geojson"
    out_h3_csv = outdir / f"population_h3_res{h3_res}.csv"
    out_wards_csv = outdir / "population_wards.csv"

    # cleanup possible extra geometry columns before saving (defensive)
    geom_cols = [c for c in h3_gdf.columns if h3_gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            h3_gdf = h3_gdf.drop(columns=[gc])
    h3_gdf = h3_gdf.set_geometry("geometry")

    print("Writing H3 GeoJSON:", out_h3_geo)
    h3_gdf.to_file(out_h3_geo, driver="GeoJSON")
    h3_gdf.drop(columns=["geometry"]).to_csv(out_h3_csv, index=False)

    # Ward-level CSV: add population_total/household_count/avg_household_size and computed totals from H3 (sanity check)
    ward_agg = h3_df.groupby("ward_id").agg({
        "population_h3": "sum",
        "household_count_h3": "sum"
    }).reset_index().rename(columns={"population_h3": "population_from_h3", "household_count_h3": "households_from_h3"})

    ward_out = ward_pop_gdf.drop(columns=["geometry"]).merge(ward_agg, on="ward_id", how="left")
    # fill numeric NaNs with zeros for convenience
    ward_out["population_from_h3"] = ward_out["population_from_h3"].fillna(0.0)
    ward_out["households_from_h3"] = ward_out["households_from_h3"].fillna(0.0)

    # Save ward-level CSV
    ward_out.to_csv(out_wards_csv, index=False)

    print("Saved outputs:")
    print(" - H3 geojson:", out_h3_geo)
    print(" - H3 csv:    ", out_h3_csv)
    print(" - Ward csv:  ", out_wards_csv)
    print("Done.")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Population -> H3 allocation pipeline")
    p.add_argument("--wards", required=False, help="Path to wards geojson (default /mnt/data/wards_master_enriched.geojson)")
    p.add_argument("--census_csv", required=False, help="Path to census CSV (default /mnt/data/bangalore-ward-level-census-2011.csv)")
    p.add_argument("--h3_res", required=False, default=8, type=int, help="H3 resolution (default 8)")
    p.add_argument("--outdir", required=False, default="./outputs", help="Output directory")
    p.add_argument("--default_hh_size", required=False, default=4.0, type=float, help="Default average household size if not present (default 4.0)")
    p.add_argument("--fill_missing_pop_with_zero", action="store_true", help="If set, fill missing ward populations with zero")
    args = p.parse_args()
    main(args)
