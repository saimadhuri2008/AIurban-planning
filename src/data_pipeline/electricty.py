#!/usr/bin/env python3
"""
bescom_to_wards_h3.py

Reads BESCOM / electricity CSVs, attempts to extract real substation/transformer attributes,
synthesizes missing values, links each asset to wards and H3 cells, and writes GeoJSON/CSV outputs.

Outputs:
 - electricity_assets.geojson  (points: transformers / substations)
 - electricity_assets.csv
 - h3_aggregation.csv   (optional aggregation per H3 cell)

Usage:
 python bescom_to_wards_h3.py --bescom_csv path/to/BESCOM...csv --electricity_csv path/to/electricity.csv --wards_geojson path/to/wards.geojson --h3_res 8 --outdir ./outputs

If wards_geojson is not provided, wards linking will be synthetic (ward_id generated).
"""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import h3

# Deterministic randomness
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_csv_try(filepath: Path):
    if not filepath.exists():
        print(f"[WARN] CSV not found: {filepath}")
        return None
    try:
        return pd.read_csv(filepath, dtype=str)
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return None


def find_latlon_columns(df: pd.DataFrame):
    if df is None:
        return None, None
    lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude", "y", "ycoord")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "x", "xcoord")]
    # fallback: look for columns containing 'lat' or 'lon'
    if not lat_cols:
        lat_cols = [c for c in df.columns if "lat" in c.lower()]
    if not lon_cols:
        lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
    return (lat_cols[0] if lat_cols else None, lon_cols[0] if lon_cols else None)


def build_assets_from_csvs(bescom_df: pd.DataFrame, elec_df: pd.DataFrame):
    """
    Create an asset table (one row per transformer or substation)
    Attempt to use existing ids/names and lat/lon if present, otherwise create a geometry column (None)
    so GeoDataFrame can be constructed safely. Later code will synthesize points if needed.
    """
    # Prefer electric csv rows, else bescom, else empty DataFrame
    assets = None
    if elec_df is not None and len(elec_df) > 0:
        assets = elec_df.copy()
    elif bescom_df is not None and len(bescom_df) > 0:
        assets = bescom_df.copy()
    else:
        assets = pd.DataFrame()

    # Clean column names (strip whitespace)
    assets.columns = [c.strip() for c in assets.columns]

    # Try to detect lat/lon columns
    def find_latlon_columns_local(df):
        lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude", "y", "ycoord")]
        lon_cols = [c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "x", "xcoord")]
        if not lat_cols:
            lat_cols = [c for c in df.columns if "lat" in c.lower()]
        if not lon_cols:
            lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
        return (lat_cols[0] if lat_cols else None, lon_cols[0] if lon_cols else None)

    latcol, loncol = find_latlon_columns_local(assets)

    # If lat/lon exist, coerce and create geometry Points and drop rows missing coords
    if latcol and loncol:
        assets[latcol] = pd.to_numeric(assets[latcol].astype(str).str.strip(), errors="coerce")
        assets[loncol] = pd.to_numeric(assets[loncol].astype(str).str.strip(), errors="coerce")
        assets = assets.dropna(subset=[latcol, loncol]).reset_index(drop=True)
        geoms = [Point(xy) for xy in zip(assets[loncol].astype(float), assets[latcol].astype(float))]
        assets["geometry"] = geoms
        gdf = gpd.GeoDataFrame(assets, geometry="geometry", crs="EPSG:4326")
    else:
        # No coords: ensure there's a geometry column (fill with None). Later stages will synthesize locations.
        assets["geometry"] = [None] * len(assets)
        # Now it's safe to create a GeoDataFrame
        gdf = gpd.GeoDataFrame(assets, geometry="geometry", crs="EPSG:4326")

    # Standardize IDs / names
    if "substation_id" not in gdf.columns and "transformer_id" not in gdf.columns:
        gdf["transformer_id"] = [f"trf_{i+1}" for i in range(len(gdf))]

    # Name column
    name_cols = [c for c in gdf.columns if re.search(r"(name|station|transformer)", c, flags=re.I)]
    if "name" not in gdf.columns:
        if name_cols:
            gdf["name"] = gdf[name_cols[0]].astype(str)
        else:
            # create name from id if nothing else
            idcol = "transformer_id" if "transformer_id" in gdf.columns else ("substation_id" if "substation_id" in gdf.columns else None)
            if idcol:
                gdf["name"] = gdf[idcol].astype(str)
            else:
                gdf["name"] = [f"asset_{i+1}" for i in range(len(gdf))]

    return gdf


def synthesize_asset_point_in_bbox(idx, bbox):
    """
    Synthesizes a point inside bbox = (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bbox
    x = float(np.random.uniform(minx, maxx))
    y = float(np.random.uniform(miny, maxy))
    return Point(x, y)


def ensure_electric_fields(gdf: gpd.GeoDataFrame, bescom_df: pd.DataFrame = None):
    """
    Ensure required electricity fields exist on each asset and fill from real data if available,
    otherwise synthesize deterministically.

    Required fields:
    substation_id, transformer_id, phase (1 or 3), capacity_MVA, current_load_MVA,
    industrial_consumption_kwh, residential_consumption_kwh, commercial_consumption_kwh,
    load_shedding_events, ev_charging_load_estimate, solar_potential_kw
    """
    N = len(gdf)
    if N == 0:
        return gdf

    gdf = gdf.copy()

    # Ensure ID fields
    if "substation_id" not in gdf.columns:
        gdf["substation_id"] = ""

    if "transformer_id" not in gdf.columns:
        gdf["transformer_id"] = [f"trf_{i+1}" for i in range(N)]

    # Required columns
    req_cols = [
        "phase",
        "capacity_MVA",
        "current_load_MVA",
        "industrial_consumption_kwh",
        "residential_consumption_kwh",
        "commercial_consumption_kwh",
        "load_shedding_events",
        "ev_charging_load_estimate",
        "solar_potential_kw"
    ]

    for c in req_cols:
        if c not in gdf.columns:
            gdf[c] = np.nan

    # Build BESCOM mapping (if any)
    bescom_map = {}
    if bescom_df is not None:
        for idx, row in bescom_df.iterrows():
            key = None
            for candidate in ("transformer", "substation", "name", "station"):
                if candidate in bescom_df.columns:
                    key = str(row[candidate])
                    break
            if key is None:
                continue

            bescom_map[key] = {
                "industrial": pd.to_numeric(row.get("industrial", None), errors="ignore"),
                "residential": pd.to_numeric(row.get("residential", None), errors="ignore"),
                "commercial": pd.to_numeric(row.get("commercial", None), errors="ignore"),
            }

    # Geometry: if missing, will be synthesized later
    if gdf.geometry.isnull().all():
        bbox = (77.45, 12.80, 77.75, 13.10)  # Generic Bengaluru bounding box
        gdf.geometry = [
            Point(
                float(np.random.uniform(bbox[0], bbox[2])),
                float(np.random.uniform(bbox[1], bbox[3]))
            )
            for _ in range(N)
        ]
        gdf.set_crs(epsg=4326, inplace=True)

    # Fill values row-by-row
    for idx in gdf.index:

        # Phase
        if pd.isna(gdf.at[idx, "phase"]) or str(gdf.at[idx, "phase"]).strip() == "":
            gdf.at[idx, "phase"] = int(np.random.choice([1, 3], p=[0.2, 0.8]))

        # Capacity MVA
        if pd.isna(gdf.at[idx, "capacity_MVA"]):
            name = str(gdf.at[idx, "name"]).lower()

            if "sub" in name or "ss" in name:
                gdf.at[idx, "capacity_MVA"] = round(float(np.random.uniform(10, 200)), 3)
            else:
                if np.random.rand() < 0.2:
                    gdf.at[idx, "capacity_MVA"] = round(float(np.random.uniform(1, 10)), 3)
                else:
                    gdf.at[idx, "capacity_MVA"] = round(float(np.random.uniform(0.01, 1.0)), 3)

        # Current load = 20%–95% of capacity
        if pd.isna(gdf.at[idx, "current_load_MVA"]):
            cap = float(gdf.at[idx, "capacity_MVA"])
            gdf.at[idx, "current_load_MVA"] = round(cap * float(np.random.uniform(0.2, 0.95)), 3)

        # Consumption from BESCOM data if exists
        name_key = str(gdf.at[idx, "name"])
        mapped = bescom_map.get(name_key, {})

        if pd.isna(gdf.at[idx, "industrial_consumption_kwh"]):
            if "industrial" in mapped and mapped["industrial"] is not None:
                gdf.at[idx, "industrial_consumption_kwh"] = float(mapped["industrial"])
            else:
                gdf.at[idx, "industrial_consumption_kwh"] = round(
                    float(gdf.at[idx, "capacity_MVA"]) * 10000 * float(np.random.uniform(0.0, 0.6))
                )

        if pd.isna(gdf.at[idx, "residential_consumption_kwh"]):
            if "residential" in mapped and mapped["residential"] is not None:
                gdf.at[idx, "residential_consumption_kwh"] = float(mapped["residential"])
            else:
                gdf.at[idx, "residential_consumption_kwh"] = round(
                    float(gdf.at[idx, "capacity_MVA"]) * 15000 * float(np.random.uniform(0.2, 0.8))
                )

        if pd.isna(gdf.at[idx, "commercial_consumption_kwh"]):
            if "commercial" in mapped and mapped["commercial"] is not None:
                gdf.at[idx, "commercial_consumption_kwh"] = float(mapped["commercial"])
            else:
                gdf.at[idx, "commercial_consumption_kwh"] = round(
                    float(gdf.at[idx, "capacity_MVA"]) * 8000 * float(np.random.uniform(0.0, 0.6))
                )

        # Load-shedding events
        if pd.isna(gdf.at[idx, "load_shedding_events"]):
            gdf.at[idx, "load_shedding_events"] = int(np.random.poisson(0.5))

        # EV charging load estimate
        if pd.isna(gdf.at[idx, "ev_charging_load_estimate"]):
            gdf.at[idx, "ev_charging_load_estimate"] = round(float(np.random.uniform(0.0, 200.0)), 2)

        # Solar potential kW
        if pd.isna(gdf.at[idx, "solar_potential_kw"]):
            gdf.at[idx, "solar_potential_kw"] = round(float(np.random.uniform(0.0, 2000.0)), 2)

    return gdf


def attach_wards_and_h3(gdf: gpd.GeoDataFrame, wards_geojson: Path = None, h3_res: int = 8):
    """
    spatially join to wards (if provided) to set ward_id; else create synthetic ward ids by grid.
    Compute h3 index for each asset at resolution h3_res.
    """
    gdf = gdf.copy()
    # wards spatial join
    if wards_geojson and Path(wards_geojson).exists():
        wards = gpd.read_file(wards_geojson)
        # ensure both in same CRS
        try:
            wards = wards.to_crs(epsg=4326)
        except Exception:
            pass
        # if wards have ward_id column standardize name
        ward_id_col = None
        for c in wards.columns:
            if c.lower() in ("ward_id", "ward", "wardcode", "wardno", "ward_no"):
                ward_id_col = c
                break
        if ward_id_col is None:
            ward_id_col = wards.columns[0]
            wards = wards.rename(columns={ward_id_col: "ward_id"})
            ward_id_col = "ward_id"

        # spatial join (point-in-polygon)
        joined = gpd.sjoin(gdf, wards[[ward_id_col, "geometry"]], how="left", predicate="within")
        # fill ward_id column
        gdf["ward_id"] = joined[ward_id_col].fillna("")
    else:
        # create synthetic ward ids based on integer grid of lat/lon
        gdf["ward_id"] = [f"ward_{(i % 100) + 1}" for i in range(len(gdf))]

    # compute H3 index per asset
    def point_to_h3(pt, res):
        if pt is None or pt.is_empty:
            return None
        lat = pt.y
        lon = pt.x
        return h3.geo_to_h3(lat, lon, res)

    gdf["h3_index"] = gdf.geometry.apply(lambda g: point_to_h3(g.centroid if g.geom_type != "Point" else g, h3_res))
    return gdf


def aggregate_per_h3(gdf: gpd.GeoDataFrame, h3_res: int = 8):
    """
    Optional aggregation: sum consumption per H3 cell.
    """
    agg = gdf.groupby("h3_index").agg({
        "industrial_consumption_kwh": "sum",
        "residential_consumption_kwh": "sum",
        "commercial_consumption_kwh": "sum",
        "ev_charging_load_estimate": "sum",
        "solar_potential_kw": "sum",
        "capacity_MVA": "sum",
        "current_load_MVA": "sum",
        "load_shedding_events": "sum",
        "transformer_id": "count"
    }).rename(columns={"transformer_id": "num_assets"}).reset_index()
    return agg


def recommended_data_amounts(population_estimate=3000000):
    """
    Provide a guideline for how much data you may need for a large project.
    We base rough rules of thumb on city size.
    """
    # heuristics:
    # substations: 1 per ~50k-200k population (substation = high-voltage distribution)
    # transformers: 1 per ~500-2000 population (distribution transformers)
    # H3 resolution suggestion: res8 ~ 0.74 km^2, res9 ~ 0.26 km^2
    pop = population_estimate
    substations = max(10, int(pop / 100000))  # one per 100k
    transformers = max(200, int(pop / 1000))  # one per 1k people
    h3_res = 8
    res8_km2 = 0.74
    res9_km2 = 0.26
    return {
        "population_estimate": pop,
        "recommended_substations": substations,
        "recommended_transformers": transformers,
        "recommended_h3_res": h3_res,
        "h3_res8_km2_approx": res8_km2,
        "h3_res9_km2_approx": res9_km2
    }


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bescom_df = load_csv_try(Path(args.bescom_csv)) if args.bescom_csv else None
    elec_df = load_csv_try(Path(args.electricity_csv)) if args.electricity_csv else None

    # build base assets gdf
    base_gdf = build_assets_from_csvs(bescom_df, elec_df)

    # ensure required electricity fields (fill or synthesize)
    assets_gdf = ensure_electric_fields(base_gdf, bescom_df)

    # attach wards and compute h3
    assets_gdf = attach_wards_and_h3(assets_gdf, Path(args.wards_geojson) if args.wards_geojson else None, args.h3_res)

    # optional aggregation
    agg_h3 = aggregate_per_h3(assets_gdf, args.h3_res)

    # Save outputs
    out_geojson = outdir / "electricity_assets.geojson"
    out_csv = outdir / "electricity_assets.csv"
    out_h3_csv = outdir / "h3_aggregation.csv"

    try:
        assets_gdf.to_file(out_geojson, driver="GeoJSON")
    except Exception as e:
        print(f"[WARN] Could not write GeoJSON (fiona/geopandas issue): {e}. Falling back to CSV with lon/lat columns.")
        assets_gdf["lon"] = assets_gdf.geometry.x
        assets_gdf["lat"] = assets_gdf.geometry.y
        assets_gdf.drop(columns="geometry", inplace=True)
        assets_gdf.to_csv(out_csv, index=False)
    else:
        # also write CSV (with lat/lon)
        assets_gdf["lon"] = assets_gdf.geometry.x
        assets_gdf["lat"] = assets_gdf.geometry.y
        assets_gdf.to_csv(out_csv, index=False)

    agg_h3.to_csv(out_h3_csv, index=False)

    # Print recommended dataset size guidance
    rec = recommended_data_amounts(population_estimate=int(args.population_estimate))
    print("=== OUTPUT ===")
    print(f"Wrote: {out_geojson}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_h3_csv}")
    print("\n=== RECOMMENDED DATA SIZES (guideline) ===")
    print(f"For population ~{rec['population_estimate']}:")
    print(f" - Substations recommended: {rec['recommended_substations']}")
    print(f" - Transformers recommended: {rec['recommended_transformers']}")
    print(f" - H3 resolution recommended: res{rec['recommended_h3_res']} (~{rec['h3_res8_km2_approx']} km^2 per cell at res8)")
    print("\nNotes:")
    print(" - If you have wards.geojson, pass it with --wards_geojson to spatially link assets to wards.")
    print(" - H3 aggregation file gives summed consumption & counts per H3 cell.")
    print(" - All synthetic values are deterministic (seeded). If you want different randomness, change RANDOM_SEED near top.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BESCOM electricity assets and link to wards + H3.")
    parser.add_argument("--bescom_csv", required=False, help="Path to BESCOM category-wise CSV (optional).")
    parser.add_argument("--electricity_csv", required=False, help="Path to electricity assets CSV (optional).")
    parser.add_argument("--wards_geojson", required=False, help="Path to wards geojson (optional).")
    parser.add_argument("--h3_res", type=int, default=8, help="H3 resolution to compute (default 8).")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    parser.add_argument("--population_estimate", default="3000000", help="Population estimate used for recommended sizing.")
    args = parser.parse_args()
    main(args)
