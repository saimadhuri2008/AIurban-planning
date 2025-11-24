#!/usr/bin/env python3
"""
build_master.py

Merge all processed datasets into:
 - wards_master.parquet (per-ward features)
 - h3_master.parquet   (per-h3 features)
 - master_fused.parquet (joined ward+h3 rows for exploration)

Rules:
 - If ward_id missing, attempt to fill from h3_to_wards mapping (h3 -> ward)
 - If still missing, spatially join geometry to wards (requires geometry)
 - If still missing, assign nearest ward centroid
 - For missing numeric features, synthesize conservative defaults (mean/median) or simple synthetic time-aggregates
 - Outputs are saved to: data/processed/master/

Usage:
    python build_master.py --help
"""
import os
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import h3
import random
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("build_master")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------------
# Default paths (from your workspace)
# -----------------------------
DEFAULTS = {
    "wards": r"C:\AIurban-planning\data\processed\wards\wards_master_enriched.geojson",
    "h3_grid": r"C:\AIurban-planning\data\processed\h3_wards\h3_grid_res8_mapped.geojson",
    "h3_to_wards": r"C:\AIurban-planning\data\processed\h3_wards\h3_to_wards_mapping_res8.csv",
    "buildings": r"C:\AIurban-planning\data\processed\buildings_processed.ndjson",
    "population_wards": r"C:\AIurban-planning\data\processed\population_wards.csv",
    "population_h3": r"C:\AIurban-planning\data\processed\population_h3_res8.geojson",
    "income_h3": r"C:\AIurban-planning\outputs\income_housing_h3_gentrification.geojson",
    "economy_h3": r"C:\AIurban-planning\outputs\economy_h3_growth.geojson",
    "roads": r"C:\AIurban-planning\outputs\roads_enriched.geojson",
    "od": r"C:\AIurban-planning\data\processed\od_flows.csv",
    "metro_csv": r"C:\AIurban-planning\data\processed\metro_stations_enriched.csv",
    "metro_geo": r"C:\AIurban-planning\data\processed\metro_stations.geojson",
    "traffic": r"C:\AIurban-planning\data\processed\traffic_congestion.geojson",
    "water": r"C:\AIurban-planning\data\processed\water_network.geojson",
    "sewage": r"C:\AIurban-planning\data\processed\sewage_network.geojson",
    "electricity": r"C:\AIurban-planning\data\processed\electricty\electricity_assets.geojson",
    "airq_geo": r"C:\AIurban-planning\data\processed\air_quality\air_quality_readings.geojson",
    "lst": r"C:\AIurban-planning\outputs\lst_climate_trend_h3.geojson",
    "ndvi": r"C:\AIurban-planning\outputs\ndvi_urban_hotspot_h3.geojson",
    "flood": r"C:\AIurban-planning\outputs\flood_climate_trend_h3.geojson",
    "schools_h3": r"C:\AIurban-planning\outputs\schools_h3_res8.geojson",
    "health_h3": r"C:\AIurban-planning\outputs\health_h3_res8.geojson",
    "parks_wards": r"C:\AIurban-planning\data\processed\parks_wards_res8.csv",
    "emergency_h3": r"C:\AIurban-planning\outputs\emergency_services_h3.geojson",
    "h3_metrics": r"C:\AIurban-planning\outputs\h3_metrics.geojson",
    "computed_h3": r"C:\AIurban-planning\outputs\computed_features_h3.geojson",
    "outdir": r"C:\AIurban-planning\data\processed\master"
}

# -----------------------------
# Utilities
# -----------------------------
def path_exists(p):
    if p is None:
        return False
    return Path(p).exists()

def read_geo_maybe(path):
    if not path_exists(path):
        LOG.warning("Missing file: %s", path)
        return None
    try:
        gdf = gpd.read_file(path)
        LOG.info("Loaded %s: %d features", path, len(gdf))
        return gdf
    except Exception as e:
        LOG.exception("Failed to read geofile %s: %s", path, e)
        return None

def read_table_maybe(path):
    if not path_exists(path):
        LOG.warning("Missing table: %s", path)
        return None
    try:
        ext = Path(path).suffix.lower()
        if ext in [".csv", ".txt"]:
            df = pd.read_csv(path)
        elif ext in [".parquet"]:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        LOG.info("Loaded table %s: %d rows", path, len(df))
        return df
    except Exception as e:
        LOG.exception("Failed to read table %s: %s", path, e)
        return None

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Normalization helpers
# -----------------------------
def normalize_h3_column_names(df, possible_names=("h3_index", "h3_id", "h3")):
    """
    Normalize variations of H3 index column to 'h3_index'.
    Works for pandas DataFrame and GeoDataFrame.
    """
    if df is None:
        return df
    cols = list(df.columns)
    for cand in possible_names:
        if cand in cols and "h3_index" not in cols:
            try:
                df = df.rename(columns={cand: "h3_index"})
                LOG.info("Normalized column '%s' -> 'h3_index'", cand)
            except Exception:
                pass
    return df

def normalize_ward_column_names(df, possible_names=("ward_id", "ward")):
    if df is None:
        return df
    cols = list(df.columns)
    for cand in possible_names:
        if cand in cols and "ward_id" not in cols:
            try:
                df = df.rename(columns={cand: "ward_id"})
                LOG.info("Normalized column '%s' -> 'ward_id'", cand)
            except Exception:
                pass
    return df

# -----------------------------
# Mapping helpers
# -----------------------------
def fill_ward_from_h3(df, h3_to_wards_df, h3_col="h3_index", ward_col="ward_id"):
    """Given records with h3 id, fill ward_id from mapping table where ward_id is null."""
    if df is None or h3_to_wards_df is None:
        return df
    # ensure mapping has normalized columns
    h3_to_wards_df = normalize_h3_column_names(h3_to_wards_df, ("h3_index","h3_id","h3"))
    h3_to_wards_df = normalize_ward_column_names(h3_to_wards_df, ("ward_id","ward"))
    if "h3_index" not in h3_to_wards_df.columns or "ward_id" not in h3_to_wards_df.columns:
        LOG.warning("h3_to_wards mapping lacks expected columns 'h3_index' and 'ward_id' -> skipping mapping fill")
        return df
    if df is None:
        return df
    if h3_col not in df.columns:
        return df
    LOG.info("Filling ward_id from h3_to_wards mapping for rows missing ward_id")
    # prepare mapping dict
    m = dict(zip(h3_to_wards_df["h3_index"].astype(str), h3_to_wards_df["ward_id"]))
    def mapf(x):
        try:
            return m.get(str(x), None)
        except Exception:
            return None
    # ensure ward_col exists and is object dtype to accept None
    if ward_col not in df.columns:
        df[ward_col] = pd.Series([None]*len(df), index=df.index)
    # fill only for missing ward ids
    mask_missing = df[ward_col].isna()
    if mask_missing.any():
        try:
            df.loc[mask_missing, ward_col] = df.loc[mask_missing, h3_col].astype(str).map(m)
        except Exception:
            # fallback iterate
            df[ward_col] = df[ward_col].astype(object)
            for idx in df[mask_missing].index:
                val = df.at[idx, h3_col]
                df.at[idx, ward_col] = mapf(val)
    missing_after = df[ward_col].isna().sum()
    LOG.info("After mapping, missing ward_id count: %d", missing_after)
    return df

def spatial_fill_ward(df_gdf, wards_gdf, geom_col="geometry", ward_col="ward_id"):
    """Spatially join to wards for features with geometry to fill ward_id."""
    if df_gdf is None or wards_gdf is None:
        return df_gdf
    if not isinstance(df_gdf, gpd.GeoDataFrame):
        try:
            df_gdf = gpd.GeoDataFrame(df_gdf, geometry=geom_col, crs="EPSG:4326")
        except Exception:
            LOG.warning("Could not convert df to gdf for spatial join")
            return df_gdf
    LOG.info("Spatial joining %d features into wards to fill missing ward_id", len(df_gdf))
    # ensure both in same CRS
    try:
        if wards_gdf.crs and df_gdf.crs and wards_gdf.crs != df_gdf.crs:
            df_gdf = df_gdf.to_crs(wards_gdf.crs)
    except Exception:
        pass
    try:
        # use 'within' predicate; ensure ward_id column exists on wards_gdf (normalize)
        wards_gdf = normalize_ward_column_names(wards_gdf)
        if "ward_id" not in wards_gdf.columns:
            LOG.warning("Wards file lacks 'ward_id'; spatial join will not assign ward ids by attribute")
            left = gpd.sjoin(df_gdf, wards_gdf, how="left", predicate="within")
        else:
            left = gpd.sjoin(df_gdf, wards_gdf[["ward_id", "geometry"]], how="left", predicate="within")
        # left now contains index_right, ward_id (if ward_id present in wards)
        if "ward_id" in left.columns:
            # assign only where missing
            if ward_col not in df_gdf.columns:
                df_gdf[ward_col] = pd.Series([None]*len(df_gdf), index=df_gdf.index)
            mask_missing = df_gdf[ward_col].isna()
            df_gdf.loc[mask_missing, ward_col] = left.loc[mask_missing, "ward_id"].values
    except Exception as e:
        LOG.warning("Spatial fill failed: %s", e)
    return df_gdf

def assign_nearest_ward(df_gdf, wards_gdf, ward_col="ward_id"):
    """As fallback, assign nearest ward centroid for rows with no ward_id."""
    if df_gdf is None or wards_gdf is None:
        return df_gdf
    if ward_col not in df_gdf.columns:
        df_gdf[ward_col] = None
    # build ward centroids (ensure geometry present)
    try:
        wards_cent = wards_gdf.copy()
        # ensure CRS same
        if wards_cent.crs is None:
            wards_cent = wards_cent.set_crs("EPSG:4326")
        wards_cent["centroid_geom"] = wards_cent.geometry.centroid
        centroids = wards_cent[["ward_id", "centroid_geom"]].dropna()
        ward_pts = list(centroids["centroid_geom"].values)
        ward_ids = list(centroids["ward_id"].values)
    except Exception:
        LOG.warning("Failed building ward centroids for nearest assignment")
        return df_gdf
    LOG.info("Assigning nearest ward for remaining nulls (fallback)")
    # ensure df_gdf has geometry
    if not isinstance(df_gdf, gpd.GeoDataFrame):
        try:
            df_gdf = gpd.GeoDataFrame(df_gdf, geometry="geometry", crs="EPSG:4326")
        except Exception:
            LOG.warning("Data has no geometry; cannot assign nearest ward")
            return df_gdf
    # reproject to metric for accurate distance if possible
    try:
        df_tmp = df_gdf.copy().to_crs(epsg=3857)
        ward_tmp = centroids.copy().set_geometry("centroid_geom").to_crs(epsg=3857)
        ward_pts_m = list(ward_tmp["centroid_geom"].values)
        ward_ids_m = list(ward_tmp["ward_id"].values)
    except Exception:
        # fallback to geographic distances
        df_tmp = df_gdf.copy()
        ward_pts_m = ward_pts
        ward_ids_m = ward_ids
    mask_missing = df_tmp[ward_col].isna() if ward_col in df_tmp.columns else pd.Series([True]*len(df_tmp), index=df_tmp.index)
    for idx in df_tmp[mask_missing].index:
        try:
            pt = df_tmp.at[idx, "geometry"]
            if pt is None or pt.is_empty:
                continue
            dmin = None
            wid = None
            for wpt, wid_candidate in zip(ward_pts_m, ward_ids_m):
                d = pt.distance(wpt)
                if dmin is None or d < dmin:
                    dmin = d
                    wid = wid_candidate
            df_gdf.at[idx, ward_col] = wid
        except Exception:
            continue
    return df_gdf

# -----------------------------
# Synthesis helpers (when datasets absent)
# -----------------------------
def synthesize_numeric_series(index, mean=0.0, std=1.0, low=0, high=1, dtype=float, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    vals = rng.normal(loc=mean, scale=std, size=len(index))
    vals = np.clip(vals, low, high)
    if dtype is int:
        vals = np.rint(vals).astype(int)
    return pd.Series(vals, index=index)

def synthesize_population_by_h3(h3_gdf, total_pop=None, pop_per_h3_mean=None):
    """If population H3 missing, create synthetic using ward population distribution or uniform mean."""
    if h3_gdf is None:
        return None
    LOG.info("Synthesizing population per H3 cell (simple proportional or mean)")
    n = len(h3_gdf)
    if total_pop is None:
        if pop_per_h3_mean is None:
            pop_per_h3_mean = 1000
        vals = np.full(n, pop_per_h3_mean, dtype=int)
    else:
        if "hex_area_sqm" in h3_gdf.columns:
            areas = h3_gdf["hex_area_sqm"].fillna(1.0)
            w = areas / areas.sum()
            vals = (w * total_pop).round().astype(int).values
        else:
            vals = np.full(n, int(total_pop / max(n,1)))
    return pd.DataFrame({"h3_index": h3_gdf.get("h3_index", h3_gdf.index.astype(str)), "population_h3": vals})

# -----------------------------
# Main merge logic
# -----------------------------
def build_master(args):
    ensure_dir(args.outdir)
    LOG.info("Phase: build_master. Output dir: %s", args.outdir)

    # ---------- LOAD SOURCES ----------
    wards_gdf = read_geo_maybe(args.wards)
    h3_gdf = read_geo_maybe(args.h3_grid)
    h3_to_wards = read_table_maybe(args.h3_to_wards)
    buildings_gdf = read_geo_maybe(args.buildings)
    population_h3_gdf = read_geo_maybe(args.population_h3)
    population_wards = read_table_maybe(args.population_wards)
    income_h3 = read_geo_maybe(args.income_h3)
    economy_h3 = read_geo_maybe(args.economy_h3)
    roads_gdf = read_geo_maybe(args.roads)
    od_df = read_table_maybe(args.od)
    metro_df = read_table_maybe(args.metro_csv)
    if metro_df is None:
        metro_df = read_geo_maybe(args.metro_geo)
    traffic_gdf = read_geo_maybe(args.traffic)
    water_gdf = read_geo_maybe(args.water)
    sewage_gdf = read_geo_maybe(args.sewage)
    elec_gdf = read_geo_maybe(args.electricity)
    airq_gdf = read_geo_maybe(args.airq_geo)
    lst_gdf = read_geo_maybe(args.lst)
    ndvi_gdf = read_geo_maybe(args.ndvi)
    flood_gdf = read_geo_maybe(args.flood)
    schools_gdf = read_geo_maybe(args.schools_h3)
    health_gdf = read_geo_maybe(args.health_h3)
    parks_df = read_table_maybe(args.parks_wards)
    emergency_gdf = read_geo_maybe(args.emergency_h3)
    h3_metrics_gdf = read_geo_maybe(args.h3_metrics)
    computed_h3_gdf = read_geo_maybe(args.computed_h3)

    # Normalize potential column names in mapping file
    if h3_to_wards is not None:
        h3_to_wards = normalize_h3_column_names(h3_to_wards, ("h3_index","h3_id","h3"))
        h3_to_wards = normalize_ward_column_names(h3_to_wards, ("ward_id","ward"))

    # normalize h3 column names across loaded geoframes/tables
    for varname in ["h3_gdf","population_h3_gdf","income_h3","economy_h3","traffic_gdf","airq_gdf",
                    "ndvi_gdf","lst_gdf","flood_gdf","schools_gdf","health_gdf","h3_metrics_gdf","computed_h3_gdf"]:
        df = locals().get(varname)
        if df is not None:
            try:
                locals()[varname] = normalize_h3_column_names(df, ("h3_index","h3_id","h3"))
            except Exception:
                pass

    # ---------- PREPROCESS & FILL WARD IDS ----------
    # Some source tables may use alternative names for ward_id; normalize
    if population_wards is not None:
        population_wards = normalize_ward_column_names(population_wards, ("ward_id","ward"))

    feature_geo_list = [
        ("buildings", buildings_gdf),
        ("roads", roads_gdf),
        ("traffic", traffic_gdf),
        ("airq", airq_gdf),
        ("metro", metro_df if isinstance(metro_df, gpd.GeoDataFrame) else None),
        ("water", water_gdf),
        ("sewage", sewage_gdf),
        ("electricity", elec_gdf),
        ("ndvi", ndvi_gdf),
        ("lst", lst_gdf),
        ("flood", flood_gdf),
        ("schools", schools_gdf),
        ("health", health_gdf),
        ("emergency", emergency_gdf),
        ("h3_metrics", h3_metrics_gdf),
        ("computed_h3", computed_h3_gdf)
    ]

    def attempt_fill(df, name):
        if df is None:
            return None
        LOG.info("Attempting fill ward for: %s", name)
        # normalize ward column name if present
        if isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
            df = normalize_ward_column_names(df, ("ward_id","ward"))
            df = normalize_h3_column_names(df, ("h3_index","h3_id","h3"))
        # If mapping exists and df has h3_index, try mapping
        if isinstance(df, pd.DataFrame) and "h3_index" in df.columns and h3_to_wards is not None:
            try:
                df = fill_ward_from_h3(df, h3_to_wards, h3_col="h3_index", ward_col="ward_id")
            except Exception as e:
                LOG.warning("fill_ward_from_h3 failed for %s: %s", name, e)
        # For GeoDataFrames: spatial fill then nearest fallback
        if isinstance(df, gpd.GeoDataFrame):
            if wards_gdf is not None:
                try:
                    df = spatial_fill_ward(df, wards_gdf, geom_col="geometry", ward_col="ward_id")
                except Exception as e:
                    LOG.warning("Spatial fill failed for %s: %s", name, e)
            # after spatial attempt mapping fallback if still missing and mapping exists
            if "ward_id" in df.columns and df["ward_id"].isna().sum() > 0 and h3_to_wards is not None and "h3_index" in df.columns:
                try:
                    df = fill_ward_from_h3(df, h3_to_wards, h3_col="h3_index", ward_col="ward_id")
                except Exception:
                    pass
            # final fallback nearest ward centroid
            if "ward_id" not in df.columns or df["ward_id"].isna().sum() > 0:
                try:
                    df = assign_nearest_ward(df, wards_gdf, ward_col="ward_id")
                except Exception:
                    pass
        else:
            # non-geo tables: only mapping possible
            if isinstance(df, pd.DataFrame) and "h3_index" in df.columns and h3_to_wards is not None:
                df = fill_ward_from_h3(df, h3_to_wards, h3_col="h3_index", ward_col="ward_id")
        return df

    # Apply attempt_fill
    for name, gdf in feature_geo_list:
        if gdf is not None:
            filled = attempt_fill(gdf, name)
            if name == "buildings": buildings_gdf = filled
            if name == "roads": roads_gdf = filled
            if name == "traffic": traffic_gdf = filled
            if name == "water": water_gdf = filled
            if name == "sewage": sewage_gdf = filled
            if name == "electricity": elec_gdf = filled
            if name == "ndvi": ndvi_gdf = filled
            if name == "lst": lst_gdf = filled
            if name == "flood": flood_gdf = filled
            if name == "schools": schools_gdf = filled
            if name == "health": health_gdf = filled
            if name == "emergency": emergency_gdf = filled
            if name == "h3_metrics": h3_metrics_gdf = filled
            if name == "computed_h3": computed_h3_gdf = filled

    # ---------- BUILD H3 MASTER (per-h3 aggregated features) ----------
    LOG.info("Building H3 master aggregates...")
    if h3_gdf is not None:
        h3_master = h3_gdf.copy()
        h3_master = normalize_h3_column_names(h3_master, ("h3_index","h3_id","h3"))
    else:
        if h3_to_wards is not None:
            LOG.info("Creating minimal h3 master from mapping table")
            tmp = h3_to_wards.copy()
            tmp = normalize_h3_column_names(tmp, ("h3_index","h3_id","h3"))
            uniq = tmp["h3_index"].astype(str).unique()
            df_min = pd.DataFrame({"h3_index": uniq})
            lats, lons = [], []
            for hh in df_min["h3_index"].values:
                try:
                    lat, lon = h3.h3_to_geo(hh)
                except Exception:
                    lat, lon = (np.nan, np.nan)
                lats.append(lat); lons.append(lon)
            df_min["centroid_lat"] = lats
            df_min["centroid_lon"] = lons
            h3_master = gpd.GeoDataFrame(df_min, geometry=gpd.points_from_xy(df_min["centroid_lon"], df_min["centroid_lat"]), crs="EPSG:4326")
        else:
            LOG.error("No H3 grid or mapping available; cannot build full H3 master")
            h3_master = None

    # ensure consistent h3_index column
    if h3_master is not None:
        if "h3_index" not in h3_master.columns:
            if "h3_id" in h3_master.columns:
                h3_master = h3_master.rename(columns={"h3_id": "h3_index"})
            else:
                try:
                    h3_master["h3_index"] = h3_master.geometry.apply(lambda p: h3.geo_to_h3(p.y, p.x, args.h3_res) if p is not None else None)
                except Exception:
                    h3_master["h3_index"] = h3_master.index.astype(str)
        h3_master["h3_index"] = h3_master["h3_index"].astype(str)

    # aggregate dictionary per-h3
    h3_aggs = {}

    # Population per h3
    if population_h3_gdf is not None:
        population_h3_gdf = normalize_h3_column_names(population_h3_gdf, ("h3_index","h3_id","h3"))
        pop_col = next((c for c in population_h3_gdf.columns if "population" in c.lower()), None)
        if pop_col and "h3_index" in population_h3_gdf.columns:
            dfp = population_h3_gdf[["h3_index", pop_col]].rename(columns={pop_col: "population_h3"})
            h3_aggs["population_h3"] = dict(zip(dfp["h3_index"].astype(str), dfp["population_h3"]))
        else:
            LOG.warning("population_h3 missing clear population column or h3_index; will synthesize later")

    # Built-up area from buildings
    if buildings_gdf is not None:
        LOG.info("Aggregating building footprints into H3 cells")
        buildings_gdf = normalize_h3_column_names(buildings_gdf, ("h3_index","h3_id","h3"))
        if "h3_index" not in buildings_gdf.columns:
            LOG.info("Mapping building centroids to H3 indexes (res=%d)", args.h3_res)
            try:
                # safe centroid: ensure geometry valid and in EPSG:4326
                if buildings_gdf.crs and buildings_gdf.crs.to_string() != "EPSG:4326":
                    try:
                        buildings_gdf = buildings_gdf.to_crs("EPSG:4326")
                    except Exception:
                        pass
                buildings_gdf["centroid_lon"] = buildings_gdf.geometry.centroid.x
                buildings_gdf["centroid_lat"] = buildings_gdf.geometry.centroid.y
                buildings_gdf["h3_index"] = buildings_gdf.apply(lambda r: h3.geo_to_h3(float(r["centroid_lat"]), float(r["centroid_lon"]), args.h3_res) if not pd.isna(r["centroid_lat"]) else None, axis=1)
            except Exception as e:
                LOG.warning("Failed mapping building centroids to H3: %s", e)
        # area
        if "footprint_m2" in buildings_gdf.columns:
            buildings_gdf["area_m2"] = buildings_gdf["footprint_m2"].fillna(0).astype(float)
        else:
            try:
                buildings_proj = buildings_gdf.to_crs(epsg=3857)
                buildings_gdf["area_m2"] = buildings_proj.geometry.area
            except Exception:
                LOG.warning("Could not compute building area via reprojection; defaulting to 0")
                buildings_gdf["area_m2"] = 0.0
        if "h3_index" in buildings_gdf.columns:
            b_ag = buildings_gdf.groupby("h3_index")["area_m2"].sum().reset_index().rename(columns={"area_m2": "built_area_m2"})
            h3_aggs["built_area_m2"] = dict(zip(b_ag["h3_index"].astype(str), b_ag["built_area_m2"]))
        else:
            LOG.warning("Buildings aggregation skipped: no h3_index available")

    # Jobs / economy
    if economy_h3 is not None:
        economy_h3 = normalize_h3_column_names(economy_h3, ("h3_index","h3_id","h3"))
        job_col = next((c for c in economy_h3.columns if "job" in c.lower() or "it_job" in c.lower()), None)
        if job_col and "h3_index" in economy_h3.columns:
            e = economy_h3[["h3_index", job_col]].rename(columns={job_col: "it_job_density"})
            h3_aggs["it_job_density"] = dict(zip(e["h3_index"].astype(str), e["it_job_density"]))

    # Income/housing
    if income_h3 is not None:
        income_h3 = normalize_h3_column_names(income_h3, ("h3_index","h3_id","h3"))
        inc_col = next((c for c in income_h3.columns if "income" in c.lower() or "income_index" in c.lower()), None)
        if inc_col and "h3_index" in income_h3.columns:
            inc = income_h3[["h3_index", inc_col]].rename(columns={inc_col: "income_index"})
            h3_aggs["income_index"] = dict(zip(inc["h3_index"].astype(str), inc["income_index"]))

    # Traffic congestion
    if traffic_gdf is not None:
        traffic_gdf = normalize_h3_column_names(traffic_gdf, ("h3_index","h3_id","h3"))
        cong_col = next((c for c in traffic_gdf.columns if "congestion" in c.lower()), None)
        if cong_col and "h3_index" in traffic_gdf.columns:
            tdf = traffic_gdf[["h3_index", cong_col]].groupby("h3_index")[cong_col].mean().reset_index().rename(columns={cong_col: "congestion_index"})
            h3_aggs["congestion_index"] = dict(zip(tdf["h3_index"].astype(str), tdf["congestion_index"]))

    # Air quality
    if airq_gdf is not None:
        airq_gdf = normalize_h3_column_names(airq_gdf, ("h3_index","h3_id","h3"))
        aqi_col = next((c for c in airq_gdf.columns if "aqi" in c.lower() or "pm2" in c.lower()), None)
        if aqi_col and "h3_index" in airq_gdf.columns:
            a = airq_gdf[["h3_index", aqi_col]].groupby("h3_index")[aqi_col].mean().reset_index().rename(columns={aqi_col: "aqi"})
            h3_aggs["aqi"] = dict(zip(a["h3_index"].astype(str), a["aqi"]))

    # NDVI, LST, Flood
    for (gdf, key_word, outcol) in [(ndvi_gdf, "ndvi", "ndvi_mean"), (lst_gdf, "lst", "lst_day"), (flood_gdf, "flood", "flood_risk_index")]:
        if gdf is not None:
            gdf = normalize_h3_column_names(gdf, ("h3_index","h3_id","h3"))
            colc = next((cc for cc in gdf.columns if key_word in cc.lower()), None)
            if colc and "h3_index" in gdf.columns:
                dfc = gdf[["h3_index", colc]].groupby("h3_index")[colc].mean().reset_index().rename(columns={colc: outcol})
                h3_aggs[outcol] = dict(zip(dfc["h3_index"].astype(str), dfc[outcol]))

    # OD flows
    if od_df is not None and {"origin_h3", "destination_h3", "od_flow"}.issubset(set(od_df.columns)):
        LOG.info("Aggregating OD flows (in/out) per h3")
        od_df["origin_h3"] = od_df["origin_h3"].astype(str)
        od_df["destination_h3"] = od_df["destination_h3"].astype(str)
        outflow = od_df.groupby("origin_h3")["od_flow"].sum().reset_index().rename(columns={"od_flow": "total_outflow"})
        inflow = od_df.groupby("destination_h3")["od_flow"].sum().reset_index().rename(columns={"od_flow": "total_inflow"})
        h3_aggs["total_outflow"] = dict(zip(outflow["origin_h3"], outflow["total_outflow"]))
        h3_aggs["total_inflow"] = dict(zip(inflow["destination_h3"], inflow["total_inflow"]))

    # counts per h3 for schools/health
    def count_feature_per_h3(gdf, colname):
        if gdf is None:
            return {}
        gdf = normalize_h3_column_names(gdf, ("h3_index","h3_id","h3"))
        if "h3_index" in gdf.columns:
            c = gdf.groupby("h3_index").size().reset_index().rename(columns={0: colname})
            return dict(zip(c["h3_index"].astype(str), c[colname]))
        return {}

    h3_aggs["schools_count"] = count_feature_per_h3(schools_gdf, "schools_count")
    h3_aggs["health_count"] = count_feature_per_h3(health_gdf, "health_count")

    # ---------- Merge aggregated dicts into h3_master dataframe ----------
    LOG.info("Preparing final H3 master GeoDataFrame")
    if h3_master is None:
        LOG.error("No H3 master to write. Exiting.")
        return

    # attach aggregated values
    h3_df = h3_master.copy()
    if "h3_index" not in h3_df.columns:
        h3_df["h3_index"] = h3_df.index.astype(str)
    h3_df["h3_index"] = h3_df["h3_index"].astype(str)

    for key, mapdict in h3_aggs.items():
        h3_df[key] = h3_df["h3_index"].apply(lambda h: mapdict.get(str(h), np.nan))

    # built_density if hex area present
    if "built_area_m2" in h3_df.columns and "hex_area_sqm" in h3_df.columns:
        h3_df["hex_area_sqm"] = h3_df["hex_area_sqm"].replace({0: np.nan})
        h3_df["built_density"] = h3_df["built_area_m2"] / h3_df["hex_area_sqm"]
    else:
        h3_df["built_density"] = h3_df.get("built_area_m2", 0) * 0.0

    # synthesize population if missing
    if "population_h3" not in h3_df.columns or h3_df["population_h3"].isna().sum() > 0:
        LOG.info("Synthesizing missing population_h3 values")
        total_pop = None
        if population_wards is not None:
            # normalize population_wards column names for population_total
            population_wards = normalize_ward_column_names(population_wards, ("ward_id","ward"))
            pop_col = next((c for c in population_wards.columns if "population" in c.lower()), None)
            if pop_col:
                try:
                    total_pop = int(population_wards[pop_col].sum())
                except Exception:
                    total_pop = None
        pop_synth = synthesize_population_by_h3(h3_df, total_pop=total_pop, pop_per_h3_mean=2000)
        h3_df = h3_df.merge(pop_synth, on="h3_index", how="left", suffixes=("", "_synth"))
        if "population_h3" not in h3_df.columns or h3_df["population_h3"].isna().all():
            h3_df["population_h3"] = h3_df["population_h3_synth"]
        else:
            h3_df["population_h3"] = h3_df["population_h3"].fillna(h3_df["population_h3_synth"])
        h3_df = h3_df.drop(columns=[c for c in h3_df.columns if c.endswith("_synth")])

    # fill numeric NaNs with medians
    num_cols = h3_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if h3_df[c].isna().sum() > 0:
            median = float(h3_df[c].dropna().median()) if h3_df[c].dropna().size > 0 else 0.0
            h3_df[c] = h3_df[c].fillna(median)

    # assign ward_id for each H3 using mapping if available
    if "ward_id" not in h3_df.columns:
        if h3_to_wards is not None and "h3_index" in h3_to_wards.columns and "ward_id" in h3_to_wards.columns:
            mp = dict(zip(h3_to_wards["h3_index"].astype(str), h3_to_wards["ward_id"]))
            h3_df["ward_id"] = h3_df["h3_index"].apply(lambda h: mp.get(str(h), None))
        else:
            h3_df["ward_id"] = None

    # ---------- Build Wards master (ward-level aggregates) ----------
    LOG.info("Building wards master (aggregates per ward)")
    if wards_gdf is not None:
        wards = wards_gdf.copy()
        wards = normalize_ward_column_names(wards, ("ward_id","ward"))
        if "ward_id" not in wards.columns:
            wards["ward_id"] = wards.index.astype(str)
    else:
        LOG.error("No wards file found; cannot produce wards master")
        wards = None

    ward_df = None
    if wards is not None:
        # take ward attributes (non-geometry) as base
        ward_df = wards.drop(columns=[c for c in wards.columns if c == "geometry"]).copy()
        # merge population_wards if available
        if population_wards is not None and "ward_id" in population_wards.columns:
            ward_df = ward_df.merge(population_wards, on="ward_id", how="left", suffixes=("", "_pop"))
    else:
        ward_df = pd.DataFrame()

    # aggregate H3 metrics to wards
    if h3_df is not None and "ward_id" in h3_df.columns:
        LOG.info("Aggregating H3 metrics into wards (mean/sum depending on metric)")
        agg_map = {
            "population_h3": "sum",
            "built_area_m2": "sum",
            "built_density": "mean",
            "it_job_density": "mean",
            "income_index": "mean",
            "congestion_index": "mean",
            "aqi": "mean",
            "total_outflow": "sum",
            "total_inflow": "sum",
            "schools_count": "sum",
            "health_count": "sum"
        }
        for c in list(agg_map.keys()):
            if c not in h3_df.columns:
                h3_df[c] = 0
        h3_df["ward_id"] = h3_df["ward_id"].astype(object)
        ward_agg = h3_df.groupby("ward_id").agg(agg_map).reset_index()
        ward_agg = ward_agg.rename(columns={
            "population_h3": "population_est",
            "built_area_m2": "built_area_m2_sum",
            "built_density": "built_density_mean",
            "it_job_density": "it_job_density_mean",
            "income_index": "income_index_mean",
            "congestion_index": "congestion_index_mean",
            "aqi": "aqi_mean",
            "total_outflow": "total_outflow_sum",
            "total_inflow": "total_inflow_sum",
            "schools_count": "schools_count_sum",
            "health_count": "health_count_sum"
        })
        if ward_df is not None and not ward_df.empty:
            ward_df = ward_df.merge(ward_agg, on="ward_id", how="left")
        else:
            ward_df = ward_agg.copy()

    # parks per ward
    if parks_df is not None:
        parks_df = normalize_ward_column_names(parks_df, ("ward_id","ward"))
        if "ward_id" in parks_df.columns:
            parks_cnt = parks_df.groupby("ward_id").size().reset_index().rename(columns={0: "park_count"})
            ward_df = ward_df.merge(parks_cnt, on="ward_id", how="left")

    # electricity asset counts
    if elec_gdf is not None:
        elec_gdf = normalize_ward_column_names(elec_gdf, ("ward_id","ward"))
        if "ward_id" in elec_gdf.columns:
            elec_count = elec_gdf.groupby("ward_id").size().reset_index().rename(columns={0: "electricity_asset_count"})
            ward_df = ward_df.merge(elec_count, on="ward_id", how="left")

    # fill NaNs
    if isinstance(ward_df, pd.DataFrame) and not ward_df.empty:
        for c in ward_df.select_dtypes(include=[np.number]).columns:
            if ward_df[c].isna().sum() > 0:
                ward_df[c] = ward_df[c].fillna(ward_df[c].median())

    # ---------- Write outputs ----------
    LOG.info("Writing outputs to %s", args.outdir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # H3 master
    try:
        h3_out = outdir / "h3_master.parquet"
        if isinstance(h3_df, gpd.GeoDataFrame):
            h3_df.to_parquet(h3_out)
        else:
            pd.DataFrame(h3_df).to_parquet(h3_out)
        LOG.info("Wrote H3 master: %s", h3_out)
    except Exception as e:
        LOG.exception("Failed to write h3 master: %s", e)

    # wards master
    try:
        wards_out = outdir / "wards_master.parquet"
        if wards is not None:
            # merge ward attributes and numeric ward_df (avoid duplicate geometry column name collisions)
            if isinstance(ward_df, pd.DataFrame) and "ward_id" in ward_df.columns:
                # drop duplicate columns from ward_df that are already geometry fields in wards
                merge_left = ward_df.copy()
                # remove geometry if present in merge_left
                if "geometry" in merge_left.columns:
                    merge_left = merge_left.drop(columns=["geometry"])
                wards_geo_out = wards.merge(merge_left, on="ward_id", how="left")
                wards_geo_out.to_parquet(wards_out)
            else:
                wards.to_parquet(wards_out)
        else:
            pd.DataFrame(ward_df).to_parquet(wards_out)
        LOG.info("Wrote Wards master: %s", wards_out)
    except Exception as e:
        LOG.exception("Failed to write wards master: %s", e)

    # fused master
    try:
        fused_out = outdir / "master_fused.parquet"
        hf = h3_df.copy()
        wd = ward_df.copy() if isinstance(ward_df, pd.DataFrame) else pd.DataFrame()
        if "ward_id" in hf.columns and not wd.empty and "ward_id" in wd.columns:
            # drop geometry columns from wd for merge if they exist
            wd_nogeo = wd.drop(columns=[c for c in wd.columns if c == "geometry" or c == "geometry"], errors="ignore")
            fused = hf.merge(wd_nogeo, on="ward_id", how="left", suffixes=("_h3", "_ward"))
            fused.to_parquet(fused_out)
            LOG.info("Wrote fused master: %s", fused_out)
        else:
            LOG.warning("Skipping fused master: missing ward_id or empty ward table")
    except Exception as e:
        LOG.exception("Failed to write fused master: %s", e)

    # summary
    try:
        summary = {
            "h3_master_rows": int(len(h3_df)) if h3_df is not None else 0,
            "wards_master_rows": int(len(ward_df)) if ward_df is not None else 0,
            "files_used": {
                k: v for k, v in vars(args).items() if k in DEFAULTS.keys()
            }
        }
        summary_path = outdir / "master_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        LOG.info("Wrote master summary: %s", summary_path)
    except Exception as e:
        LOG.exception("Failed writing summary: %s", e)

    LOG.info("Master build complete. Outputs in: %s", outdir)
    return

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build master H3+Ward datasets for Phase0")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", default=v, help=f"path for {k} (default from workspace)")
    p.add_argument("--h3-res", type=int, default=8, help="H3 resolution to use for mapping")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LOG.info("Running build_master with args: %s", args)
    build_master(args)
