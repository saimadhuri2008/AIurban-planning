#!/usr/bin/env python3
"""
derived.py (fixed)

Compute derived urban metrics (accessibility, stress, urban form, inequality)
and write H3 + ward GeoJSON/CSV outputs.

Usage:
    python derived.py --h3-res 8 --expand-k 2 --out ./outputs

Author: Assistant (urban-planning project)
"""

import argparse
import json
import math
from pathlib import Path
from datetime import datetime
import logging
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# optional dependency
try:
    import h3
except Exception:
    h3 = None

# --------------------
# Configuration / Paths
# --------------------
DEFAULT_WARDS = Path("C:/AIurban-planning/data/processed/wards/wards_master_enriched.geojson")
DEFAULT_TEMPORAL = Path("C:\AIurban-planning\src\data_pipeline\derived.py")
DEFAULT_PARKS_KML = Path("C:\\Users\\jbhuv\\Downloads\\bbmp-parks.kml")
DEFAULT_SCHOOLS_CSV = Path("C:\\Users\\jbhuv\\number-of-schools-by-ward.csv")
DEFAULT_HEALTH_KML = Path("C:\\Users\\jbhuv\\health_centres.kml")
DEFAULT_BUS_STOPS_CSV = Path("C:\\Users\\jbhuv\\bus_stops.csv")
DEFAULT_METRO_STATIONS_CSV = Path("C:\\Users\\jbhuv\\metro_stations.csv")

RNG_SEED = 20251119
rng = np.random.RandomState(RNG_SEED)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("derived")

# Silence overly noisy geopandas warnings about geographic CRS length/centroid warnings
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS", category=UserWarning)


# --------------------
# Helpers
# --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_write_geojson(gdf: gpd.GeoDataFrame, path: Path):
    """
    Ensure GeoDataFrame has a single valid geometry column and write GeoJSON.
    This function avoids accidentally dropping the real 'geometry' column.
    """
    # If geometry column missing but there is a centroid_* pair, try to reconstruct Point geometry
    if "geometry" not in gdf.columns:
        # try to build geometry from centroid lon/lat fields
        if "centroid_lon" in gdf.columns and "centroid_lat" in gdf.columns:
            gdf["geometry"] = [Point(xy) for xy in zip(gdf["centroid_lon"], gdf["centroid_lat"])]
        else:
            # try to find any column with shapely geometries
            geom_col = None
            for c in gdf.columns:
                if gdf[c].dtype == "object":
                    sample = gdf[c].dropna().head(1)
                    if len(sample) == 1 and hasattr(sample.iloc[0], "geom_type"):
                        geom_col = c
                        break
            if geom_col:
                gdf = gdf.rename(columns={geom_col: "geometry"})
            else:
                raise ValueError("No geometry column to write to GeoJSON")

    # Ensure only a single geometry column remains named 'geometry'
    # Drop other columns whose dtype is shapely geometry (but don't drop 'geometry').
    other_geom_cols = []
    for c in list(gdf.columns):
        if c == "geometry":
            continue
        # detect if column contains shapely geometries
        if gdf[c].dropna().apply(lambda x: hasattr(x, "geom_type")).any():
            other_geom_cols.append(c)
    if other_geom_cols:
        gdf = gdf.drop(columns=other_geom_cols)

    # Ensure geometry is set and valid
    gdf = gdf.set_geometry("geometry")
    # convert any non-serializable objects (dict/list) columns to JSON string
    for c in gdf.columns:
        if gdf[c].dtype == object:
            if gdf[c].apply(lambda x: isinstance(x, (dict, list))).any():
                gdf[c] = gdf[c].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    # final write
    gdf.to_file(str(path), driver="GeoJSON")
    log.info("Wrote GeoJSON: %s", path)


def parse_kml_points(kml_path: Path):
    """
    Lightweight KML point parser (returns DataFrame with name, lat, lon).
    If file doesn't exist returns empty DataFrame.
    """
    if not kml_path.exists():
        return pd.DataFrame(columns=["name", "lat", "lon"])
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    import re
    placemarks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL | re.IGNORECASE)
    rows = []
    for pm in placemarks:
        name_m = re.search(r"<name>(.*?)</name>", pm, flags=re.DOTALL | re.IGNORECASE)
        coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.DOTALL | re.IGNORECASE)
        if not coords_m:
            continue
        try:
            first = coords_m.group(1).strip().split()[0]
            lon, lat = [float(x) for x in first.split(",")[:2]]
        except Exception:
            continue
        name = name_m.group(1).strip() if name_m else ""
        rows.append({"name": name, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def get_h3_api():
    if h3 is None:
        raise RuntimeError("h3 library not available. Install 'h3' python package.")
    return h3


def h3_boundary_poly(h):
    b = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(pt[1], pt[0]) for pt in b])


# --------------------
# H3 cellset builder
# --------------------
def build_h3_cellset_from_wards(wards_gdf: gpd.GeoDataFrame, res: int, expand_k: int = 2):
    api = get_h3_api()
    if wards_gdf is None or wards_gdf.empty:
        center = (12.9716, 77.5946)
        ch = api.geo_to_h3(center[0], center[1], res)
        cells = list(api.k_ring(ch, expand_k))
        return [str(c) for c in cells]
    # collect candidate H3s from polygon vertices
    hset = set()
    for geom in wards_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if hasattr(geom, "geoms") else [geom]
        for poly in polys:
            try:
                for lon, lat in poly.exterior.coords:
                    try:
                        hset.add(api.geo_to_h3(lat, lon, res))
                    except Exception:
                        continue
            except Exception:
                continue
    # expand
    initial = list(hset)
    for h in initial:
        try:
            hset.update(api.k_ring(h, expand_k))
        except Exception:
            pass
    # filter by whether centroid within any ward using union_all()
    try:
        union = wards_gdf.geometry.unary_union if not hasattr(wards_gdf.geometry, "union_all") else wards_gdf.geometry.union_all()
        # prefer union_all if available; fallback satisfies deprecation
    except Exception:
        union = None

    cells = []
    for c in hset:
        try:
            lat, lon = api.h3_to_geo(c)
            pt = Point(lon, lat)
            if union is None or union.contains(pt):
                cells.append(str(c))
        except Exception:
            continue
    return cells


# --------------------
# Load inputs
# --------------------
def load_wards(wards_path: Path):
    if wards_path.exists():
        wg = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
        # heuristically find ward id
        ward_col = next((c for c in wg.columns if c.lower().startswith("ward")), wg.columns[0])
        if ward_col != "ward_id":
            wg = wg.rename(columns={ward_col: "ward_id"})
        wg["ward_id"] = wg["ward_id"].astype(str)
        return wg
    else:
        log.warning("Wards not found at %s. Generating synthetic wards (10).", wards_path)
        rows = []
        lats = np.linspace(12.85, 13.12, 10)
        lons = np.linspace(77.45, 77.75, 10)
        for i in range(10):
            lat = lats[i % len(lats)]; lon = lons[i % len(lons)]
            poly = Polygon([(lon-0.02, lat-0.01), (lon+0.02, lat-0.01), (lon+0.02, lat+0.01), (lon-0.02, lat+0.01)])
            rows.append({"ward_id": f"ward_{i+1}", "geometry": poly})
        wg = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        return wg


def load_amenities(parks_kml: Path, schools_csv: Path, health_kml: Path, bus_csv: Path, metro_csv: Path):
    parks_df = parse_kml_points(parks_kml) if parks_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    health_df = parse_kml_points(health_kml) if health_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    schools_df = pd.DataFrame(columns=["name","lat","lon"])
    if schools_csv.exists():
        try:
            sd = pd.read_csv(schools_csv, dtype=str)
            latcol = next((c for c in sd.columns if "lat" in c.lower()), None)
            loncol = next((c for c in sd.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            namecol = next((c for c in sd.columns if "name" in c.lower() or "school" in c.lower()), sd.columns[0])
            rows = []
            if latcol and loncol:
                for i, r in sd.iterrows():
                    try:
                        rows.append({"name": r.get(namecol, f"school_{i}"), "lat": float(r[latcol]), "lon": float(r[loncol])})
                    except Exception:
                        continue
                schools_df = pd.DataFrame(rows)
        except Exception:
            log.warning("Failed to read schools CSV; generating synthetic schools")

    bus_df = pd.DataFrame(columns=["name","lat","lon"])
    metro_df = pd.DataFrame(columns=["name","lat","lon"])
    if bus_csv.exists():
        try:
            bd = pd.read_csv(bus_csv, dtype=str)
            latcol = next((c for c in bd.columns if "lat" in c.lower()), None)
            loncol = next((c for c in bd.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            if latcol and loncol:
                bus_df = pd.DataFrame([{"name": f"bus_{i+1}", "lat": float(r[latcol]), "lon": float(r[loncol])} for i, r in bd.iterrows() if pd.notna(r.get(latcol)) and pd.notna(r.get(loncol))])
        except Exception:
            pass
    if metro_csv.exists():
        try:
            md = pd.read_csv(metro_csv, dtype=str)
            latcol = next((c for c in md.columns if "lat" in c.lower()), None)
            loncol = next((c for c in md.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            if latcol and loncol:
                metro_df = pd.DataFrame([{"name": f"metro_{i+1}", "lat": float(r[latcol]), "lon": float(r[loncol])} for i, r in md.iterrows() if pd.notna(r.get(latcol)) and pd.notna(r.get(loncol))])
        except Exception:
            pass

    # Synthesize large volumes if missing
    if parks_df.empty:
        N = 200
        parks_df = pd.DataFrame({"name":[f"synt_park_{i+1}" for i in range(N)], "lat": rng.uniform(12.85, 13.12, size=N), "lon": rng.uniform(77.45, 77.75, size=N)})
    if schools_df.empty:
        N = 5000
        schools_df = pd.DataFrame({"name":[f"synt_school_{i+1}" for i in range(N)], "lat": rng.uniform(12.85, 13.12, size=N), "lon": rng.uniform(77.45, 77.75, size=N)})
    if health_df.empty:
        N = 300
        health_df = pd.DataFrame({"name":[f"synt_health_{i+1}" for i in range(N)], "lat": rng.uniform(12.85, 13.12, size=N), "lon": rng.uniform(77.45, 77.75, size=N)})
    if bus_df.empty:
        N = 1000
        bus_df = pd.DataFrame({"name":[f"synt_bus_{i+1}" for i in range(N)], "lat": rng.uniform(12.85, 13.12, size=N), "lon": rng.uniform(77.45, 77.75, size=N)})
    if metro_df.empty:
        N = 150
        metro_df = pd.DataFrame({"name":[f"synt_metro_{i+1}" for i in range(N)], "lat": rng.uniform(12.85, 13.12, size=N), "lon": rng.uniform(77.45, 77.75, size=N)})

    return parks_df, schools_df, health_df, bus_df, metro_df


def load_temporal_stats(temporal_path: Path, h3_cells):
    n = len(h3_cells)
    stats = {}
    if temporal_path.exists():
        try:
            # attempt to read a small sample and compute group means if columns exist
            # read metadata columns only to check availability
            import pyarrow.parquet as _pq
            pf = _pq.ParquetFile(str(temporal_path))
            cols = pf.metadata.schema.to_arrow_schema().names
            want = ["h3_id", "aqi", "electricity_kwh", "water_liters", "traffic_speed_kmph", "mobility_count"]
            available = [c for c in want if c in cols]
            if "h3_id" in available:
                df = pd.read_parquet(str(temporal_path), columns=available)
                agg = df.groupby("h3_id").mean()
                def map_arr(col):
                    return np.array([agg.at[h, col] if (h in agg.index and col in agg.columns) else np.nan for h in h3_cells], dtype=float)
                mean_aqi = map_arr("aqi") if "aqi" in available else np.full(n, np.nan)
                mean_elec = map_arr("electricity_kwh") if "electricity_kwh" in available else np.full(n, np.nan)
                mean_water = map_arr("water_liters") if "water_liters" in available else np.full(n, np.nan)
                mean_traffic = map_arr("traffic_speed_kmph") if "traffic_speed_kmph" in available else np.full(n, np.nan)
                mean_mob = map_arr("mobility_count") if "mobility_count" in available else np.full(n, np.nan)
                # fill NaNs with medians or synthetic fallback
                for arr in (mean_aqi, mean_elec, mean_water, mean_traffic, mean_mob):
                    if np.isnan(arr).all():
                        raise ValueError("All NaN in temporal stats -> fallback")
                return {"aqi": np.where(np.isnan(mean_aqi), np.nanmedian(mean_aqi[np.isfinite(mean_aqi)]), mean_aqi),
                        "elec": np.where(np.isnan(mean_elec), np.nanmedian(mean_elec[np.isfinite(mean_elec)]), mean_elec),
                        "water": np.where(np.isnan(mean_water), np.nanmedian(mean_water[np.isfinite(mean_water)]), mean_water),
                        "traffic": np.where(np.isnan(mean_traffic), np.nanmedian(mean_traffic[np.isfinite(mean_traffic)]), mean_traffic),
                        "mobility": np.where(np.isnan(mean_mob), np.nanmedian(mean_mob[np.isfinite(mean_mob)]), mean_mob)}
        except Exception as e:
            log.warning("Failed to extract stats from temporal parquet (%s): %s. Falling back to synthetic.", temporal_path, e)

    # fallback synth
    log.info("Generating synthetic temporal stats (fallback).")
    return {"aqi": rng.uniform(40, 200, size=n),
            "elec": rng.uniform(50, 300, size=n),
            "water": rng.uniform(200, 1200, size=n),
            "traffic": rng.uniform(10, 45, size=n),
            "mobility": rng.uniform(100, 5000, size=n)}


# --------------------
# Derived computations
# --------------------
def normalize_series(arr):
    arr = np.array(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mn = np.nanmin(arr); mx = np.nanmax(arr)
    if np.isnan(mn) or np.isnan(mx) or np.isclose(mx, mn):
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def compute_infrastructure_stress(elec_arr, water_arr, sewage_arr=None):
    e = normalize_series(elec_arr)
    w = normalize_series(water_arr)
    s = normalize_series(sewage_arr) if sewage_arr is not None else np.zeros_like(e)
    score = 0.45 * e + 0.35 * w + 0.2 * s
    return np.clip(score, 0.0, 1.0)


def compute_urban_form_metrics(built_density_arr, green_cover_arr):
    bd = normalize_series(built_density_arr)
    gr = normalize_series(green_cover_arr)
    compactness = bd * (1 - gr)
    sprawl = 1 - compactness
    return compactness, sprawl


def compute_accessibility_times(h3_cells, amenities_df, speed_kmph=30.0):
    res = {}
    if amenities_df is None or len(amenities_df) == 0:
        for h in h3_cells:
            res[h] = 999.0
        return res
    pts = [(float(r["lat"]), float(r["lon"])) for _, r in amenities_df.iterrows()]
    for h in h3_cells:
        lat, lon = h3.h3_to_geo(h)
        dmin_km = min(haversine_km(lon, lat, p_lon, p_lat) for p_lat, p_lon in pts)
        minutes = (dmin_km / max(0.1, speed_kmph)) * 60.0
        res[h] = round(float(minutes), 2)
    return res


def compute_inequality_metrics(h3_cells, access_times_dicts):
    keys = list(access_times_dicts.keys())
    n_keys = len(keys)
    n_cells = len(h3_cells)
    mat = np.full((n_keys, n_cells), np.nan)
    for i, k in enumerate(keys):
        d = access_times_dicts[k]
        for j, h in enumerate(h3_cells):
            v = d.get(h, np.nan)
            mat[i, j] = float(v) if v is not None else np.nan
    with np.errstate(invalid="ignore"):
        stds = np.nanstd(mat, axis=0)
    gap = normalize_series(np.nan_to_num(stds, nan=0.0))
    return gap


# --------------------
# Main routine
# --------------------
def main(args):
    outdir = Path(args.out)
    ensure_dir(outdir)

    # load wards
    wards = load_wards(DEFAULT_WARDS)
    log.info("Loaded wards: %d features", len(wards))

    # build H3 cell set
    h3_cells = build_h3_cellset_from_wards(wards, res=args.h3_res, expand_k=args.expand_k)
    n_cells = len(h3_cells)
    log.info("H3 cells generated: %d", n_cells)

    # prepare H3 GeoDataFrame
    api = get_h3_api()
    rows = []
    for h in h3_cells:
        lat, lon = api.h3_to_geo(h)
        poly = h3_boundary_poly(h)
        rows.append({"h3_id": h, "geometry": poly, "centroid_lat": lat, "centroid_lon": lon})
    h3_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    # attach ward via centroid spatial join (robust)
    cent = h3_gdf.set_geometry("geometry").centroid
    cent = gpd.GeoDataFrame({"h3_id": h3_gdf["h3_id"]}, geometry=cent, crs="EPSG:4326")
    try:
        # use within predicate; ensure wards in same CRS
        wards_proj = wards.to_crs(epsg=4326)
        joined = gpd.sjoin(cent, wards_proj[["ward_id", "geometry"]], how="left", predicate="within")
        h3_gdf = h3_gdf.merge(joined[["h3_id", "ward_id"]], on="h3_id", how="left")
        h3_gdf["ward_id"] = h3_gdf["ward_id"].fillna("ward_unknown")
    except Exception as e:
        log.warning("Ward spatial join failed: %s. Filling ward_unknown.", e)
        h3_gdf["ward_id"] = "ward_unknown"

    # load amenities & temporal stats
    parks_df, schools_df, health_df, bus_df, metro_df = load_amenities(DEFAULT_PARKS_KML, DEFAULT_SCHOOLS_CSV, DEFAULT_HEALTH_KML, DEFAULT_BUS_STOPS_CSV, DEFAULT_METRO_STATIONS_CSV)
    temporal_stats = load_temporal_stats(DEFAULT_TEMPORAL, h3_cells)

    # basic proxies
    built_density = np.linspace(50, 300, n_cells)
    green_cover_counts = np.zeros(n_cells, dtype=float)
    for i, h in enumerate(h3_cells):
        lat, lon = api.h3_to_geo(h)
        if not parks_df.empty:
            green_cover_counts[i] = parks_df.apply(lambda r: 1 if haversine_km(lon, lat, r["lon"], r["lat"]) < 0.7 else 0, axis=1).sum()
    green_cover_ratio = green_cover_counts / (green_cover_counts.max() if green_cover_counts.max() > 0 else 1)
    green_cover_ratio = np.clip(green_cover_ratio, 0.0, 1.0)

    # accessibility
    log.info("Computing accessibility to bus/metro/hospital/school/park...")
    access_bus = compute_accessibility_times(h3_cells, bus_df, speed_kmph=25.0)
    access_metro = compute_accessibility_times(h3_cells, metro_df, speed_kmph=35.0)
    access_hospital = compute_accessibility_times(h3_cells, health_df, speed_kmph=30.0)
    access_school = compute_accessibility_times(h3_cells, schools_df, speed_kmph=20.0)
    access_park = compute_accessibility_times(h3_cells, parks_df, speed_kmph=5.0)

    # infrastructure stress from temporal stats
    mean_aqi = temporal_stats["aqi"]
    mean_elec = temporal_stats["elec"]
    mean_water = temporal_stats["water"]
    mean_traffic = temporal_stats["traffic"]
    mean_mob = temporal_stats["mobility"]
    mean_sewage = np.linspace(10, 100, n_cells)

    electricity_stress = compute_infrastructure_stress(mean_elec, mean_water, mean_sewage)
    water_stress = compute_infrastructure_stress(mean_elec, mean_water, mean_sewage)
    sewage_stress = compute_infrastructure_stress(mean_elec, mean_water, mean_sewage)
    pollution_stress = normalize_series(mean_aqi)
    mobility_stress = normalize_series(1.0 / np.clip(mean_traffic, 0.1, None))

    # urban form
    compactness, sprawl = compute_urban_form_metrics(built_density, green_cover_ratio)

    # inequality metrics
    log.info("Computing inequality metrics...")
    amenity_gap = compute_inequality_metrics(h3_cells, {"hospital": access_hospital, "school": access_school, "park": access_park})
    transport_gap = compute_inequality_metrics(h3_cells, {"bus": access_bus, "metro": access_metro})
    resource_gap = compute_inequality_metrics(h3_cells, {"electricity": dict(zip(h3_cells, electricity_stress)), "water": dict(zip(h3_cells, water_stress)), "sewage": dict(zip(h3_cells, sewage_stress))})

    # populate h3_gdf
    h3_gdf["mean_aqi"] = mean_aqi
    h3_gdf["mean_electricity_kwh"] = mean_elec
    h3_gdf["mean_water_liters"] = mean_water
    h3_gdf["mean_traffic_speed_kmph"] = mean_traffic
    h3_gdf["mean_mobility_count"] = mean_mob

    h3_gdf["green_cover_ratio"] = green_cover_ratio
    h3_gdf["built_density_score"] = built_density
    h3_gdf["compactness_score"] = compactness
    h3_gdf["sprawl_index"] = sprawl

    h3_gdf["time_to_nearest_bus_min"] = [access_bus.get(h, 999.0) for h in h3_cells]
    h3_gdf["time_to_nearest_metro_min"] = [access_metro.get(h, 999.0) for h in h3_cells]
    h3_gdf["time_to_nearest_hospital_min"] = [access_hospital.get(h, 999.0) for h in h3_cells]
    h3_gdf["time_to_nearest_school_min"] = [access_school.get(h, 999.0) for h in h3_cells]
    h3_gdf["time_to_nearest_park_min"] = [access_park.get(h, 999.0) for h in h3_cells]

    h3_gdf["electricity_stress_index"] = electricity_stress
    h3_gdf["water_stress_index"] = water_stress
    h3_gdf["sewage_stress_index"] = sewage_stress
    h3_gdf["pollution_stress_index"] = pollution_stress
    h3_gdf["mobility_stress_index"] = mobility_stress

    h3_gdf["amenity_gap_score"] = amenity_gap
    h3_gdf["transport_gap_score"] = transport_gap
    h3_gdf["resource_gap_score"] = resource_gap

    # aggregate to wards
    log.info("Aggregating H3 metrics to ward level...")
    cent = h3_gdf.set_geometry("geometry").centroid
    cent = gpd.GeoDataFrame({"h3_id": h3_gdf["h3_id"]}, geometry=cent, crs="EPSG:4326")
    try:
        wards_proj = wards.to_crs(epsg=4326)
        sj = gpd.sjoin(cent, wards_proj[["ward_id", "geometry"]], how="left", predicate="within")
        mapping = sj.set_index("h3_id")["ward_id"].to_dict()
        h3_gdf["ward_id"] = [mapping.get(h, "ward_unknown") for h in h3_cells]
    except Exception as e:
        log.warning("Ward mapping via sjoin failed at aggregation: %s", e)

    agg_cols = ["mean_aqi", "mean_electricity_kwh", "mean_water_liters", "mean_traffic_speed_kmph",
                "green_cover_ratio", "built_density_score", "compactness_score", "sprawl_index",
                "electricity_stress_index", "water_stress_index", "sewage_stress_index",
                "pollution_stress_index", "mobility_stress_index", "amenity_gap_score",
                "transport_gap_score", "resource_gap_score"]

    wards_agg = h3_gdf.dropna(subset=["ward_id"]).groupby("ward_id")[agg_cols].mean().reset_index()
    wards_out = wards.merge(wards_agg, on="ward_id", how="left")
    for c in agg_cols:
        if c not in wards_out.columns:
            wards_out[c] = 0.0
    wards_out[agg_cols] = wards_out[agg_cols].fillna(0.0)

    # Write outputs
    out_h3_geo = Path(outdir) / "derived_h3.geojson"
    out_h3_csv = Path(outdir) / "derived_h3.csv"
    out_wards_geo = Path(outdir) / "derived_wards.geojson"
    out_wards_csv = Path(outdir) / "derived_wards.csv"

    # ensure geometry exists
    if "geometry" not in h3_gdf.columns:
        # try to reconstruct from h3 ids
        try:
            h3_gdf["geometry"] = [h3_boundary_poly(h) for h in h3_gdf["h3_id"]]
        except Exception:
            raise RuntimeError("No geometry present for H3 records and cannot reconstruct.")

    safe_write_geojson(h3_gdf, out_h3_geo)
    h3_gdf.drop(columns=["geometry"]).to_csv(out_h3_csv, index=False)
    safe_write_geojson(wards_out, out_wards_geo)
    wards_out.drop(columns=["geometry"]).to_csv(out_wards_csv, index=False)

    log.info("Wrote outputs:\n - %s\n - %s\n - %s\n - %s", out_h3_geo, out_h3_csv, out_wards_geo, out_wards_csv)
    return out_h3_geo, out_wards_geo


# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h3-res", type=int, default=8)
    p.add_argument("--expand-k", type=int, default=2)
    p.add_argument("--out", default="./outputs")
    args = p.parse_args()
    main(args)
