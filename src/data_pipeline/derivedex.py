#!/usr/bin/env python3
"""
derived.py (corrected)

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
import logging
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# optional dependency
try:
    import h3
except ImportError:
    h3 = None

# --------------------
# Paths & Config
# --------------------
DEFAULT_WARDS = Path(r"C:/AIurban-planning/data/processed/wards/wards_master_enriched.geojson")
DEFAULT_TEMPORAL = Path(r"C:/AIurban-planning/src/data_pipeline/derived.parquet")
DEFAULT_PARKS_KML = Path(r"C:/Users/jbhuv/Downloads/bbmp-parks.kml")
DEFAULT_SCHOOLS_CSV = Path(r"C:/Users/jbhuv/number-of-schools-by-ward.csv")
DEFAULT_HEALTH_KML = Path(r"C:/Users/jbhuv/health_centres.kml")
DEFAULT_BUS_STOPS_CSV = Path(r"C:/Users/jbhuv/bus_stops.csv")
DEFAULT_METRO_STATIONS_CSV = Path(r"C:/Users/jbhuv/metro_stations.csv")

RNG_SEED = 20251119
rng = np.random.RandomState(RNG_SEED)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("derived")

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS", category=UserWarning)

# --------------------
# Helper Functions
# --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_write_geojson(gdf: gpd.GeoDataFrame, path: Path):
    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame has no geometry column")
    gdf = gdf.set_geometry("geometry")
    for c in gdf.columns:
        if gdf[c].dtype == object and gdf[c].apply(lambda x: isinstance(x, (dict, list))).any():
            gdf[c] = gdf[c].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    gdf.to_file(str(path), driver="GeoJSON")
    log.info("Wrote GeoJSON: %s", path)

def parse_kml_points(kml_path: Path):
    if not kml_path.exists():
        return pd.DataFrame(columns=["name","lat","lon"])
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
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_h3_api():
    if h3 is None:
        raise RuntimeError("h3 library not available. Install 'h3' package.")
    return h3

def h3_boundary_poly(h):
    b = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(pt[1], pt[0]) for pt in b])

def normalize_series(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if np.isnan(mn) or np.isnan(mx) or np.isclose(mx, mn):
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

# --------------------
# Load Data
# --------------------
def load_wards(path: Path):
    if path.exists():
        wg = gpd.read_file(str(path)).to_crs(epsg=4326)
        ward_col = next((c for c in wg.columns if "ward" in c.lower()), wg.columns[0])
        wg = wg.rename(columns={ward_col:"ward_id"}) if ward_col != "ward_id" else wg
        wg["ward_id"] = wg["ward_id"].astype(str)
        return wg
    log.warning("Wards not found. Generating synthetic 10 wards.")
    rows = []
    lats = np.linspace(12.85, 13.12, 10)
    lons = np.linspace(77.45, 77.75, 10)
    for i in range(10):
        lat, lon = lats[i % len(lats)], lons[i % len(lons)]
        poly = Polygon([(lon-0.02, lat-0.01),(lon+0.02, lat-0.01),(lon+0.02, lat+0.01),(lon-0.02, lat+0.01),(lon-0.02, lat-0.01)])
        rows.append({"ward_id":f"ward_{i+1}", "geometry": poly})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

def load_amenities(parks_kml, schools_csv, health_kml, bus_csv, metro_csv):
    parks = parse_kml_points(parks_kml) if parks_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    health = parse_kml_points(health_kml) if health_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    schools = pd.DataFrame(columns=["name","lat","lon"])
    if schools_csv.exists():
        try:
            sd = pd.read_csv(schools_csv, dtype=str)
            latcol = next((c for c in sd.columns if "lat" in c.lower()), None)
            loncol = next((c for c in sd.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            namecol = next((c for c in sd.columns if "name" in c.lower() or "school" in c.lower()), sd.columns[0])
            if latcol and loncol:
                schools = pd.DataFrame([{"name": r.get(namecol,f"school_{i+1}"), "lat":float(r[latcol]), "lon":float(r[loncol])} for i,r in sd.iterrows() if pd.notna(r.get(latcol)) and pd.notna(r.get(loncol))])
        except:
            log.warning("Failed reading schools CSV.")
    bus = pd.DataFrame(columns=["name","lat","lon"])
    metro = pd.DataFrame(columns=["name","lat","lon"])
    for csv_path, df_name, prefix in [(bus_csv, bus, "bus"), (metro_csv, metro, "metro")]:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, dtype=str)
                latcol = next((c for c in df.columns if "lat" in c.lower()), None)
                loncol = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
                if latcol and loncol:
                    val = pd.DataFrame([{"name":f"{prefix}_{i+1}","lat":float(r[latcol]),"lon":float(r[loncol])} for i,r in df.iterrows() if pd.notna(r.get(latcol)) and pd.notna(r.get(loncol))])
                    if prefix=="bus": bus=val
                    else: metro=val
            except:
                pass
    # Synthesize if missing
    if parks.empty: parks = pd.DataFrame({"name":[f"synt_park_{i+1}" for i in range(200)], "lat":rng.uniform(12.85,13.12,200), "lon":rng.uniform(77.45,77.75,200)})
    if schools.empty: schools = pd.DataFrame({"name":[f"synt_school_{i+1}" for i in range(5000)], "lat":rng.uniform(12.85,13.12,5000), "lon":rng.uniform(77.45,77.75,5000)})
    if health.empty: health = pd.DataFrame({"name":[f"synt_health_{i+1}" for i in range(300)], "lat":rng.uniform(12.85,13.12,300), "lon":rng.uniform(77.45,77.75,300)})
    if bus.empty: bus = pd.DataFrame({"name":[f"synt_bus_{i+1}" for i in range(1000)], "lat":rng.uniform(12.85,13.12,1000), "lon":rng.uniform(77.45,77.75,1000)})
    if metro.empty: metro = pd.DataFrame({"name":[f"synt_metro_{i+1}" for i in range(150)], "lat":rng.uniform(12.85,13.12,150), "lon":rng.uniform(77.45,77.75,150)})
    return parks, schools, health, bus, metro

def load_temporal_stats(temporal_path: Path, h3_cells):
    n = len(h3_cells)
    if temporal_path.exists():
        try:
            import pyarrow.parquet as pq
            df = pd.read_parquet(str(temporal_path))
            agg = df.groupby("h3_id").mean()
            res = {}
            for col in ["aqi","electricity_kwh","water_liters","traffic_speed_kmph","mobility_count"]:
                res[col] = np.array([agg.at[h,col] if h in agg.index and col in agg.columns else np.nan for h in h3_cells],dtype=float)
            for k in res: res[k] = np.where(np.isnan(res[k]), np.nanmedian(res[k][np.isfinite(res[k])]), res[k])
            return {"aqi":res["aqi"],"elec":res["electricity_kwh"],"water":res["water_liters"],"traffic":res["traffic_speed_kmph"],"mobility":res["mobility_count"]}
        except Exception as e:
            log.warning("Temporal stats read failed: %s. Using synthetic.", e)
    return {"aqi":rng.uniform(40,200,n),"elec":rng.uniform(50,300,n),"water":rng.uniform(200,1200,n),"traffic":rng.uniform(10,45,n),"mobility":rng.uniform(100,5000,n)}

# --------------------
# H3 & Metrics
# --------------------
def build_h3_cellset_from_wards(wards_gdf, res=8, expand_k=2):
    api = get_h3_api()
    hset = set()
    for geom in wards_gdf.geometry:
        if geom is None or geom.is_empty: continue
        polys = geom.geoms if hasattr(geom,"geoms") else [geom]
        for poly in polys:
            for lon, lat in poly.exterior.coords:
                try: hset.add(api.geo_to_h3(lat, lon, res))
                except: pass
    # expand
    for h in list(hset):
        try: hset.update(api.k_ring(h, expand_k))
        except: pass
    # filter by centroid inside any ward
    union = wards_gdf.geometry.unary_union
    cells = []
    for h in hset:
        lat, lon = api.h3_to_geo(h)
        if union.contains(Point(lon, lat)):
            cells.append(str(h))
    return cells

def compute_accessibility_times(h3_cells, amenities_df, speed_kmph=30.0):
    res = {}
    if amenities_df.empty:
        return {h:999.0 for h in h3_cells}
    pts = [(float(r["lat"]), float(r["lon"])) for _, r in amenities_df.iterrows()]
    api = get_h3_api()
    for h in h3_cells:
        lat, lon = api.h3_to_geo(h)
        dmin = min(haversine_km(lon, lat, p_lon, p_lat) for p_lat,p_lon in pts)
        res[h] = round(dmin/speed_kmph*60,2)
    return res

def compute_infrastructure_stress(elec, water, sewage=None):
    e = normalize_series(elec)
    w = normalize_series(water)
    s = normalize_series(sewage) if sewage is not None else np.zeros_like(e)
    return np.clip(0.45*e + 0.35*w + 0.2*s,0,1)

def compute_urban_form_metrics(built_density, green_cover):
    bd, gr = normalize_series(built_density), normalize_series(green_cover)
    compactness = bd*(1-gr)
    sprawl = 1-compactness
    return compactness, sprawl

def compute_inequality_metrics(h3_cells, access_dicts):
    keys = list(access_dicts.keys())
    mat = np.full((len(keys),len(h3_cells)),np.nan)
    for i,k in enumerate(keys):
        d = access_dicts[k]
        for j,h in enumerate(h3_cells):
            mat[i,j] = float(d.get(h,np.nan))
    stds = np.nanstd(mat, axis=0)
    return normalize_series(np.nan_to_num(stds, nan=0.0))

# --------------------
# Main
# --------------------
def main(args):
    outdir = Path(args.out)
    ensure_dir(outdir)

    wards = load_wards(DEFAULT_WARDS)
    log.info("Loaded wards: %d features", len(wards))

    h3_cells = build_h3_cellset_from_wards(wards, res=args.h3_res, expand_k=args.expand_k)
    n_cells = len(h3_cells)
    log.info("H3 cells: %d", n_cells)

    api = get_h3_api()
    rows = []
    for h in h3_cells:
        lat, lon = api.h3_to_geo(h)
        rows.append({"h3_id":h,"geometry":h3_boundary_poly(h),"centroid_lat":lat,"centroid_lon":lon})
    h3_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    # Assign ward to H3 via centroid
    cent = h3_gdf.set_geometry("geometry").centroid
    cent_gdf = gpd.GeoDataFrame({"h3_id":h3_gdf["h3_id"]}, geometry=cent, crs="EPSG:4326")
    try:
        wards["geometry"] = wards.buffer(0)  # fix invalid polygons
        joined = gpd.sjoin(cent_gdf, wards[["ward_id","geometry"]], how="left", predicate="intersects")
        h3_gdf["ward_id"] = h3_gdf["h3_id"].map(joined.set_index("h3_id")["ward_id"])

# For any still unmatched, assign nearest ward
        unmatched = h3_gdf[h3_gdf["ward_id"].isna()]
        if len(unmatched):
            ward_centroids = wards.set_geometry("geometry").centroid
            for idx, row in unmatched.iterrows():
                distances = ward_centroids.distance(Point(row["centroid_lon"], row["centroid_lat"]))
                h3_gdf.at[idx, "ward_id"] = wards.iloc[distances.idxmin()]["ward_id"]
    except:
        h3_gdf["ward_id"]="ward_unknown"

    parks, schools, health, bus, metro = load_amenities(DEFAULT_PARKS_KML, DEFAULT_SCHOOLS_CSV, DEFAULT_HEALTH_KML, DEFAULT_BUS_STOPS_CSV, DEFAULT_METRO_STATIONS_CSV)
    temporal_stats = load_temporal_stats(DEFAULT_TEMPORAL, h3_cells)

    # proxies
    built_density = np.linspace(50,300,n_cells)
    green_cover_counts = np.zeros(n_cells)
    for i,h in enumerate(h3_cells):
        lat, lon = api.h3_to_geo(h)
        green_cover_counts[i] = parks.apply(lambda r: 1 if haversine_km(lon,lat,r["lon"],r["lat"])<0.7 else 0, axis=1).sum()
    green_cover_ratio = np.clip(green_cover_counts/green_cover_counts.max() if green_cover_counts.max()>0 else green_cover_counts,0,1)

    # accessibility
    access_bus = compute_accessibility_times(h3_cells, bus, 25.0)
    access_metro = compute_accessibility_times(h3_cells, metro, 35.0)
    access_hospital = compute_accessibility_times(h3_cells, health, 30.0)
    access_school = compute_accessibility_times(h3_cells, schools, 20.0)
    access_park = compute_accessibility_times(h3_cells, parks, 5.0)

    # infrastructure stress
    mean_aqi = temporal_stats["aqi"]
    mean_elec = temporal_stats["elec"]
    mean_water = temporal_stats["water"]
    mean_traffic = temporal_stats["traffic"]
    mean_mob = temporal_stats["mobility"]
    mean_sewage = np.linspace(10,100,n_cells)

    electricity_stress = compute_infrastructure_stress(mean_elec, mean_water, mean_sewage)
    water_stress = compute_infrastructure_stress(mean_elec, mean_water, mean_sewage)
    sewage_stress = compute_infrastructure_stress(mean_elec, mean_water, mean_sewage)
    pollution_stress = normalize_series(mean_aqi)
    mobility_stress = normalize_series(1.0/np.clip(mean_traffic,0.1,None))

    # urban form
    compactness, sprawl = compute_urban_form_metrics(built_density, green_cover_ratio)

    # inequality
    access_dicts = {"bus":access_bus,"metro":access_metro,"hospital":access_hospital,"school":access_school,"park":access_park}
    inequality_score = compute_inequality_metrics(h3_cells, access_dicts)

    # update H3 GDF
    h3_gdf["elec_stress"]=electricity_stress
    h3_gdf["water_stress"]=water_stress
    h3_gdf["sewage_stress"]=sewage_stress
    h3_gdf["pollution_stress"]=pollution_stress
    h3_gdf["mobility_stress"]=mobility_stress
    h3_gdf["compactness"]=compactness
    h3_gdf["sprawl"]=sprawl
    h3_gdf["inequality_score"]=inequality_score
    h3_gdf["access_bus"] = h3_gdf["h3_id"].map(access_bus)
    h3_gdf["access_metro"] = h3_gdf["h3_id"].map(access_metro)
    h3_gdf["access_hospital"] = h3_gdf["h3_id"].map(access_hospital)
    h3_gdf["access_school"] = h3_gdf["h3_id"].map(access_school)
    h3_gdf["access_park"] = h3_gdf["h3_id"].map(access_park)

    # --------------------
    # Ward-level aggregation
    # --------------------
    agg_dict = {
        "elec_stress":"mean","water_stress":"mean","sewage_stress":"mean",
        "pollution_stress":"mean","mobility_stress":"mean",
        "compactness":"mean","sprawl":"mean","inequality_score":"mean",
        "access_bus":"mean","access_metro":"mean","access_hospital":"mean",
        "access_school":"mean","access_park":"mean"
    }
    ward_gdf = h3_gdf.dissolve(by="ward_id", aggfunc=agg_dict).reset_index()
    # pick representative geometry: union of H3 polygons
    geom = h3_gdf.dissolve(by="ward_id")["geometry"].reset_index(drop=True)
    ward_gdf["geometry"] = geom

    # --------------------
    # Output
    # --------------------
    safe_write_geojson(h3_gdf, outdir/"h3_metrics.geojson")
    h3_gdf.drop(columns="geometry").to_csv(outdir/"h3_metrics.csv", index=False)

    safe_write_geojson(ward_gdf, outdir/"wards_metrics.geojson")
    ward_gdf.drop(columns="geometry").to_csv(outdir/"wards_metrics.csv", index=False)

    log.info("All outputs written to %s", outdir)

# --------------------
# CLI
# --------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h3-res", type=int, default=8, help="H3 resolution")
    parser.add_argument("--expand-k", type=int, default=2, help="H3 expansion rings")
    parser.add_argument("--out", type=str, default="./outputs", help="Output folder")
    args = parser.parse_args()
    main(args)
