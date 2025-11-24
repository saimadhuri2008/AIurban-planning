#!/usr/bin/env python3
"""
metro_stations.py

Load metro stations (KML if present), optionally seed with ridership XLSX,
assign wards robustly, optionally synthesize extra stations to reach a large count,
and produce station GeoJSON/CSV + hourly ridership parquet.

Usage:
  python src/data_pipeline/metro_stations.py --wards "C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson" --kml /mnt/data/bengaluru-metro-stations.kml --ridership /mnt/data/station-hourly.xlsx --out outputs --days 30 --target-stations 300

Defaults expect:
 - wards: C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson
 - kml: /mnt/data/bengaluru-metro-stations.kml
 - ridership: /mnt/data/station-hourly.xlsx

Produces:
 - outputs/metro_stations.geojson
 - outputs/metro_stations.csv
 - outputs/metro_station_ridership.parquet
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import math
import logging
import random

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# Try to import pyarrow for faster parquet write; fall back to pandas
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

# h3 not required here
RNG = np.random.RandomState(20251119)
random.seed(20251119)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("metro_stations")

# ----------------- Defaults -----------------
DEFAULT_WARDS = Path(r"C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
DEFAULT_KML = Path("/mnt/data/bengaluru-metro-stations.kml")
DEFAULT_RIDERS = Path("/mnt/data/station-hourly.xlsx")

# ----------------- Helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_kml_points(kml_path: Path):
    """Extract simple <Placemark> name + coordinates from KML. Returns DataFrame(name,lat,lon)."""
    if not kml_path.exists():
        return pd.DataFrame(columns=["name","lat","lon"])
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    import re
    placemarks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL | re.IGNORECASE)
    rows=[]
    for pm in placemarks:
        name_m = re.search(r"<name>(.*?)</name>", pm, flags=re.DOTALL|re.IGNORECASE)
        coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.DOTALL|re.IGNORECASE)
        if not coords_m:
            continue
        try:
            first = coords_m.group(1).strip().split()[0]
            lon, lat = [float(x) for x in first.split(",")[:2]]
            name = name_m.group(1).strip() if name_m else ""
            rows.append({"name": name, "lat": lat, "lon": lon})
        except Exception:
            continue
    return pd.DataFrame(rows)

def load_wards(wards_path: Path) -> gpd.GeoDataFrame:
    if not wards_path.exists():
        raise FileNotFoundError(f"Wards geojson not found: {wards_path}")
    wg = gpd.read_file(str(wards_path))
    # normalize to WGS84
    try:
        wg = wg.to_crs(epsg=4326)
    except Exception:
        pass
    # find ward id column heuristically
    ward_col = next((c for c in wg.columns if c.lower().startswith("ward")), None)
    if ward_col is None:
        ward_col = wg.columns[0]
    if ward_col != "ward_id":
        wg = wg.rename(columns={ward_col: "ward_id"})
    wg["ward_id"] = wg["ward_id"].astype(str)
    return wg

def assign_points_to_wards(points_gdf: gpd.GeoDataFrame, wards_gdf: gpd.GeoDataFrame, verbose=True):
    """
    Assign ward_id to each point robustly:
      1) spatial join intersects/within
      2) fallback: nearest ward by centroid (metric distance)
    """
    # ensure CRS
    if wards_gdf.crs is None:
        wards_gdf = wards_gdf.set_crs(epsg=4326)
    wards = wards_gdf.to_crs(epsg=4326)
    pts = points_gdf.to_crs(epsg=4326).copy()

    # 1) intersection join
    try:
        joined = gpd.sjoin(pts, wards[["ward_id","geometry"]], how="left", predicate="within")
    except Exception:
        # older geopandas: predicate="within" sometimes not available; use intersects
        joined = gpd.sjoin(pts, wards[["ward_id","geometry"]], how="left", predicate="intersects")
    # map ward_id back
    joined = joined.reset_index().set_index("index")
    pts["ward_id"] = joined["ward_id"]

    matched = pts["ward_id"].notna().sum()
    if verbose:
        log.info("Assigned by polygon intersection: %d / %d", int(matched), len(pts))

    # 2) fallback nearest by centroid for unmatched
    unmatched = pts[pts["ward_id"].isna()].copy()
    if len(unmatched) > 0:
        log.info("Nearest-ward fallback for %d stations...", len(unmatched))
        # compute ward centroids in metric CRS for distance
        wards_cent = wards.copy().to_crs(epsg=3857)
        wards_cent["wcx"] = wards_cent.geometry.centroid.x
        wards_cent["wcy"] = wards_cent.geometry.centroid.y
        ward_points = gpd.GeoDataFrame(wards_cent[["ward_id"]], geometry=wards_cent.geometry.centroid, crs=wards_cent.crs)
        sindex = ward_points.sindex

        # transform unmatched to metric
        unmatched_m = unmatched.to_crs(epsg=3857)
        # arrays
        ward_coords = list(zip(wards_cent["wcx"].values, wards_cent["wcy"].values))
        ward_ids = list(wards_cent["ward_id"].values)

        assignments = {}
        for idx, row in unmatched_m.iterrows():
            px = row.geometry.x; py = row.geometry.y
            try:
                # query nearest candidates (spatial index)
                cand_idx = list(sindex.nearest((px,py,px,py), 5)) if hasattr(sindex, "nearest") else list(sindex.nearest((px,py,px,py)))
            except Exception:
                try:
                    cand_idx = list(sindex.intersection((px-2000,py-2000,px+2000,py+2000)))
                except Exception:
                    cand_idx = list(range(len(ward_coords)))
            best_id = None; best_d = float("inf")
            for c in cand_idx:
                try:
                    wx, wy = ward_coords[c]
                    d = (px - wx)**2 + (py - wy)**2
                    if d < best_d:
                        best_d = d; best_id = ward_ids[c]
                except Exception:
                    continue
            if best_id is None:
                # brute force fallback
                dists = [ (px-wx)**2 + (py-wy)**2 for (wx,wy) in ward_coords ]
                best = int(np.argmin(dists))
                best_id = ward_ids[best]
            assignments[idx] = best_id
        for idx, wid in assignments.items():
            pts.at[idx, "ward_id"] = wid
        log.info("Nearest-ward assigned: %d", len(assignments))

    # any remaining fill with "ward_unknown"
    pts["ward_id"] = pts["ward_id"].fillna("ward_unknown")
    n_unknown = (pts["ward_id"] == "ward_unknown").sum()
    if n_unknown > 0:
        log.warning("%d stations remain with ward_unknown after assignment", int(n_unknown))
    return pts

def synth_ridership_for_station(n_hours, base=100.0, diurnal_scale=2.0):
    """Generate synthetic hourly ridership for n_hours using a diurnal pattern."""
    mult = np.sin(np.linspace(0, 2*math.pi, n_hours) - math.pi/2) * 0.5 + 0.5
    # morning + evening peaks + noise
    vals = base * (0.6 + diurnal_scale * mult) + np.random.normal(0, base*0.1, size=n_hours)
    vals = np.clip(vals, 0, None).round().astype(int)
    return vals

# ----------------- main routine -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wards", default=str(DEFAULT_WARDS))
    p.add_argument("--kml", default=str(DEFAULT_KML))
    p.add_argument("--ridership", default=str(DEFAULT_RIDERS))
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--freq", default="h", help="timestamp freq (use 'h' for hourly')")
    p.add_argument("--target-stations", type=int, default=300, help="Expand stations to at least this count by synthesizing")
    p.add_argument("--seed-augment", type=int, default=20251119)
    args = p.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # load wards
    wards = None
    try:
        wards = load_wards(Path(args.wards))
        log.info("Loaded wards: %d features from %s", len(wards), args.wards)
    except Exception as e:
        log.warning("Could not load wards: %s", e)

    # load KML stations
    kml_path = Path(args.kml)
    stations_df = parse_kml_points(kml_path) if kml_path.exists() else pd.DataFrame(columns=["name","lat","lon"])
    if stations_df.empty:
        log.warning("No stations found in KML: %s", kml_path)
    else:
        log.info("Parsed %d stations from KML: %s", len(stations_df), kml_path)

    # If ridership file exists, try to read per-station historical seeding
    riders_path = Path(args.ridership)
    ridership_seed = None
    if riders_path.exists():
        try:
            # try reading excel; we'll attempt to infer a sheet/structure
            xr = pd.read_excel(riders_path, sheet_name=0)
            # heuristics: if has station_id and hour columns, use
            if "station_id" in xr.columns and any(col.lower().startswith("hour") or col.lower().startswith("timestamp") for col in xr.columns):
                ridership_seed = xr
                log.info("Loaded ridership seed table with shape %s", xr.shape)
            else:
                # fallback: keep entire sheet as seed (we'll use station-level aggregated numbers)
                ridership_seed = xr
                log.info("Loaded ridership sheet as seed (shape %s)", xr.shape)
        except Exception as e:
            log.warning("Failed to read ridership xlsx: %s", e)
    else:
        log.info("No ridership file at %s; will synthesize hourly ridership.", riders_path)

    # Build GeoDataFrame for stations
    if stations_df.empty:
        # create a few synthetic seed stations around city center if none
        log.info("Generating 50 synthetic seed stations around city center (12.9716,77.5946)")
        seed_cent_lat, seed_cent_lon = 12.9716, 77.5946
        recs=[]
        for i in range(50):
            lat = seed_cent_lat + RNG.normal(0, 0.05)
            lon = seed_cent_lon + RNG.normal(0, 0.05)
            recs.append({"name": f"synt_metro_{i+1}", "lat": lat, "lon": lon})
        stations_df = pd.DataFrame(recs)

    stations_gdf = gpd.GeoDataFrame(stations_df.copy(), geometry=[Point(xy) for xy in zip(stations_df["lon"], stations_df["lat"])], crs="EPSG:4326")

    # If too few stations, augment by jittering existing ones to reach target count
    target = max(1, int(args.target_stations))
    existing = len(stations_gdf)
    if existing < target:
        log.info("Augmenting stations: existing=%d target=%d", existing, target)
        added = []
        to_make = target - existing
        base_pts = list(stations_gdf[["lat","lon","name"]].to_dict(orient="records"))
        for i in range(to_make):
            b = base_pts[i % len(base_pts)]
            # jitter scale depends on index to produce spatial spread
            jitter_km = 0.5 + RNG.rand() * 3.0  # up to ~3km jitter
            # convert approx km to degrees (~111 km per degree)
            jitter_deg = jitter_km / 111.0
            lat = float(b["lat"]) + RNG.normal(0, jitter_deg)
            lon = float(b["lon"]) + RNG.normal(0, jitter_deg)
            name = f"{b.get('name','metro')}_aug_{i+1}"
            added.append({"name": name, "lat": lat, "lon": lon})
        aug_df = pd.DataFrame(added)
        aug_gdf = gpd.GeoDataFrame(aug_df, geometry=[Point(xy) for xy in zip(aug_df["lon"], aug_df["lat"])], crs="EPSG:4326")
        stations_gdf = pd.concat([stations_gdf.reset_index(drop=True), aug_gdf.reset_index(drop=True)], ignore_index=True)
        log.info("Augmented stations created. New total: %d", len(stations_gdf))

    # assign station_id
    stations_gdf = stations_gdf.reset_index(drop=True)
    stations_gdf["station_id"] = stations_gdf.index.map(lambda x: f"station_{x+1}")
    # station_name column
    if "name" not in stations_gdf.columns:
        stations_gdf["name"] = stations_gdf["station_id"]

    # assign wards robustly if wards loaded
    if wards is not None and not wards.empty:
        stations_gdf = assign_points_to_wards(stations_gdf, wards, verbose=True)
    else:
        stations_gdf["ward_id"] = "ward_unknown"
        log.warning("Wards not available: all stations get ward_unknown")

    # add line (if available in name try to infer), otherwise mark 'unknown'
    def infer_line_from_name(n):
        n = str(n).lower()
        if "purple" in n: return "purple"
        if "green" in n: return "green"
        if "yellow" in n: return "yellow"
        if "pink" in n: return "pink"
        return "unknown"
    stations_gdf["line"] = stations_gdf["name"].apply(infer_line_from_name)

    # prepare timestamp index for ridership
    days = int(args.days)
    freq = args.freq or "h"
    # use lowercase 'h' to avoid FutureWarning
    end = pd.Timestamp.now().floor("h")
    n_hours = 24 * days
    timestamps = pd.date_range(end=end, periods=n_hours, freq=freq, tz="Asia/Kolkata")

    # if ridership_seed exists, try to map station-level base ridership values; else synth
    base_by_station = {}
    if ridership_seed is not None:
        # attempt: if sheet has station_id and base_hour columns, aggregate per station
        if "station_id" in ridership_seed.columns:
            # compute mean or last value per station in available columns
            try:
                if "hour" in ridership_seed.columns or "timestamp" in ridership_seed.columns:
                    # tidy case: pivot into station->mean
                    grp = ridership_seed.groupby("station_id").mean(numeric_only=True)
                    for sid in stations_gdf["station_id"]:
                        # try to match by name or station id heuristics
                        mapped = None
                        if sid in grp.index:
                            mapped = float(grp.loc[sid].mean())
                        else:
                            # sometimes seed uses station names - try fuzzy match on name
                            pass
                        if mapped is not None and not math.isnan(mapped):
                            base_by_station[sid] = max(5.0, mapped)
                else:
                    # fallback: use numeric columns mean as base scale
                    numeric_mean = ridership_seed.select_dtypes("number").mean().mean()
                    for sid in stations_gdf["station_id"]:
                        base_by_station[sid] = float(max(5.0, numeric_mean if not math.isnan(numeric_mean) else 50.0))
            except Exception as e:
                log.warning("Could not derive station base ridership from seed: %s", e)
        else:
            # generic fallback
            numeric_mean = ridership_seed.select_dtypes("number").mean().mean()
            for sid in stations_gdf["station_id"]:
                base_by_station[sid] = float(max(5.0, numeric_mean if not math.isnan(numeric_mean) else 50.0))
    # ensure every station has a base
    for sid in stations_gdf["station_id"]:
        if sid not in base_by_station:
            # base depends on line/population proxy: randomize but deterministic
            base_by_station[sid] = float(max(5.0, 50.0 + RNG.normal(0, 30)))

    # create ridership timeseries frame (wide or long). We'll produce long format parquet.
    rows = []
    log.info("Generating hourly ridership for %d stations x %d hours = %d rows", len(stations_gdf), n_hours, len(stations_gdf)*n_hours)
    for _, s in tqdm(stations_gdf.iterrows(), total=len(stations_gdf), desc="stations"):
        sid = s["station_id"]
        base = base_by_station.get(sid, 50.0)
        # diurnal pattern scaled by station base and slight weekday effect
        diurnal = (np.sin(np.linspace(0, 2*math.pi, n_hours) - math.pi/2) * 0.5 + 0.5)
        # weekly pattern: reduce nights and weekends
        # generate weekdays mask
        # create a repeating 24*7 pattern starting from timestamps[0]
        mask_weekday = np.array([1.0 if pd.Timestamp(ts).weekday() < 5 else 0.7 for ts in timestamps])
        noise = RNG.normal(0, base*0.12, size=n_hours)
        vals = np.clip(base * (0.6 + 2.0 * diurnal) * mask_weekday + noise, 0, None).round().astype(int)
        # optionally bump ridership for core lines
        if s.get("line","unknown").lower() in ("purple","green"):
            vals = (vals * 1.1).astype(int)
        for t_idx, ts in enumerate(timestamps):
            rows.append({
                "station_id": sid,
                "timestamp": pd.Timestamp(ts).isoformat(),
                "ridership": int(vals[t_idx])
            })

    ridership_df = pd.DataFrame(rows)
    # write ridership parquet (use pyarrow if available)
    ridership_out = outdir / "metro_station_ridership.parquet"
    try:
        if pa is not None and pq is not None:
            table = pa.Table.from_pandas(ridership_df)
            pq.write_table(table, str(ridership_out), compression="snappy")
        else:
            ridership_df.to_parquet(ridership_out, compression="snappy", index=False)
        log.info("WROTE ridership timeseries: %s", ridership_out)
    except Exception as e:
        log.warning("Failed to write ridership parquet using pyarrow: %s. Falling back to pandas.", e)
        ridership_df.to_parquet(ridership_out, compression="snappy", index=False)

    # Export stations GeoJSON + CSV (include ward_id)
    stations_out_geo = outdir / "metro_stations.geojson"
    stations_out_csv = outdir / "metro_stations.csv"
    # ensure 'geometry' is set and crs is WGS84
    stations_gdf = stations_gdf.set_geometry("geometry").to_crs(epsg=4326)
    # drop extraneous geometry dtype columns if any, then write
    try:
        stations_gdf.to_file(str(stations_out_geo), driver="GeoJSON")
        # CSV without geometry (but keep lat/lon & ward)
        save_csv = stations_gdf.drop(columns=["geometry"]).copy()
        # ensure lat/lon numeric columns exist
        if "lat" not in save_csv.columns:
            save_csv["lat"] = stations_gdf.geometry.y
        if "lon" not in save_csv.columns:
            save_csv["lon"] = stations_gdf.geometry.x
        save_csv.to_csv(str(stations_out_csv), index=False)
        log.info("WROTE metro stations: %s, %s", stations_out_geo, stations_out_csv)
    except Exception as e:
        log.warning("GeoJSON write failed (%s). Writing CSV with WKT geometry instead.", e)
        stations_gdf["geometry_wkt"] = stations_gdf.geometry.apply(lambda g: g.wkt if g is not None else None)
        stations_gdf.drop(columns=["geometry"]).to_csv(str(stations_out_csv), index=False)
        log.info("WROTE metro stations CSV with geometry WKT: %s", stations_out_csv)

    log.info("Done. Stations: %d, Ridership rows: %d", len(stations_gdf), len(ridership_df))

if __name__ == "__main__":
    main()
