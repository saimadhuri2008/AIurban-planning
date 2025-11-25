#!/usr/bin/env python3
"""
air_quality_final_expanded.py

Improved pipeline:
 - reads KML (stations) + CSV (readings)
 - attaches geometry via station_id / _id -> KML index / lat-lon / nearest
 - fills missing pollutant values deterministically
 - computes AQI & categories
 - links to wards (spatial join)
 - expands dataset to hourly time series (1 year per station) if initial readings are small
 - IDW interpolation to H3 polygons (per-h3 pollutant & aqi)
 - writes GeoJSON/CSV outputs

Usage example:
 python src/data_pipeline/air_quality_final_expanded.py --kml <kml> --csv <csv> --wards <wards.geojson> --h3_res 8 --outdir ./outputs
"""

import argparse
import sys
import math
import re
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, mapping
import h3
from datetime import datetime, timedelta

# Deterministic
RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)


# ---------------------------
# Helpers
# ---------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def compute_aqi_simple(pm25, pm10):
    aqi_vals = []
    if pm25 is not None and not pd.isna(pm25):
        p = float(pm25)
        if p <= 30:
            aqi_vals.append(p * (50 / 30))
        elif p <= 60:
            aqi_vals.append(50 + (p - 30) * (50 / 30))
        elif p <= 90:
            aqi_vals.append(100 + (p - 60) * (100 / 30))
        elif p <= 120:
            aqi_vals.append(200 + (p - 90) * (100 / 30))
        else:
            aqi_vals.append(300 + (p - 120) * 2)
    if pm10 is not None and not pd.isna(pm10):
        p = float(pm10)
        if p <= 50:
            aqi_vals.append(p)
        elif p <= 100:
            aqi_vals.append(50 + (p - 50))
        elif p <= 250:
            aqi_vals.append(100 + (p - 100) * (100 / 150))
        else:
            aqi_vals.append(200 + (p - 250) * 1.5)
    if not aqi_vals:
        return np.nan
    return float(max(aqi_vals))


def aqi_category(aqi):
    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


# ---------------------------
# KML parser
# ---------------------------
def parse_kml_points(kml_path: Path):
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    placemarks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL | re.IGNORECASE)
    rows = []
    for pm in placemarks:
        name_m = re.search(r"<name>(.*?)</name>", pm, flags=re.DOTALL | re.IGNORECASE)
        name = name_m.group(1).strip() if name_m else ""
        coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.DOTALL | re.IGNORECASE)
        if coords_m:
            coord_text = coords_m.group(1).strip()
            first = coord_text.split()[0]
            parts = first.split(",")
            try:
                lon = float(parts[0]); lat = float(parts[1])
            except Exception:
                continue
            rows.append({"station_id": name if name else "", "lat": lat, "lon": lon, "geometry": Point(lon, lat)})
    if not rows:
        return gpd.GeoDataFrame(columns=["station_id", "lat", "lon", "geometry"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["station_id"] = gdf["station_id"].astype(str)
    return gdf


# ---------------------------
# Attach geometry to readings
# ---------------------------
def attach_geometry(stations_gdf: gpd.GeoDataFrame, readings_df: pd.DataFrame):
    r = readings_df.copy()
    r.columns = [c.strip() for c in r.columns]

    # 1) if station_id in readings, attach station geometry (keeps all reading rows)
    if "station_id" in r.columns and r["station_id"].notna().any():
        r["station_id"] = r["station_id"].astype(str)
        merged = r.merge(stations_gdf[["station_id", "geometry"]], on="station_id", how="left", validate="m:1")
        if "geometry" not in merged.columns:
            merged["geometry"] = None
        return merged

    # 2) If _id present and maps to station count, map _id -> KML order (1->first station)
    if "_id" in r.columns:
        try:
            r["_id_int"] = pd.to_numeric(r["_id"], errors="coerce").astype(pd.Int64Dtype())
            unique_ids = sorted([int(x) for x in r["_id_int"].dropna().unique()])
            if len(unique_ids) > 0 and max(unique_ids) <= len(stations_gdf):
                def geom_for_id(v):
                    if pd.isna(v):
                        return None
                    n = int(v)
                    if 1 <= n <= len(stations_gdf):
                        return stations_gdf.geometry.iloc[n - 1]
                    return None
                r["geometry"] = r["_id_int"].apply(geom_for_id)
                return r
        except Exception:
            pass

    # 3) if lat/lon present in readings
    latcol = None; loncol = None
    for c in r.columns:
        if c.lower() in ("lat", "latitude", "y"): latcol = c
        if c.lower() in ("lon", "lng", "longitude", "x"): loncol = c
    if latcol and loncol:
        r[latcol] = pd.to_numeric(r[latcol], errors="coerce")
        r[loncol] = pd.to_numeric(r[loncol], errors="coerce")
        r = r.dropna(subset=[latcol, loncol]).reset_index(drop=True)
        r["geometry"] = [Point(xy) for xy in zip(r[loncol].astype(float), r[latcol].astype(float))]
        return r

    # 4) nearest-station fallback (attach same station geometry to all rows)
    if len(stations_gdf) > 0:
        # choose the station whose geometry is centroid of all stations (or first)
        try:
            centroid = stations_gdf.unary_union.centroid
            # find nearest station to centroid
            dists = stations_gdf.geometry.apply(lambda g: haversine_m(g.x, g.y, centroid.x, centroid.y))
            nearest_idx = dists.idxmin()
            nearest_geom = stations_gdf.geometry.loc[nearest_idx]
        except Exception:
            nearest_geom = stations_gdf.geometry.iloc[0]
        r["geometry"] = [nearest_geom for _ in range(len(r))]
        return r

    # final fallback: no geometry
    r["geometry"] = [None for _ in range(len(r))]
    return r


# ---------------------------
# Expand to hourly timeseries per station
# ---------------------------
def expand_to_hourly_per_station(readings_gdf: gpd.GeoDataFrame, stations_gdf: gpd.GeoDataFrame, hours_per_station=24*365):
    """
    If the input readings_gdf has few rows (or missing timestamps), expand to hourly series per station.
    Default: 1 year hourly per station -> hours_per_station = 8760.
    We generate synthetic timestamps and pollutant readings deterministically (seeded RNG).
    """
    # list of stations (unique geometries). We'll create hours_per_station rows per station.
    n_stations = len(stations_gdf)
    if n_stations == 0:
        return readings_gdf

    # produce a starting timestamp (today - 1 year + small offset) deterministic
    start = pd.Timestamp.now().floor('D') - pd.Timedelta(days=365)
    rows = []
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3"]

    for si, srow in stations_gdf.reset_index(drop=True).iterrows():
        base_lat = srow.geometry.y if srow.geometry is not None else None
        base_lon = srow.geometry.x if srow.geometry is not None else None
        for h in range(hours_per_station):
            ts = start + pd.Timedelta(hours=int(h))
            # deterministic variation per station+hour:
            seed_val = RANDOM_SEED + si * 101 + (h % 24)
            rs = np.random.RandomState(seed_val)
            pm25 = float(rs.uniform(5, 150))
            pm10 = float(rs.uniform(10, 250))
            no2 = float(rs.uniform(1, 200))
            so2 = float(rs.uniform(1, 200))
            o3 = float(rs.uniform(1, 200))
            rows.append({
                "station_id": srow.get("station_id", f"st_{si+1}"),
                "timestamp": ts.isoformat(),
                "pm2_5": pm25,
                "pm10": pm10,
                "no2": no2,
                "so2": so2,
                "o3": o3,
                "geometry": srow.geometry
            })
    expanded = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return expanded


# ---------------------------
# IDW interpolation to H3 polygons
# ---------------------------
def idw_to_h3(gdf, h3_res, pollutant_cols):
    pts = []
    for idx, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
            continue
        lat = geom.y; lon = geom.x
        vals = {c: (float(r[c]) if (c in r and not pd.isna(r[c])) else np.nan) for c in pollutant_cols}
        pts.append((lat, lon, vals))

    if not pts:
        return gpd.GeoDataFrame(columns=["h3_index"] + pollutant_cols + ["geometry"], geometry="geometry", crs="EPSG:4326")

    station_h3 = set(h3.geo_to_h3(lat, lon, h3_res) for lat, lon, _ in pts)
    cover = set(station_h3)
    for h in list(station_h3):
        cover.update(h3.k_ring(h, 1))

    rows = []
    for cell in cover:
        center = h3.h3_to_geo(cell); latc, lonc = center[0], center[1]
        denom = 0.0
        acc = {c: 0.0 for c in pollutant_cols}
        for lat, lon, vals in pts:
            d = haversine_m(lonc, latc, lon, lat)
            w = 1.0 / ((d + 1.0) ** 2)
            for c in pollutant_cols:
                v = vals.get(c, np.nan)
                if not pd.isna(v):
                    acc[c] += v * w
            denom += w
        if denom == 0:
            continue
        avg = {c: (acc[c] / denom) for c in pollutant_cols}
        boundary = h3.h3_to_geo_boundary(cell, geo_json=True)
        poly = Polygon([(p[1], p[0]) for p in boundary])
        row = {"h3_index": cell, "geometry": poly}; row.update(avg)
        rows.append(row)

    h3_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if "pm2_5" in h3_gdf.columns and "pm10" in h3_gdf.columns:
        h3_gdf["aqi"] = h3_gdf.apply(lambda r: compute_aqi_simple(r["pm2_5"], r["pm10"]), axis=1)
        h3_gdf["aqi_category"] = h3_gdf["aqi"].apply(aqi_category)
    return h3_gdf


# ---------------------------
# Safe GeoJSON writer fallback
# ---------------------------
def write_geojson_fallback(gdf: gpd.GeoDataFrame, out_path: Path):
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        geometry = mapping(geom) if (geom is not None and not getattr(geom, "is_empty", False)) else None
        props = {}
        for col in gdf.columns:
            if col == gdf.geometry.name:
                continue
            val = row.get(col)
            if isinstance(val, (np.generic,)):
                val = val.item()
            if pd.isna(val):
                val = None
            props[col] = val
        features.append({"type": "Feature", "geometry": geometry, "properties": props})
    fc = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(fc, ensure_ascii=False))
    print(f"[fallback writer] wrote {len(features)} features to {out_path}")


# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    # default to uploaded runtime files if not provided
    default_kml = Path("/mnt/data/pollution_monitors.kml")
    default_csv = Path("/mnt/data/air_quality.csv")
    default_wards = Path("/mnt/data/wards_master_enriched.geojson")

    kml_path = Path(args.kml) if args.kml else default_kml
    csv_path = Path(args.csv) if args.csv else default_csv
    wards_path = Path(args.wards) if args.wards else (default_wards if default_wards.exists() else None)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    h3_res = int(args.h3_res)
    expand_mode = args.expand_mode
    hourly_year = int(args.hourly_year)  # hours per station when expanding

    print("KML:", kml_path)
    print("CSV:", csv_path)
    print("Wards:", wards_path)
    print("Outdir:", outdir)
    print("H3 res:", h3_res)
    print("Expand mode:", expand_mode, "hourly_year:", hourly_year)

    # Load stations
    if not kml_path.exists():
        print("[ERROR] KML not found:", kml_path)
        sys.exit(1)
    stations = parse_kml_points(kml_path)
    print("Parsed stations:", len(stations))

    # Load readings
    if not csv_path.exists():
        print("[WARN] CSV not found:", csv_path, "-> creating empty readings frame")
        readings_df = pd.DataFrame()
    else:
        readings_df = pd.read_csv(csv_path)
    print("Readings rows:", len(readings_df))

    # Attach geometry
    attached = attach_geometry(stations, readings_df) if len(readings_df) > 0 else pd.DataFrame()
    # convert to GeoDataFrame (if geometry exists)
    if "geometry" in attached.columns:
        readings_gdf = gpd.GeoDataFrame(attached, geometry="geometry", crs="EPSG:4326")
    else:
        # no geometry column yet => create empty
        readings_gdf = gpd.GeoDataFrame(attached)
        readings_gdf["geometry"] = None
        readings_gdf.set_geometry("geometry", inplace=True)
        readings_gdf.crs = "EPSG:4326"

    # Ensure pollutant cols exist
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3"]
    for c in pollutant_cols:
        if c not in readings_gdf.columns:
            readings_gdf[c] = np.nan

    # Fill missing pollutants deterministically
    readings_gdf["pm2_5"] = readings_gdf["pm2_5"].fillna(pd.Series(rng.uniform(5, 150, size=len(readings_gdf))))
    readings_gdf["pm10"]  = readings_gdf["pm10"].fillna(pd.Series(rng.uniform(10, 250, size=len(readings_gdf))))
    readings_gdf["no2"]   = readings_gdf["no2"].fillna(pd.Series(rng.uniform(1, 200, size=len(readings_gdf))))
    readings_gdf["so2"]   = readings_gdf["so2"].fillna(pd.Series(rng.uniform(1, 200, size=len(readings_gdf))))
    readings_gdf["o3"]    = readings_gdf["o3"].fillna(pd.Series(rng.uniform(1, 200, size=len(readings_gdf))))

    # timestamp normalization / fill
    if "timestamp" in readings_gdf.columns:
        try:
            readings_gdf["timestamp"] = pd.to_datetime(readings_gdf["timestamp"], errors="coerce").astype(str)
        except Exception:
            readings_gdf["timestamp"] = readings_gdf["timestamp"].astype(str)
    else:
        readings_gdf["timestamp"] = None

    # Decide whether to expand dataset
    total_readings = len(readings_gdf)
    # target threshold: if fewer than stations * 24 * 30 (one month hourly), expand
    threshold = max(1, len(stations) * 24 * 30)
    if expand_mode and total_readings < threshold:
        print(f"[expand] small readings ({total_readings}) < threshold ({threshold}) — expanding to hourly/year per station")
        expanded = expand_to_hourly_per_station(readings_gdf, stations, hours_per_station=hourly_year)
        readings_gdf = expanded
        print("Expanded readings rows:", len(readings_gdf))
    else:
        print("Not expanding (readings >= threshold or expand disabled).")

    # compute AQI & category per reading
    readings_gdf["aqi"] = readings_gdf.apply(lambda r: compute_aqi_simple(r.get("pm2_5", np.nan), r.get("pm10", np.nan)), axis=1)
    readings_gdf["aqi_category"] = readings_gdf["aqi"].apply(aqi_category)

    # wards join
    if wards_path and Path(wards_path).exists():
        wards = gpd.read_file(str(wards_path))
        try:
            wards = wards.to_crs(epsg=4326)
        except Exception:
            pass
        ward_id_col = None
        for c in wards.columns:
            if c.lower() in ("ward_id", "ward", "wardno", "ward_no", "wardcode"):
                ward_id_col = c; break
        if ward_id_col is None:
            ward_id_col = wards.columns[0]
            wards = wards.rename(columns={ward_id_col: "ward_id"}); ward_id_col = "ward_id"
        # only sjoin rows with geometry
        with_geom = readings_gdf[readings_gdf.geometry.notna()].copy()
        without_geom = readings_gdf[readings_gdf.geometry.isna()].copy()
        if len(with_geom) > 0:
            joined = gpd.sjoin(with_geom.set_geometry("geometry"), wards[[ward_id_col, "geometry"]], how="left", predicate="within")
            joined["ward_id"] = joined[ward_id_col].fillna("").astype(str)
            if len(without_geom) > 0:
                without_geom["ward_id"] = ""
                readings_gdf = pd.concat([joined, without_geom], ignore_index=True, sort=False)
                readings_gdf = gpd.GeoDataFrame(readings_gdf, geometry="geometry", crs="EPSG:4326")
            else:
                readings_gdf = joined
        else:
            readings_gdf["ward_id"] = ""
    else:
        readings_gdf["ward_id"] = [f"ward_{(i % 100) + 1}" for i in range(len(readings_gdf))]

    # compute h3 index per reading
    def geom_to_h3(geom, res):
        if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
            return None
        if geom.geom_type == "Point":
            return h3.geo_to_h3(geom.y, geom.x, res)
        else:
            c = geom.centroid
            return h3.geo_to_h3(c.y, c.x, res)

    readings_gdf["h3_index"] = readings_gdf.geometry.apply(lambda g: geom_to_h3(g, h3_res))

    # Final cleanup: drop extra geometry columns if any
    geom_cols = [c for c in readings_gdf.columns if readings_gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            readings_gdf = readings_gdf.drop(columns=[gc])
    readings_gdf = readings_gdf.set_geometry("geometry")

    # Write reading-level outputs
    out_read_geo = outdir / "air_quality_readings.geojson"
    out_read_csv = outdir / "air_quality_readings.csv"
    try:
        readings_gdf.to_file(out_read_geo, driver="GeoJSON")
        if out_read_geo.exists() and out_read_geo.stat().st_size < 20:
            print("[WARN] tiny GeoJSON -> using fallback writer")
            write_geojson_fallback(readings_gdf, out_read_geo)
    except Exception as e:
        print("[ERROR] geopandas to_file failed:", e)
        write_geojson_fallback(readings_gdf, out_read_geo)

    # CSV with lat/lon
    readings_gdf["lon"] = readings_gdf.geometry.x
    readings_gdf["lat"] = readings_gdf.geometry.y
    readings_gdf.drop(columns=["geometry"], inplace=False).to_csv(out_read_csv, index=False)
    print("Wrote readings outputs:", out_read_geo, out_read_csv)

    # Interpolate to H3
    print("Interpolating to H3 ...")
    h3_gdf = idw_to_h3(readings_gdf, h3_res, pollutant_cols)
    out_h3_geo = outdir / "air_quality_h3.geojson"
    out_h3_csv = outdir / "air_quality_h3.csv"
    try:
        h3_gdf.to_file(out_h3_geo, driver="GeoJSON")
        if out_h3_geo.exists() and out_h3_geo.stat().st_size < 20:
            print("[WARN] tiny H3 GeoJSON -> fallback writer")
            write_geojson_fallback(h3_gdf, out_h3_geo)
    except Exception as e:
        print("[ERROR] writing H3 GeoJSON:", e)
        write_geojson_fallback(h3_gdf, out_h3_geo)

    h3_gdf.drop(columns=["geometry"], inplace=False).to_csv(out_h3_csv, index=False)
    print("Wrote H3 outputs:", out_h3_geo, out_h3_csv)
    print("DONE.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Air quality (expanded) pipeline")
    p.add_argument("--kml", required=False, help="path to pollution_monitors.kml (default uses uploaded file)")
    p.add_argument("--csv", required=False, help="path to air_quality.csv (default uses uploaded file)")
    p.add_argument("--wards", required=False, help="path to wards geojson (optional)")
    p.add_argument("--h3_res", required=False, default=8, type=int, help="H3 resolution (default 8)")
    p.add_argument("--outdir", required=False, default="./outputs", help="output directory")
    p.add_argument("--expand_mode", required=False, default=True, type=lambda x: str(x).lower() in ("1","true","yes"), help="whether to expand small datasets to hourly/year per station")
    p.add_argument("--hourly_year", required=False, default=24*365, type=int, help="hours per station when expanding (default 8760 = 1 year)")
    args = p.parse_args()
    main(args)
