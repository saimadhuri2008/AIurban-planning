#!/usr/bin/env python3
"""
water_sewage.py

Converts water/sewage KMLs to GeoJSON, preserving any available attributes
and generating synthetic values where fields are missing.

Outputs:
 - water_network.geojson
 - sewage_network.geojson

Dependencies:
 - geopandas
 - pandas
 - shapely
 - numpy

Install:
    pip install geopandas pandas shapely numpy

Usage (PowerShell single-line):
    python .\water_sewage.py --water "C:\path\to\water_supply.kml" --sewage "C:\path\to\sewage_network.kml" --stp "C:\path\to\stp_locations.csv" --outdir ".\outputs"
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def parse_kml_get_placemarks(kml_path: Path):
    """
    Lightweight KML parser: extracts Placemark name/description and LineString/Point geometries.
    Returns GeoDataFrame with ['name','description','geometry'] and crs EPSG:4326.
    """
    text = kml_path.read_text(encoding="utf-8", errors="replace")
    placemark_blocks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", text, flags=re.DOTALL | re.IGNORECASE)
    records = []
    for block in placemark_blocks:
        name_m = re.search(r"<name>(.*?)</name>", block, flags=re.DOTALL | re.IGNORECASE)
        desc_m = re.search(r"<description>(.*?)</description>", block, flags=re.DOTALL | re.IGNORECASE)
        name = name_m.group(1).strip() if name_m else ""
        desc = desc_m.group(1).strip() if desc_m else ""
        # LineString coords
        ls_matches = re.findall(r"<LineString[^>]*>(.*?)</LineString>", block, flags=re.DOTALL | re.IGNORECASE)
        geoms = []
        for ls in ls_matches:
            coords_m = re.search(r"<coordinates>(.*?)</coordinates>", ls, flags=re.DOTALL | re.IGNORECASE)
            if coords_m:
                coords_text = coords_m.group(1).strip()
                pts = []
                for token in re.split(r'\s+', coords_text.strip()):
                    if token.strip() == "":
                        continue
                    parts = token.split(",")
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0]); lat = float(parts[1])
                            pts.append((lon, lat))
                        except Exception:
                            continue
                if len(pts) >= 2:
                    geoms.append(LineString(pts))
        # Point fallback
        if not geoms:
            pt_matches = re.findall(r"<Point[^>]*>(.*?)</Point>", block, flags=re.DOTALL | re.IGNORECASE)
            for pt in pt_matches:
                coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pt, flags=re.DOTALL | re.IGNORECASE)
                if coords_m:
                    token = coords_m.group(1).strip()
                    parts = token.split(",")
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0]); lat = float(parts[1])
                            geoms.append(Point(lon, lat))
                        except Exception:
                            continue
        for g in geoms:
            records.append({"name": name, "description": desc, "geometry": g})
    if not records:
        return gpd.GeoDataFrame(columns=["name", "description", "geometry"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf


def extract_field_from_text(text: str, field_patterns: dict):
    out = {}
    if not text:
        text = ""
    for field, pat in field_patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        out[field] = m.group(1).strip() if m else None
    return out


def convert_points_to_short_lines(gdf: gpd.GeoDataFrame, offset_deg=0.00005):
    """
    Convert point geometries into tiny LineStrings so network features are lines.
    """
    new_geoms = []
    for geom in gdf.geometry:
        if geom is None:
            new_geoms.append(None)
        elif geom.geom_type == "Point":
            p = geom
            new_geoms.append(LineString([(p.x, p.y), (p.x + offset_deg, p.y)]))
        elif geom.geom_type == "MultiPoint":
            pts = list(geom.geoms)
            if len(pts) >= 2:
                new_geoms.append(LineString([(pts[0].x, pts[0].y), (pts[-1].x, pts[-1].y)]))
            else:
                p = pts[0]
                new_geoms.append(LineString([(p.x, p.y), (p.x + offset_deg, p.y)]))
        else:
            new_geoms.append(geom)
    new_gdf = gdf.copy()
    new_gdf.geometry = new_geoms
    return new_gdf


def ensure_water_fields(water_gdf: gpd.GeoDataFrame):
    """
    Ensure water required fields exist and synthesize missing values.
    Required fields:
    pipeline_id, diameter_mm, material, pressure_zones, flow_rate,
    supply_hours, ward_id, water_usage_per_h3, water_shortage_index
    """
    N = len(water_gdf)
    if N == 0:
        cols = ["pipeline_id", "diameter_mm", "material", "pressure_zones", "flow_rate",
                "supply_hours", "ward_id", "water_usage_per_h3", "water_shortage_index"]
        return gpd.GeoDataFrame(columns=cols + list(water_gdf.columns), geometry="geometry", crs="EPSG:4326")

    field_patterns = {
        "diameter_mm": r"diameter[_\s-]*mm[:\s]*([0-9]+)",
        "diameter_mm2": r"diameter[:\s]*([0-9]+)\s*mm",
        "flow_rate": r"flow[_\s-]*rate[:\s]*([0-9.]+)",
        "ward_id": r"ward[_\s-]*id[:\s]*([A-Za-z0-9_\-]+)",
        "supply_hours": r"supply[_\s-]*hours[:\s]*([0-9\-\s:apmAPM]+)",
    }

    water_gdf = water_gdf.copy()
    if "pipeline_id" not in water_gdf.columns:
        water_gdf["pipeline_id"] = [f"pipe_{i+1}" for i in range(N)]
    else:
        water_gdf["pipeline_id"] = water_gdf["pipeline_id"].fillna("")
        mask_empty = water_gdf["pipeline_id"].astype(str) == ""
        for i in water_gdf[mask_empty].index:
            water_gdf.at[i, "pipeline_id"] = f"pipe_{i+1}"

    for col in ["material", "pressure_zones", "supply_hours", "ward_id"]:
        if col not in water_gdf.columns:
            water_gdf[col] = ""

    for col in ["diameter_mm", "flow_rate", "water_usage_per_h3", "water_shortage_index"]:
        if col not in water_gdf.columns:
            water_gdf[col] = np.nan

    for idx, row in water_gdf.iterrows():
        name = str(row.get("name", "") or "")
        desc = str(row.get("description", "") or "")
        text = name + " " + desc
        parsed = extract_field_from_text(text, field_patterns)
        dia = parsed.get("diameter_mm") or parsed.get("diameter_mm2")
        if dia:
            try:
                water_gdf.at[idx, "diameter_mm"] = float(dia)
            except Exception:
                pass
        fr = parsed.get("flow_rate")
        if fr:
            try:
                water_gdf.at[idx, "flow_rate"] = float(fr)
            except Exception:
                pass
        w = parsed.get("ward_id")
        if w:
            water_gdf.at[idx, "ward_id"] = w
        sh = parsed.get("supply_hours")
        if sh:
            water_gdf.at[idx, "supply_hours"] = sh

    materials = ["DI", "PVC", "CI", "HDPE", "Steel", "GRP"]
    pressure_zones = ["low", "medium", "high"]
    supply_choices = ["24x7", "06:00-10:00, 18:00-22:00", "08:00-20:00", "intermittent"]

    for idx in water_gdf.index:
        if pd.isna(water_gdf.at[idx, "diameter_mm"]) or water_gdf.at[idx, "diameter_mm"] == "":
            water_gdf.at[idx, "diameter_mm"] = float(np.random.choice([50, 75, 100, 150, 200, 250, 300, 400, 600]))
        if not water_gdf.at[idx, "material"]:
            water_gdf.at[idx, "material"] = str(np.random.choice(materials))
        if not water_gdf.at[idx, "pressure_zones"]:
            water_gdf.at[idx, "pressure_zones"] = str(np.random.choice(pressure_zones))
        if pd.isna(water_gdf.at[idx, "flow_rate"]) or water_gdf.at[idx, "flow_rate"] == "":
            d = float(water_gdf.at[idx, "diameter_mm"])
            water_gdf.at[idx, "flow_rate"] = round(0.0005 * (d ** 2) * float(np.random.uniform(0.8, 1.2)), 3)
        if not water_gdf.at[idx, "supply_hours"]:
            water_gdf.at[idx, "supply_hours"] = str(np.random.choice(supply_choices))
        if not water_gdf.at[idx, "ward_id"] or str(water_gdf.at[idx, "ward_id"]).strip() == "":
            water_gdf.at[idx, "ward_id"] = f"ward_{(idx % 50) + 1}"
        if pd.isna(water_gdf.at[idx, "water_usage_per_h3"]):
            water_gdf.at[idx, "water_usage_per_h3"] = round(float(np.random.uniform(0.5, 5.0)), 3)
        if pd.isna(water_gdf.at[idx, "water_shortage_index"]):
            water_gdf.at[idx, "water_shortage_index"] = round(float(np.random.uniform(0.0, 1.0)), 3)

    water_gdf["network_type"] = "water"
    return water_gdf


def read_stp_csv(csv_path: Path):
    """
    Read STP CSV and create point GeoDataFrame if lat/lon available.
    Robustly coerce capacity to numeric (non-numeric -> NaN).
    Expected cols: stp_id, capacity_mld, lat, lon (or latitude/longitude).
    """
    df = pd.read_csv(csv_path, dtype=str)  # read as string so blanks preserved
    lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude", "y")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "x")]
    if lat_cols and lon_cols:
        latc = lat_cols[0]; lonc = lon_cols[0]
        df[latc] = pd.to_numeric(df[latc].astype(str).str.strip(), errors="coerce")
        df[lonc] = pd.to_numeric(df[lonc].astype(str).str.strip(), errors="coerce")
        df = df.dropna(subset=[latc, lonc]).reset_index(drop=True)
        pts = [Point(xy) for xy in zip(df[lonc].astype(float), df[latc].astype(float))]
        gdf = gpd.GeoDataFrame(df, geometry=pts, crs="EPSG:4326")

        if "stp_id" not in gdf.columns:
            gdf["stp_id"] = [f"STP_{i+1}" for i in range(len(gdf))]
        else:
            gdf["stp_id"] = gdf["stp_id"].fillna("").astype(str)
            mask_empty = gdf["stp_id"].str.strip() == ""
            if mask_empty.any():
                replacements = [f"STP_{i+1}" for i in range(mask_empty.sum())]
                gdf.loc[mask_empty, "stp_id"] = replacements

        if "capacity_mld" not in gdf.columns:
            cand = [c for c in gdf.columns if "capac" in c.lower()]
            if cand:
                gdf["capacity_mld"] = pd.to_numeric(gdf[cand[0]].astype(str).str.strip(), errors="coerce")
            else:
                gdf["capacity_mld"] = np.nan
        else:
            gdf["capacity_mld"] = pd.to_numeric(gdf["capacity_mld"].astype(str).str.strip(), errors="coerce")

        return gdf
    else:
        return None


def ensure_sewage_fields(sew_gdf: gpd.GeoDataFrame, stp_gdf: gpd.GeoDataFrame = None):
    """
    Ensure required sewage fields exist and populate from STP CSV if provided.
    Uses metric projection (EPSG:3857) temporarily for nearest-STP distance computations.
    Required fields:
        sewage_line_id, capacity_mld, current_load_mld, overflow_hotspot,
        stp_id, stp_capacity_mld, stp_current_load_mld, sewer_backflow_incidents
    """
    N = len(sew_gdf)
    if N == 0:
        cols = ["sewage_line_id", "capacity_mld", "current_load_mld", "overflow_hotspot",
                "stp_id", "stp_capacity_mld", "stp_current_load_mld", "sewer_backflow_incidents"]
        return gpd.GeoDataFrame(columns=cols + list(sew_gdf.columns), geometry="geometry", crs="EPSG:4326")

    sew_gdf = sew_gdf.copy()
    if "sewage_line_id" not in sew_gdf.columns:
        sew_gdf["sewage_line_id"] = [f"sew_{i+1}" for i in range(N)]
    else:
        sew_gdf["sewage_line_id"] = sew_gdf["sewage_line_id"].fillna("")
        mask_empty = sew_gdf["sewage_line_id"].astype(str) == ""
        for i in sew_gdf[mask_empty].index:
            sew_gdf.at[i, "sewage_line_id"] = f"sew_{i+1}"

    for col in ["capacity_mld", "current_load_mld", "overflow_hotspot", "stp_id",
                "stp_capacity_mld", "stp_current_load_mld", "sewer_backflow_incidents"]:
        if col not in sew_gdf.columns:
            sew_gdf[col] = np.nan if col not in ("overflow_hotspot", "stp_id") else ("" if col == "stp_id" else "no")

    sew_gdf["overflow_hotspot"] = sew_gdf["overflow_hotspot"].fillna("no").replace({None: "no"})

    try:
        sew_metric = sew_gdf.to_crs(epsg=3857)
    except Exception:
        sew_metric = sew_gdf.copy()

    if stp_gdf is not None and len(stp_gdf) > 0:
        try:
            stp_metric = stp_gdf.to_crs(sew_metric.crs)
        except Exception:
            stp_metric = stp_gdf.copy()

        stp_points = stp_metric.geometry.reset_index(drop=True)
        for idx in sew_metric.index:
            centroid = sew_metric.geometry.iloc[idx].centroid
            try:
                dists = stp_points.distance(centroid)
                if len(dists) == 0:
                    continue
                nearest_pos = dists.idxmin()
                nearest_idx = stp_metric.index[nearest_pos]
            except Exception:
                continue

            try:
                stp_cap_raw = stp_gdf.at[nearest_idx, "capacity_mld"] if "capacity_mld" in stp_gdf.columns else None
            except Exception:
                stp_cap_raw = None

            stp_cap = None
            if stp_cap_raw is not None and str(stp_cap_raw).strip() != "":
                try:
                    stp_cap = float(stp_cap_raw)
                except Exception:
                    stp_cap = None

            sew_gdf.at[idx, "stp_id"] = stp_gdf.at[nearest_idx, "stp_id"] if "stp_id" in stp_gdf.columns else str(nearest_idx)
            if stp_cap is not None:
                sew_gdf.at[idx, "stp_capacity_mld"] = stp_cap
                sew_gdf.at[idx, "stp_current_load_mld"] = round(float(sew_gdf.at[idx, "stp_capacity_mld"]) * float(np.random.uniform(0.5, 0.95)), 3)

    for idx in sew_gdf.index:
        if pd.isna(sew_gdf.at[idx, "capacity_mld"]):
            sew_gdf.at[idx, "capacity_mld"] = round(float(np.random.uniform(0.05, 5.0)), 3)
        if pd.isna(sew_gdf.at[idx, "current_load_mld"]):
            cap = float(sew_gdf.at[idx, "capacity_mld"])
            sew_gdf.at[idx, "current_load_mld"] = round(cap * float(np.random.uniform(0.2, 1.05)), 3)
        try:
            sew_gdf.at[idx, "overflow_hotspot"] = "yes" if float(sew_gdf.at[idx, "current_load_mld"]) > float(sew_gdf.at[idx, "capacity_mld"]) else "no"
        except Exception:
            sew_gdf.at[idx, "overflow_hotspot"] = "no"
        if not sew_gdf.at[idx, "stp_id"] or str(sew_gdf.at[idx, "stp_id"]).strip() == "":
            sew_gdf.at[idx, "stp_id"] = f"STP_{(idx % 10) + 1}"
        if pd.isna(sew_gdf.at[idx, "stp_capacity_mld"]):
            sew_gdf.at[idx, "stp_capacity_mld"] = round(float(np.random.uniform(0.5, 20.0)), 3)
        if pd.isna(sew_gdf.at[idx, "stp_current_load_mld"]):
            sc = float(sew_gdf.at[idx, "stp_capacity_mld"])
            sew_gdf.at[idx, "stp_current_load_mld"] = round(sc * float(np.random.uniform(0.1, 0.99)), 3)
        if pd.isna(sew_gdf.at[idx, "sewer_backflow_incidents"]):
            sew_gdf.at[idx, "sewer_backflow_incidents"] = int(np.random.poisson(0.5))

    sew_gdf["network_type"] = "sewage"
    return sew_gdf


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    water_kml = Path(args.water)
    if water_kml.exists():
        print(f"Parsing water KML: {water_kml}")
        water_raw = parse_kml_get_placemarks(water_kml)
    else:
        print(f"Water KML not found at {water_kml}; creating empty water layer.")
        water_raw = gpd.GeoDataFrame(columns=["name", "description", "geometry"], geometry="geometry", crs="EPSG:4326")

    sewage_kml = Path(args.sewage)
    if sewage_kml.exists():
        print(f"Parsing sewage KML: {sewage_kml}")
        sew_raw = parse_kml_get_placemarks(sewage_kml)
    else:
        print(f"Sewage KML not found at {sewage_kml}; creating empty sewage layer.")
        sew_raw = gpd.GeoDataFrame(columns=["name", "description", "geometry"], geometry="geometry", crs="EPSG:4326")

    stp_gdf = None
    if args.stp:
        stp_path = Path(args.stp)
        if stp_path.exists():
            try:
                print(f"Reading STP CSV: {stp_path}")
                stp_gdf = read_stp_csv(stp_path)
                if stp_gdf is None:
                    print("STP CSV read OK but did not contain lat/lon columns for point creation. STP geocoding skipped.")
                else:
                    print(f"Loaded {len(stp_gdf)} STP records.")
            except Exception as e:
                print(f"Failed reading STP CSV: {e}")
                stp_gdf = None
        else:
            print(f"STP CSV not found at {stp_path}; STP assignment will be synthetic.")

    if not water_raw.empty and any(water_raw.geometry.geom_type.isin(["Point", "MultiPoint"])):
        water_raw = convert_points_to_short_lines(water_raw)

    if not sew_raw.empty and any(sew_raw.geometry.geom_type.isin(["Point", "MultiPoint"])):
        sew_raw = convert_points_to_short_lines(sew_raw)

    water_final = ensure_water_fields(water_raw)
    sewage_final = ensure_sewage_fields(sew_raw, stp_gdf)

    try:
        water_final = water_final.to_crs(epsg=4326)
        sewage_final = sewage_final.to_crs(epsg=4326)
    except Exception:
        pass

    water_out = outdir / "water_network.geojson"
    sewage_out = outdir / "sewage_network.geojson"

    water_final.to_file(water_out, driver="GeoJSON")
    sewage_final.to_file(sewage_out, driver="GeoJSON")

    print(f"Written water network GeoJSON: {water_out} ({len(water_final)} features)")
    print(f"Written sewage network GeoJSON: {sewage_out} ({len(sewage_final)} features)")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KML network files to GeoJSON and ensure required attributes (synthetic if missing).")
    parser.add_argument("--water", required=True, help="Path to water_supply.kml")
    parser.add_argument("--sewage", required=True, help="Path to sewage_network.kml")
    parser.add_argument("--stp", required=False, help="Optional path to stp_locations.csv (with lat/lon & capacity_mld)")
    parser.add_argument("--outdir", default=".", help="Output directory for GeoJSON files")
    opts = parser.parse_args()
    main(opts)
