#!/usr/bin/env python3
"""
education_pipeline.py

Builds a large education dataset using real uploaded files where available,
and deterministically generates extra records when needed.

Outputs:
 - schools_h3_res{res}.geojson  (geometry is point)
 - schools_h3_res{res}.csv
 - schools_wards_res{res}.csv   (ward aggregates)

Defaults (use your uploaded files if present):
 - /mnt/data/number-of-schools-by-ward.csv
 - /mnt/data/educaton.csv
 - /mnt/data/blr_engg_colleges.kml
 - /mnt/data/wards_master_enriched.geojson
"""
import argparse
from pathlib import Path
import re
import json
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import h3

RNG_SEED = 20250401
rng = np.random.RandomState(RNG_SEED)


# ---------- helpers ----------
def parse_kml_points(kml_path: Path):
    """Lightweight KML Placemark point parser; returns DataFrame with name, lat, lon."""
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    placemarks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL | re.IGNORECASE)
    rows = []
    for pm in placemarks:
        nm = re.search(r"<name>(.*?)</name>", pm, flags=re.DOTALL | re.IGNORECASE)
        name = nm.group(1).strip() if nm else ""
        coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.DOTALL | re.IGNORECASE)
        if not coords_m:
            continue
        coords_txt = coords_m.group(1).strip()
        first = coords_txt.split()[0]
        parts = first.split(",")
        try:
            lon = float(parts[0]); lat = float(parts[1])
        except Exception:
            continue
        rows.append({"name": name, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def safe_write_geojson(gdf, out_path: Path):
    """Ensure exactly one geometry column then write GeoJSON."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf)
    geom_cols = [c for c in gdf.columns if gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            gdf = gdf.drop(columns=[gc])
    gdf = gdf.set_geometry("geometry")
    gdf.to_file(out_path, driver="GeoJSON")


def detect_latlon(df):
    lat = next((c for c in df.columns if c.lower() in ("lat","latitude","y","latitude_deg","lat_deg")), None)
    lon = next((c for c in df.columns if c.lower() in ("lon","lng","longitude","x","lon_deg","long")), None)
    return lat, lon


def ensure_minimum_count(df, target_count, generator_fn):
    """If df has fewer than target_count rows, call generator_fn(n_needed) to append synthetic rows."""
    current = len(df)
    if current >= target_count:
        return df
    need = int(target_count - current)
    synth = generator_fn(need, start_index=current)
    return pd.concat([df, synth], ignore_index=True)


# ---------- synthetic generators ----------
def synth_schools_near(centroid_lat, centroid_lon, n, start_index=0):
    rows = []
    for i in range(n):
        idx = start_index + i + 1
        # jitter ~ few km
        lat = centroid_lat + rng.normal(0, 0.03)
        lon = centroid_lon + rng.normal(0, 0.03)
        school_type = rng.choice(["govt","private","aided"], p=[0.25, 0.65, 0.10])
        students = int(max(20, rng.normal(350 if school_type!="govt" else 200, 150)))
        sid = f"synt_school_{idx}"
        rows.append({"school_id": sid, "school_name": sid, "school_type": school_type, "students": students, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


# ---------- main pipeline ----------
def main(args):
    # defaults to uploaded files
    default_schools_csv = Path("/mnt/data/number-of-schools-by-ward.csv")
    default_education_csv = Path("/mnt/data/educaton.csv")
    default_colleges_kml = Path("/mnt/data/blr_engg_colleges.kml")
    default_wards = Path("/mnt/data/wards_master_enriched.geojson")

    schools_csv = Path(args.schools_csv) if args.schools_csv else default_schools_csv
    education_csv = Path(args.education_csv) if args.education_csv else default_education_csv
    colleges_kml = Path(args.colleges_kml) if args.colleges_kml else default_colleges_kml
    wards_path = Path(args.wards) if args.wards else default_wards
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    h3_res = int(args.h3_res)
    target_school_count = int(args.target_school_count)

    # 1) read wards (for linking)
    if not wards_path.exists():
        raise FileNotFoundError(f"Wards GeoJSON not found: {wards_path}")
    wards = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
    ward_id_col = next((c for c in wards.columns if c.lower().startswith("ward")), wards.columns[0])
    wards = wards.rename(columns={ward_id_col: "ward_id"}) if ward_id_col != "ward_id" else wards
    wards["ward_id"] = wards["ward_id"].astype(str)

    # 2) collect school records from CSVs/KML
    schools = []

    # Try education CSV(s)
    if education_csv.exists():
        try:
            ed = pd.read_csv(education_csv, dtype=str)
            latcol, loncol = detect_latlon(ed)
            idcol = next((c for c in ed.columns if re.search(r"school|institution|name|id", c, flags=re.I)), None)
            students_col = next((c for c in ed.columns if re.search(r"student|enrol|strength", c, flags=re.I)), None)
            type_col = next((c for c in ed.columns if re.search(r"type|management", c, flags=re.I)), None)
            for i,row in ed.iterrows():
                lat = float(row[latcol]) if latcol and pd.notna(row.get(latcol)) else None
                lon = float(row[loncol]) if loncol and pd.notna(row.get(loncol)) else None
                if lat is None or lon is None:
                    continue
                sid = str(row[idcol]) if idcol and pd.notna(row.get(idcol)) else f"edu_csv_{i+1}"
                st = int(row[students_col]) if students_col and pd.notna(row.get(students_col)) else int(rng.uniform(50, 1000))
                s_type = str(row[type_col]) if type_col and pd.notna(row.get(type_col)) else ("govt" if i % 5 == 0 else "private")
                schools.append({"school_id": sid, "school_name": str(row.get(idcol, sid)), "school_type": s_type, "students": st, "lat": lat, "lon": lon})
            print(f"Loaded {len(schools)} schools from {education_csv}")
        except Exception as e:
            print("Education CSV parse error:", e)

    # Try 'number-of-schools-by-ward' CSV: if it contains only counts by ward, we'll use it to seed aggregated counts later
    schools_by_ward_df = None
    if schools_csv.exists():
        try:
            schools_by_ward_df = pd.read_csv(schools_csv, dtype=str)
            print("Found schools-by-ward CSV:", schools_csv)
        except Exception as e:
            print("Schools-by-ward CSV parse error:", e)

    # Try colleges KML (higher-ed)
    if colleges_kml.exists():
        try:
            colleges = parse_kml_points(colleges_kml)
            for i,row in colleges.iterrows():
                sid = f"college_kml_{i+1}"
                schools.append({"school_id": sid, "school_name": row.get("name", sid), "school_type": "college", "students": int(rng.uniform(300,3000)), "lat": row["lat"], "lon": row["lon"]})
            print(f"Parsed {len(colleges)} colleges from {colleges_kml}")
        except Exception as e:
            print("Colleges KML parse error:", e)

    # Build DataFrame
    schools_df = pd.DataFrame(schools)
    # if none found from real data, synthesize a base set
    if len(schools_df) == 0:
        print("No real school points found — generating synthetic core set")
        center_lat, center_lon = 12.97, 77.59
        schools_df = synth_schools_near(center_lat, center_lon, n=1500, start_index=0)

    # Ensure sufficient quantity for big project: target_school_count
    # Use centroid of wards to guide synthetic additions
    centroid = wards.unary_union.centroid
    schools_df = ensure_minimum_count(schools_df, target_school_count, lambda n, start_index: synth_schools_near(centroid.y, centroid.x, n, start_index))

    # convert to GeoDataFrame
    schools_gdf = gpd.GeoDataFrame(schools_df, geometry=[Point(xy) for xy in zip(schools_df["lon"].astype(float), schools_df["lat"].astype(float))], crs="EPSG:4326")

    # compute h3 index per school
    schools_gdf["h3_index"] = schools_gdf.geometry.apply(lambda g: h3.geo_to_h3(g.y, g.x, h3_res) if g is not None else None)

    # attach ward via spatial join (centroid join)
    pts = schools_gdf.set_geometry("geometry")
    joined = gpd.sjoin(pts, wards[["ward_id","geometry"]], how="left", predicate="within")
    # if some schools not matched (outside ward polygons), try nearest by centroid
    unmatched = joined[joined["ward_id"].isna()]
    if len(unmatched) > 0:
        # find nearest ward for unmatched points
        ward_centroids = wards.copy()
        ward_centroids["centroid"] = ward_centroids.geometry.centroid
        ward_centroids = gpd.GeoDataFrame(ward_centroids.drop(columns=["geometry"]), geometry="centroid", crs="EPSG:4326")
        for idx, r in unmatched.iterrows():
            distances = ward_centroids.geometry.distance(r.geometry)
            nearest = distances.idxmin()
            joined.at[idx, "ward_id"] = ward_centroids.at[nearest, "ward_id"]

    # Final outputs
    out_geo = Path(args.outdir) / f"schools_h3_res{h3_res}.geojson"
    out_csv = Path(args.outdir) / f"schools_h3_res{h3_res}.csv"
    wards_csv = Path(args.outdir) / f"schools_wards_res{h3_res}.csv"

    safe_write_geojson(joined, out_geo)
    joined.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    # Ward aggregates: school_count, total_students
    ward_agg = joined.dropna(subset=["ward_id"]).groupby("ward_id").agg({"school_id":"count", "students":"sum"}).reset_index().rename(columns={"school_id":"school_count", "students":"students_total"})
    # If we had a schools-by-ward CSV with counts, keep both columns for comparison
    if schools_by_ward_df is not None:
        ward_agg.to_csv(wards_csv, index=False)
    else:
        ward_agg.to_csv(wards_csv, index=False)

    print("Saved:", out_geo, out_csv, wards_csv)
    print("Total schools generated/output:", len(joined))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--schools_csv", default="C:\\Users\\jbhuv\\Downloads\\number-of-schools-by-ward.csv", help="CSV with ward-level school counts (optional)")
    p.add_argument("--education_csv", default="C:\\Users\\jbhuv\\Downloads\\educaton.csv", help="Education CSV with school points if present")
    p.add_argument("--colleges_kml", default="C:\\Users\\jbhuv\\Downloads\\blr_engg_colleges.kml", help="KML of colleges (optional)")
    p.add_argument("--wards", default="C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
    p.add_argument("--h3_res", default=8, type=int)
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--target_school_count", default=5000, type=int, help="Desired minimum number of school records (will generate synthetic if fewer)")
    args = p.parse_args()
    main(args)
