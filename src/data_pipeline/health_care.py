#!/usr/bin/env python3
"""
healthcare_pipeline.py

Produces hospital/healthcentre GeoJSON + CSV with:
 - hospital_id
 - hospital_type (govt/private)
 - beds
 - doctors
 - specialties (comma-separated)
 - lat, lon
 - catchment_area_h3 (list of h3 cells within catchment radius)

Defaults (will use these uploaded files if present):
 - health CSV: /mnt/data/health_data.csv
 - health centres KML: /mnt/data/health_centres.kml
 - wards geojson: /mnt/data/wards_master_enriched.geojson

Output:
 - health_h3.geojson, health_h3.csv
 - health_wards.csv
"""
import argparse
from pathlib import Path
import math, json
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, mapping
import re
import h3

RNG_SEED = 11
rng = np.random.RandomState(RNG_SEED)

def parse_kml_points(kml_path: Path):
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    placemarks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL|re.IGNORECASE)
    rows=[]
    for pm in placemarks:
        name_m = re.search(r"<name>(.*?)</name>", pm, flags=re.IGNORECASE|re.DOTALL)
        name = name_m.group(1).strip() if name_m else ""
        coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.IGNORECASE|re.DOTALL)
        if not coords_m:
            continue
        first = coords_m.group(1).strip().split()[0]
        parts = first.split(",")
        try:
            lon=float(parts[0]); lat=float(parts[1])
        except:
            continue
        rows.append({"name": name, "lat": lat, "lon": lon})
    if not rows:
        return pd.DataFrame(columns=["name","lat","lon"])
    return pd.DataFrame(rows)

def compute_catchment_h3(lat, lon, res, radius_km=5):
    # returns h3.k_ring of center h up to ring estimated by sampling approx distance per ring
    center = h3.geo_to_h3(lat, lon, res)
    # choose k such that k_ring distance roughly >= radius_km
    # approximate cell edge length (meters) ~ h3.edge_length(res, unit='m') not available; use heuristic rings
    # we will use k_ring radius of int(radius_km*1.2) as safe overestimate
    k = max(1, int(radius_km))  # simple
    return list(h3.k_ring(center, k))

def safe_write_geojson(gdf, out_path: Path):
    # drop extra geometry-like columns and ensure single geometry
    geom_cols = [c for c in gdf.columns if gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            gdf = gdf.drop(columns=[gc])
    gdf = gdf.set_geometry("geometry")
    gdf.to_file(out_path, driver="GeoJSON")

def main(args):
    health_csv = Path(args.health_csv) if args.health_csv else Path("/mnt/data/health_data.csv")
    health_kml = Path(args.health_kml) if args.health_kml else Path("/mnt/data/health_centres.kml")
    wards_path = Path(args.wards) if args.wards else Path("/mnt/data/wards_master_enriched.geojson")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    h3_res = int(args.h3_res)

    stations = []
    # Try KML first
    if health_kml.exists():
        try:
            df = parse_kml_points(health_kml)
            for i,row in df.iterrows():
                stations.append({
                    "hospital_id": f"kml_{i+1}",
                    "hospital_type": "", "beds": np.nan, "doctors": np.nan,
                    "specialties": "", "lat": row["lat"], "lon": row["lon"]
                })
            print(f"Parsed {len(df)} health points from KML: {health_kml}")
        except Exception as e:
            print("KML parse error:", e)

    # Try CSV (more attributes)
    if health_csv.exists():
        try:
            hdf = pd.read_csv(health_csv)
            # try detect lat/lon columns
            latcol = next((c for c in hdf.columns if c.lower() in ("lat","latitude","y")), None)
            loncol = next((c for c in hdf.columns if c.lower() in ("lon","lng","longitude","x")), None)
            idcol  = next((c for c in hdf.columns if "hospital" in c.lower() or "health" in c.lower() or "id"==c.lower()), None)
            # iterate rows
            for i,row in hdf.iterrows():
                lat = float(row[latcol]) if latcol and pd.notna(row.get(latcol)) else None
                lon = float(row[loncol]) if loncol and pd.notna(row.get(loncol)) else None
                if lat is None or lon is None:
                    # skip rows without coords here (we may use KML)
                    continue
                # try to find beds/doctors/specialty columns heuristically
                beds_col = next((c for c in hdf.columns if "bed" in c.lower()), None)
                docs_col = next((c for c in hdf.columns if "doc" in c.lower()), None)
                spec_col = next((c for c in hdf.columns if "special" in c.lower()), None)
                type_col = next((c for c in hdf.columns if "type" in c.lower()), None)
                stations.append({
                    "hospital_id": str(row[idcol]) if idcol else f"csv_{i+1}",
                    "hospital_type": str(row[type_col]) if type_col and pd.notna(row.get(type_col)) else "",
                    "beds": float(row[beds_col]) if beds_col and pd.notna(row.get(beds_col)) else np.nan,
                    "doctors": float(row[docs_col]) if docs_col and pd.notna(row.get(docs_col)) else np.nan,
                    "specialties": str(row[spec_col]) if spec_col and pd.notna(row.get(spec_col)) else "",
                    "lat": lat, "lon": lon
                })
            print(f"Loaded {len(stations)} health stations (including KML/CSV).")
        except Exception as e:
            print("Health CSV read error:", e)

    # If nothing found, generate synthetic hospitals
    if len(stations) == 0:
        print("No real health data found — generating synthetic hospitals.")
        # generate N synthetic hospitals around city center
        center_lat, center_lon = 12.97, 77.59
        N = int(args.synthetic_count)
        for i in range(N):
            # jitter in degrees (~ km scale)
            jitter_lat = rng.normal(0, 0.03)
            jitter_lon = rng.normal(0, 0.03)
            lat = center_lat + jitter_lat
            lon = center_lon + jitter_lon
            typ = "govt" if (i % 3 == 0) else "private"
            beds = int(max(5, rng.poisson(50) + (0 if typ=="govt" else 20)))
            docs = int(max(1, int(beds/5) + rng.randint(-2,5)))
            specs = ",".join(rng.choice(["general","pediatrics","cardiology","orthopedics","maternity","oncology"], size=rng.randint(1,3)).tolist())
            stations.append({"hospital_id": f"synt_{i+1}", "hospital_type": typ, "beds": beds, "doctors": docs, "specialties": specs, "lat": lat, "lon": lon})

    # Build GeoDataFrame
    df = pd.DataFrame(stations)
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")

    # compute catchment area (h3 list) for each hospital
    def catchment_list(row):
        try:
            return compute_catchment_h3(row["lat"], row["lon"], h3_res, radius_km=float(args.catchment_km))
        except:
            return []
    gdf["catchment_area_h3"] = gdf.apply(catchment_list, axis=1)

    # attach ward by spatial join using centroid
    wards_gdf = gpd.read_file(str(wards_path)) if Path(wards_path).exists() else None
    if wards_gdf is not None:
        wards_gdf = wards_gdf.to_crs(epsg=4326)
        # create centroids for join
        pts = gdf.copy()
        pts = pts.set_geometry("geometry")
        joined = gpd.sjoin(pts, wards_gdf[[wards_gdf.columns[0],"geometry"]], how="left", predicate="within")
        ward_col_candidates = [c for c in joined.columns if "ward" in c.lower()]
        if len(ward_col_candidates)>0:
            ward_key = ward_col_candidates[0]
        else:
            ward_key = wards_gdf.columns[0]
        joined = joined.rename(columns={ward_key: "ward_id"}) if ward_key!="ward_id" else joined
        # keep only needed columns and ensure geometry single
        out = joined[["hospital_id","hospital_type","beds","doctors","specialties","lat","lon","catchment_area_h3","geometry","ward_id"]].copy()
    else:
        gdf["ward_id"] = None
        out = gdf[["hospital_id","hospital_type","beds","doctors","specialties","lat","lon","catchment_area_h3","geometry"]].copy()
        out["ward_id"] = None

    # write outputs
    out_geo = Path(outdir)/f"health_h3_res{h3_res}.geojson"
    out_csv = Path(outdir)/f"health_h3_res{h3_res}.csv"
    try:
        safe_write_geojson(out, out_geo)
    except Exception as e:
        # fallback: simple JSON writer
        out.to_file(out_geo, driver="GeoJSON")
    # CSV (include lat/lon)
    out["lon"] = out.geometry.x
    out["lat"] = out.geometry.y
    out.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    # ward-level aggregation: count hospitals per ward and sum beds
    if wards_gdf is not None:
        wards_agg = out.dropna(subset=["ward_id"]).groupby("ward_id").agg({"hospital_id":"count","beds":"sum","doctors":"sum"}).reset_index().rename(columns={"hospital_id":"hospital_count"})
        wards_out = Path(outdir)/f"health_wards_res{h3_res}.csv"
        wards_agg.to_csv(wards_out, index=False)
        print("Wrote:", out_geo, out_csv, wards_out)
    else:
        print("Wrote:", out_geo, out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--health_csv", default="C:\\Users\\jbhuv\\Downloads\\health_data.csv")
    p.add_argument("--health_kml", default="C:\\Users\\jbhuv\\Downloads\\health_centres.kml")
    p.add_argument("--wards", default="C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
    p.add_argument("--h3_res", type=int, default=8)
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--catchment_km", type=float, default=5.0)
    p.add_argument("--synthetic_count", type=int, default=100)
    args = p.parse_args()
    main(args)
