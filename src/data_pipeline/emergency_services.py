#!/usr/bin/env python3
"""
emergency_services.py

Generates Fire Stations, Police Stations & Ambulance Bases dataset.
Uses real uploaded sources where available, and generates synthetic
records when needed (deterministic).

Outputs:
 - emergency_services_h3.geojson
 - emergency_services_h3.csv
 - emergency_services_wards.csv
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import h3

RNG = np.random.RandomState(99)


# ---------------------------------------------------------
# Helper: KML Point Reader
# ---------------------------------------------------------
def parse_kml_points(kml_path: Path):
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    placemarks = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL | re.IGNORECASE)
    rows = []
    for pm in placemarks:
        nm = re.search(r"<name>(.*?)</name>", pm, flags=re.I)
        name = nm.group(1).strip() if nm else ""
        coord = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.I)
        if coord:
            first = coord.group(1).strip().split()[0]
            parts = first.split(",")
            try:
                lon = float(parts[0]); lat = float(parts[1])
            except:
                continue
            rows.append({"name": name, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Helper: Safe GeoJSON Writer
# ---------------------------------------------------------
def safe_write_geojson(gdf, out_path: Path):
    geom_cols = [c for c in gdf.columns if gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            gdf = gdf.drop(columns=[gc])
    gdf.set_geometry("geometry").to_file(out_path, driver="GeoJSON")


# ---------------------------------------------------------
# Synthetic Data
# ---------------------------------------------------------
def synth_points(center_lat, center_lon, n, prefix):
    rows = []
    for i in range(n):
        lat = center_lat + RNG.normal(0, 0.04)
        lon = center_lon + RNG.normal(0, 0.04)
        rows.append({
            "name": f"{prefix}_{i+1}",
            "lat": lat,
            "lon": lon
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):

    # INPUTS
    fire_kml = Path(args.fire_kml)
    police_csv = Path(args.police_csv)
    ambulance_csv = Path(args.ambulance_csv)
    wards_path = Path(args.wards)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h3_res = int(args.h3_res)
    target_fire = int(args.target_fire)
    target_police = int(args.target_police)
    target_ambul = int(args.target_ambul)

    # LOAD WARDS
    wards = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
    ward_id_col = next((c for c in wards.columns if c.lower().startswith("ward")), wards.columns[0])
    if ward_id_col != "ward_id":
        wards = wards.rename(columns={ward_id_col: "ward_id"})
    wards["ward_id"] = wards["ward_id"].astype(str)

    # centroid to guide synthetic generation
    city_centroid = wards.unary_union.centroid

    # ---------------------------------------------------------
    # FIRE STATIONS
    # ---------------------------------------------------------
    fire_df = pd.DataFrame()

    if fire_kml.exists():
        real_fire = parse_kml_points(fire_kml)
        real_fire["src"] = "real"
        fire_df = pd.concat([fire_df, real_fire], ignore_index=True)
        print(f"Loaded {len(real_fire)} fire stations from KML.")

    if len(fire_df) < target_fire:
        shortage = target_fire - len(fire_df)
        synth = synth_points(city_centroid.y, city_centroid.x, shortage, prefix="synt_fire")
        synth["src"] = "synthetic"
        fire_df = pd.concat([fire_df, synth], ignore_index=True)

    fire_df["type"] = "fire_station"


    # ---------------------------------------------------------
    # POLICE STATIONS
    # ---------------------------------------------------------
    police_df = pd.DataFrame()

    if police_csv.exists():
        try:
            df = pd.read_csv(police_csv)
            latcol = next((c for c in df.columns if "lat" in c.lower()), None)
            loncol = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            namecol = next((c for c in df.columns if re.search("name|station", c, re.I)), None)

            for i, r in df.iterrows():
                if pd.notna(r.get(latcol)) and pd.notna(r.get(loncol)):
                    police_df = pd.concat([
                        police_df,
                        pd.DataFrame([{
                            "name": r.get(namecol, f"police_{i+1}"),
                            "lat": float(r[latcol]),
                            "lon": float(r[loncol])
                        }])
                    ], ignore_index=True)
            print(f"Loaded {len(police_df)} police stations from CSV.")
        except:
            pass

    if len(police_df) < target_police:
        shortage = target_police - len(police_df)
        synth = synth_points(city_centroid.y, city_centroid.x, shortage, prefix="synt_pol")
        police_df = pd.concat([police_df, synth], ignore_index=True)

    police_df["type"] = "police_station"


    # ---------------------------------------------------------
    # AMBULANCE BASES
    # ---------------------------------------------------------
    ambul_df = pd.DataFrame()

    if ambulance_csv.exists():
        try:
            df = pd.read_csv(ambulance_csv)
            latcol = next((c for c in df.columns if "lat" in c.lower()), None)
            loncol = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            namecol = next((c for c in df.columns if re.search("name|base", c, re.I)), None)

            for i, r in df.iterrows():
                if pd.notna(r.get(latcol)) and pd.notna(r.get(loncol)):
                    ambul_df = pd.concat([
                        ambul_df,
                        pd.DataFrame([{
                            "name": r.get(namecol, f"ambul_{i+1}"),
                            "lat": float(r[latcol]),
                            "lon": float(r[loncol])
                        }])
                    ], ignore_index=True)
            print(f"Loaded {len(ambul_df)} ambulance bases from CSV.")
        except:
            pass

    if len(ambul_df) < target_ambul:
        shortage = target_ambul - len(ambul_df)
        synth = synth_points(city_centroid.y, city_centroid.x, shortage, prefix="synt_ambul")
        ambul_df = pd.concat([ambul_df, synth], ignore_index=True)

    ambul_df["type"] = "ambulance_base"

    # ---------------------------------------------------------
    # MERGE ALL SERVICES
    # ---------------------------------------------------------
    all_df = pd.concat([fire_df, police_df, ambul_df], ignore_index=True)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        all_df,
        geometry=[Point(xy) for xy in zip(all_df["lon"], all_df["lat"])],
        crs="EPSG:4326"
    )

    # H3 index
    gdf["h3_index"] = gdf.geometry.apply(lambda g: h3.geo_to_h3(g.y, g.x, h3_res))

    # Ward join
    joined = gpd.sjoin(gdf, wards[["ward_id", "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")

    # Response time synthetic field (per H3)
    joined["response_time_h3"] = np.round(np.clip(RNG.normal(8, 3, len(joined)), 3, 25), 2)

    # ---------------------------------------------------------
    # OUTPUTS
    # ---------------------------------------------------------
    out_geo = outdir / "emergency_services_h3.geojson"
    out_csv = outdir / "emergency_services_h3.csv"
    wards_csv = outdir / "emergency_services_wards.csv"

    safe_write_geojson(joined, out_geo)
    joined.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    # Ward-level aggregates
    ward_agg = joined.groupby("ward_id").agg({
        "name": "count",
        "response_time_h3": "mean"
    }).reset_index().rename(columns={"name": "service_count"})
    ward_agg.to_csv(wards_csv, index=False)

    print("Saved:", out_geo, out_csv, wards_csv)
    print("Total emergency service points:", len(joined))


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fire_kml", default="C:\\Users\\jbhuv\\Downloads\\bengaluru_fire_stations.kml")
    parser.add_argument("--police_csv", default="C:\\Users\\jbhuv\\Downloads\\bbmppublicgoods.csv")
    parser.add_argument("--ambulance_csv", default="C:\\Users\\jbhuv\\Downloads\\fire_services.csv")
    parser.add_argument("--wards", default="C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")

    parser.add_argument("--h3_res", default=8, type=int)
    parser.add_argument("--target_fire", default=150, type=int)
    parser.add_argument("--target_police", default=200, type=int)
    parser.add_argument("--target_ambul", default=100, type=int)

    parser.add_argument("--outdir", default="./outputs")

    args = parser.parse_args()
    main(args)
