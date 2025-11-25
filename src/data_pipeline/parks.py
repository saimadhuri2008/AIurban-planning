#!/usr/bin/env python3
"""
parks_pipeline.py

Builds parks & green areas dataset using real uploaded sources where available,
and deterministically generates extra parks to ensure plentiful data.

Outputs:
 - parks_h3_res{res}.geojson  (park geometries as points; area_sqkm provided)
 - parks_h3_res{res}.csv
 - parks_wards_res{res}.csv
Defaults:
 - parks KML: /mnt/data/bbmp-parks.kml
 - bbmp public goods CSV: /mnt/data/bbmppublicgoods.csv
 - wards: /mnt/data/wards_master_enriched.geojson
"""
import argparse
from pathlib import Path
import re, math
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
import h3

RNG_SEED = 4242
rng = np.random.RandomState(RNG_SEED)


# ---------- helpers ----------
def parse_kml_points(kml_path: Path):
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
        rows.append({"park_name": name, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def safe_write_geojson(gdf, out_path: Path):
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf)
    geom_cols = [c for c in gdf.columns if gdf[c].dtype == "geometry"]
    for gc in geom_cols:
        if gc != "geometry":
            gdf = gdf.drop(columns=[gc])
    gdf = gdf.set_geometry("geometry")
    gdf.to_file(out_path, driver="GeoJSON")


def estimate_area_from_point(lat, lon):
    center_lat, center_lon = 12.97, 77.59
    dist = math.hypot(lat - center_lat, lon - center_lon)
    base = max(0.002, 0.02 + 0.05 * (0.5 + dist))
    return float(abs(rng.normal(base, base * 0.5)))


def ensure_minimum_count(df, target, generator_fn):
    if len(df) >= target:
        return df
    need = target - len(df)
    synth = generator_fn(need, start_index=len(df))
    return pd.concat([df, synth], ignore_index=True)


# ---------- synthetic parks ----------
def synth_parks(center_lat, center_lon, n, start_index=0):
    rows = []
    for i in range(n):
        idx = start_index + i + 1
        lat = center_lat + rng.normal(0, 0.035)
        lon = center_lon + rng.normal(0, 0.035)
        area = estimate_area_from_point(lat, lon)
        amenities = ",".join(
            rng.choice(
                ["playground","pond","garden","sports","trail","open_lawn"],
                size=rng.randint(1,3)
            ).tolist()
        )
        rows.append({
            "park_id": f"synt_park_{idx}",
            "park_name": f"synt_park_{idx}",
            "area_sqkm": round(area, 5),
            "lat": lat,
            "lon": lon,
            "amenities_available": amenities
        })
    return pd.DataFrame(rows)


# ---------- pipeline ----------
def main(args):
    # paths
    default_parks_kml = Path("/mnt/data/bbmp-parks.kml")
    default_bbmp_csv = Path("/mnt/data/bbmppublicgoods.csv")
    default_wards = Path("/mnt/data/wards_master_enriched.geojson")

    parks_kml = Path(args.parks_kml) if args.parks_kml else default_parks_kml
    bbmp_csv = Path(args.bbmp_csv) if args.bbmp_csv else default_bbmp_csv
    wards_path = Path(args.wards) if args.wards else default_wards

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h3_res = int(args.h3_res)
    target_parks = int(args.target_parks)

    # load wards
    if not wards_path.exists():
        raise FileNotFoundError("Wards geojson not found: " + str(wards_path))

    wards = gpd.read_file(str(wards_path)).to_crs(epsg=4326)
    ward_id_col = next((c for c in wards.columns if c.lower().startswith("ward")), wards.columns[0])
    if ward_id_col != "ward_id":
        wards = wards.rename(columns={ward_id_col: "ward_id"})
    wards["ward_id"] = wards["ward_id"].astype(str)

    parks = []

    # parse KML parks
    if parks_kml.exists():
        try:
            pk = parse_kml_points(parks_kml)
            for i, row in pk.iterrows():
                area = estimate_area_from_point(row["lat"], row["lon"])
                parks.append({
                    "park_id": f"kmlpark_{i+1}",
                    "park_name": row.get("park_name", f"kmlpark_{i+1}"),
                    "area_sqkm": round(area, 5),
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "amenities_available": ""
                })
            print(f"Parsed {len(pk)} parks from KML: {parks_kml}")
        except Exception as e:
            print("Parks KML parse error:", e)

    # parse BBMP CSV
    if bbmp_csv.exists():
        try:
            bdf = pd.read_csv(bbmp_csv, dtype=str)
            latcol = next((c for c in bdf.columns if "lat" in c.lower()), None)
            loncol = next((c for c in bdf.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            namecol = next((c for c in bdf.columns if re.search(r"name|title", c, flags=re.I)), None)
            area_col = next((c for c in bdf.columns if "area" in c.lower()), None)
            amen_col = next((c for c in bdf.columns if "amen" in c.lower() or "facility" in c.lower()), None)

            for i, row in bdf.iterrows():
                if latcol and loncol and pd.notna(row.get(latcol)) and pd.notna(row.get(loncol)):
                    try:
                        lat = float(row.get(latcol)); lon = float(row.get(loncol))
                    except:
                        continue

                    name = str(row.get(namecol, f"bbmp_{i}"))
                    area = float(row.get(area_col, estimate_area_from_point(lat, lon))) if area_col else estimate_area_from_point(lat, lon)
                    amenities = str(row.get(amen_col, ""))

                    parks.append({
                        "park_id": name,
                        "park_name": name,
                        "area_sqkm": round(float(area), 5),
                        "lat": lat,
                        "lon": lon,
                        "amenities_available": amenities
                    })

            print(f"Loaded {len(parks)} parks (including CSV entries).")

        except Exception as e:
            print("BBMP CSV parse error:", e)

    parks_df = pd.DataFrame(parks)

    # fallback synthetic core
    if parks_df.empty:
        print("No real parks found — generating synthetic core set")
        parks_df = synth_parks(12.97, 77.59, n=300, start_index=0)

    # ---------- FIXED union_all + centroid ----------
    center_union = wards.union_all()
    center_centroid = center_union.centroid

    # ensure minimum count
    parks_df = ensure_minimum_count(
        parks_df,
        target_parks,
        lambda n, start_index: synth_parks(center_centroid.y, center_centroid.x, n, start_index)
    )

    # Create GeoDataFrame
    parks_gdf = gpd.GeoDataFrame(
        parks_df,
        geometry=[Point(xy) for xy in zip(parks_df["lon"].astype(float), parks_df["lat"].astype(float))],
        crs="EPSG:4326"
    )

    # H3 index
    parks_gdf["h3_index"] = parks_gdf.geometry.apply(
        lambda g: h3.geo_to_h3(g.y, g.x, h3_res) if g is not None else None
    )

    # spatial join (within)
    joined = gpd.sjoin(
        parks_gdf.set_geometry("geometry"),
        wards[["ward_id", "geometry"]],
        how="left",
        predicate="within"
    )

    # nearest-ward fix for unmatched
    unmatched = joined[joined["ward_id"].isna()]
    if len(unmatched) > 0:
        ward_centroids = wards[['ward_id', 'geometry']].copy()
        ward_centroids["centroid"] = ward_centroids.geometry.to_crs(3857).centroid
        ward_centroids = ward_centroids.set_geometry("centroid").to_crs(4326)

        for idx, r in unmatched.iterrows():
            # distance in projected CRS
            ward_pts = ward_centroids.to_crs(3857)
            park_pt = gpd.GeoSeries([r.geometry], crs="EPSG:4326").to_crs(3857).iloc[0]

            distances = ward_pts.distance(park_pt)
            nearest = distances.idxmin()
            joined.at[idx, "ward_id"] = ward_centroids.at[nearest, "ward_id"]

    # outputs
    out_geo = Path(args.outdir) / f"parks_h3_res{h3_res}.geojson"
    out_csv = Path(args.outdir) / f"parks_h3_res{h3_res}.csv"
    wards_csv = Path(args.outdir) / f"parks_wards_res{h3_res}.csv"

    safe_write_geojson(joined, out_geo)
    joined.drop(columns=["geometry"]).to_csv(out_csv, index=False)

    # ward aggregates
    ward_agg = (
        joined.dropna(subset=["ward_id"])
        .groupby("ward_id")
        .agg({"park_id": "count", "area_sqkm": "sum"})
        .reset_index()
        .rename(columns={
            "park_id": "park_count",
            "area_sqkm": "park_area_sqkm"
        })
    )
    ward_agg.to_csv(wards_csv, index=False)

    print("Saved:", out_geo, out_csv, wards_csv)
    print("Total parks generated/output:", len(joined))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parks_kml", default=r"C:\Users\jbhuv\Downloads\bbmp-parks.kml")
    p.add_argument("--bbmp_csv", default=r"C:\Users\jbhuv\Downloads\bbmppublicgoods.csv")
    p.add_argument("--wards", default=r"C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
    p.add_argument("--h3_res", default=8, type=int)
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--target_parks", default=800, type=int)
    args = p.parse_args()
    main(args)
