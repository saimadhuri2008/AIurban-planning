#!/usr/bin/env python3
"""
temporal_full_pipeline.py

Corrected pipeline that assigns ward_id robustly by:
 1) polygon intersection (H3 polygon intersects ward polygon)
 2) fallback: nearest ward by centroid distance using spatial index

Other behavior:
 - generates temporal series (synthetic if needed)
 - writes daily parquet + validated combined parquet (streamed if pyarrow available)
 - writes h3_cells.geojson and temporal_latest_h3.geojson
"""

import os, sys, json, math, argparse, logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import shapely
from tqdm import tqdm

# h3 and pyarrow optional
try:
    import h3
except Exception:
    h3 = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None; pq = None

# deterministic RNG
RNG_SEED = 20251119
rng = np.random.RandomState(RNG_SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("temporal_full_pipeline")

# ----------------- user default paths (windows) -----------------
DEFAULT_WARDS = Path(r"C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
DEFAULT_POLLUTION_KML = Path(r"C:\Users\jbhuv\Downloads\pollution_monitors.kml")
DEFAULT_POLLUTION_CSV = Path(r"C:\Users\jbhuv\Downloads\air_quality.csv")
DEFAULT_WATER_KML = Path(r"C:\Users\jbhuv\Downloads\water_supply.kml")
DEFAULT_SEWAGE_KML = Path(r"C:\Users\jbhuv\Downloads\sewage_network.kml")
DEFAULT_STP_CSV = Path(r"C:\Users\jbhuv\Downloads\stp_locations.csv")
DEFAULT_HEALTH_KML = Path(r"C:\Users\jbhuv\Downloads\health_centres.kml")
DEFAULT_SCHOOLS_CSV = Path(r"C:\Users\jbhuv\Downloads\number-of-schools-by-ward.csv")
DEFAULT_PARKS_KML = Path(r"C:\Users\jbhuv\Downloads\bbmp-parks.kml")
DEFAULT_FIRE_KML = Path(r"C:\Users\jbhuv\Downloads\bengaluru_fire_stations.kml")

# ----------------- helpers -----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def safe_write_geojson(gdf: gpd.GeoDataFrame, path: Path):
    if gdf is None or gdf.empty:
        with open(path, "w") as f:
            json.dump({"type":"FeatureCollection","features":[]}, f)
        log.warning("Wrote empty GeoJSON: %s", path)
        return
    if "geometry" not in gdf.columns:
        raise ValueError("No geometry column to write to GeoJSON")
    g = gdf.set_geometry("geometry").to_crs(epsg=4326)
    # drop extra geometry dtype columns if any
    geom_cols = [c for c in g.columns if getattr(g[c], "dtype", None) == "geometry" and c != "geometry"]
    if geom_cols:
        g = g.drop(columns=geom_cols)
    g.to_file(str(path), driver="GeoJSON")
    log.info("WROTE GeoJSON: %s (features=%d)", path, len(g))

def parse_kml_points(kml_path: Path):
    if not kml_path.exists():
        return pd.DataFrame(columns=["name","lat","lon"])
    txt = kml_path.read_text(encoding="utf-8", errors="replace")
    import re
    pms = re.findall(r"<Placemark[^>]*>(.*?)</Placemark>", txt, flags=re.DOTALL|re.IGNORECASE)
    rows=[]
    for pm in pms:
        name_m = re.search(r"<name>(.*?)</name>", pm, flags=re.DOTALL|re.IGNORECASE)
        coords_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, flags=re.DOTALL|re.IGNORECASE)
        if not coords_m:
            continue
        try:
            first = coords_m.group(1).strip().split()[0]
            lon, lat = [float(x) for x in first.split(",")[:2]]
            name = name_m.group(1).strip() if name_m else ""
            rows.append({"name":name, "lat":lat, "lon":lon})
        except Exception:
            continue
    return pd.DataFrame(rows)

def h3_boundary_poly(h):
    b = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon([(pt[1], pt[0]) for pt in b])  # (lon,lat)

def haversine_km(lon1, lat1, lon2, lat2):
    R=6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

# ----------------- load wards & seeds -----------------
def load_wards(wards_path: Path) -> gpd.GeoDataFrame:
    if not wards_path.exists():
        raise FileNotFoundError(f"Wards not found: {wards_path}")
    wg = gpd.read_file(str(wards_path))
    try:
        wg = wg.to_crs(epsg=4326)
    except Exception:
        pass
    ward_col = next((c for c in wg.columns if c.lower().startswith("ward")), None)
    if ward_col is None:
        ward_col = wg.columns[0]
    if ward_col != "ward_id":
        wg = wg.rename(columns={ward_col: "ward_id"})
    wg["ward_id"] = wg["ward_id"].astype(str)
    return wg

def load_amenities(health_kml: Path, schools_csv: Path, parks_kml: Path, fire_kml: Path):
    health = parse_kml_points(health_kml) if health_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    parks = parse_kml_points(parks_kml) if parks_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    fire = parse_kml_points(fire_kml) if fire_kml.exists() else pd.DataFrame(columns=["name","lat","lon"])
    schools = pd.DataFrame()
    if schools_csv.exists():
        try:
            sd = pd.read_csv(schools_csv)
            latcol = next((c for c in sd.columns if "lat" in c.lower()), None)
            loncol = next((c for c in sd.columns if "lon" in c.lower() or "lng" in c.lower()), None)
            namecol = next((c for c in sd.columns if "name" in c.lower() or "school" in c.lower()), None)
            if latcol and loncol:
                recs=[]
                for i,row in sd.iterrows():
                    try:
                        recs.append({"name": row.get(namecol, f"school_{i}"), "lat": float(row[latcol]), "lon": float(row[loncol])})
                    except Exception:
                        continue
                schools = pd.DataFrame(recs)
        except Exception:
            pass
    return health, schools, parks, fire

# ----------------- h3 generation -----------------
def generate_h3_cells_for_wards(wards_gdf: gpd.GeoDataFrame, res:int, expand_k:int=2):
    if h3 is None:
        raise RuntimeError("h3 library required")
    log.info("Building H3 cells (res=%d)...", res)
    hset=set()
    for geom in wards_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            parts = geom.geoms if hasattr(geom,"geoms") else [geom]
            for poly in parts:
                ext = getattr(poly,"exterior", None)
                if ext is None:
                    continue
                for lon, lat in ext.coords:
                    try:
                        hset.add(h3.geo_to_h3(lat, lon, res))
                    except Exception:
                        continue
        except Exception:
            continue
    # expand
    initial = list(hset)
    for hcell in initial:
        try:
            hset.update(h3.k_ring(hcell, expand_k))
        except Exception:
            pass
    # filter
    union = wards_gdf.union_all if hasattr(wards_gdf, "unary_union") else wards_gdf.geometry.union_all
    cells=[]
    for c in hset:
        try:
            lat, lon = h3.h3_to_geo(c)
            pt = Point(lon, lat)
            if union.intersects(pt):
                cells.append(c)
        except Exception:
            continue
    log.info("H3 cells built: %d", len(cells))
    return sorted(cells)

# ----------------- assign ward robustly -----------------
def assign_wards_to_h3(h3_gdf: gpd.GeoDataFrame, wards_gdf: gpd.GeoDataFrame):
    # CRS
    if wards_gdf.crs is None:
        wards_gdf = wards_gdf.set_crs(epsg=4326)
    if h3_gdf.crs is None:
        h3_gdf = h3_gdf.set_crs(epsg=4326)

    wards = wards_gdf.to_crs(epsg=4326)
    h3 = h3_gdf.to_crs(epsg=4326)

    # ---- 1. polygon intersection ----
    try:
        inter = gpd.sjoin(h3[["h3_id","geometry"]], wards[["ward_id","geometry"]], how="left", predicate="intersects")
        inter_simple = inter.drop_duplicates(subset=["h3_id"]).set_index("h3_id")
        h3["ward_id"] = h3["h3_id"].map(inter_simple["ward_id"])
        n_matched = h3["ward_id"].notna().sum()
        log.info("Polygon-intersection ward matches: %d / %d", int(n_matched), len(h3))
    except Exception as e:
        log.warning("Polygon intersection sjoin failed: %s", e)
        h3["ward_id"] = None
        n_matched = 0

    # ---- 2. nearest-centroid fallback ----
    unmatched = h3[h3["ward_id"].isna()].copy()
    if len(unmatched) > 0:
        log.info("Falling back to nearest-ward assignment for %d cells...", len(unmatched))

        # ward centroids
        wards_cent = wards.to_crs(epsg=3857)
        wards_cent["wc_x"] = wards_cent.geometry.centroid.x
        wards_cent["wc_y"] = wards_cent.geometry.centroid.y

        ward_points = gpd.GeoDataFrame(wards_cent[["ward_id"]],
                                       geometry=wards_cent.geometry.centroid,
                                       crs=wards_cent.crs)

        sindex = ward_points.sindex

        unmatched_cent = unmatched.to_crs(epsg=3857)
        unmatched_cent["cx"] = unmatched_cent.geometry.centroid.x
        unmatched_cent["cy"] = unmatched_cent.geometry.centroid.y

        ward_coords = list(zip(wards_cent["wc_x"].values, wards_cent["wc_y"].values))
        ward_ids = list(wards_cent["ward_id"].values)

        assignments = {}

        for idx, row in unmatched_cent.iterrows():
            px = row["cx"]; py = row["cy"]
            pt = Point(px, py)

            # --- Correct pygeos/shapely2 nearest() ---
            if hasattr(sindex, "nearest"):
                # returns generator of indices (no num_results argument allowed)
                possible = list(sindex.nearest(pt))[:5]
            else:
                # older rtree
                possible = list(
                    sindex.intersection((px-2000, py-2000, px+2000, py+2000))
                )

            # compute nearest of candidates
            best_id = None; best_d = float("inf")
            for cand in possible:
                try:
                    wx, wy = ward_coords[cand]
                    d = (px - wx)**2 + (py - wy)**2
                    if d < best_d:
                        best_d = d
                        best_id = ward_ids[cand]
                except Exception:
                    continue

            # brute force backup
            if best_id is None:
                d_all = [ (px - wx)**2 + (py - wy)**2 for (wx,wy) in ward_coords ]
                best_id = ward_ids[int(np.argmin(d_all))]

            assignments[row["h3_id"]] = best_id

        # apply
        h3.loc[h3["h3_id"].isin(assignments.keys()), "ward_id"] = \
            h3.loc[h3["h3_id"].isin(assignments.keys()), "h3_id"].map(assignments)

        log.info("Nearest-ward assignments done: %d", len(assignments))

    # fill unknowns
    h3["ward_id"] = h3["ward_id"].fillna("ward_unknown")
    unk = int((h3["ward_id"] == "ward_unknown").sum())
    if unk > 0:
        log.warning("%d H3 cells remain ward_unknown", unk)

    return h3

# ----------------- minimal temporal generation helpers -----------------
def generate_time_index(days:int, freq:str="1H", tz="Asia/Kolkata"):
    end = pd.Timestamp.now(tz=tz).replace(minute=0, second=0, microsecond=0)
    start = end - pd.Timedelta(days=days)
    return pd.date_range(start=start, end=end - pd.Timedelta(seconds=1), freq=freq, tz=tz)

def diurnal_multiplier(n_steps):
    x = np.linspace(0, 2*math.pi, n_steps)
    return (np.sin(x - math.pi/2) + 1.0) / 2.0

def synth_pollution_series(n_cells, n_steps, base_aqi=70):
    mult = diurnal_multiplier(n_steps)
    out = np.zeros((n_steps, n_cells), dtype=int)
    for t in range(n_steps):
        base = base_aqi + (mult[t] - 0.5) * 40
        spatial = (np.linspace(0,1,n_cells)*0.4 + 0.8)
        vals = np.clip(base * spatial + rng.normal(0,12,n_cells), 0, 500)
        out[t] = vals.astype(int)
    return out

def synth_traffic_series(n_cells,n_steps):
    mult = diurnal_multiplier(n_steps)
    out=np.zeros((n_steps,n_cells))
    for t in range(n_steps):
        base = 35 - (mult[t]*15) + rng.normal(0,3,n_cells)
        out[t] = np.clip(base,5,60)
    return out

# ----------------- parquet validation -----------------
def is_valid_parquet(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            start = f.read(4)
            if start != b"PAR1": return False
            f.seek(-4,2); foot = f.read(4)
            if foot != b"PAR1": return False
        if pq is not None:
            pq.ParquetFile(str(path)).metadata
        else:
            pd.read_parquet(path, columns=[])
        return True
    except Exception:
        return False

# ----------------- main pipeline -----------------
def generate_temporal_dataset(days=30, h3_res=8, k_ring=6, freq="1H", outdir="./outputs", use_real_files=True):
    out_dir = Path(outdir); ensure_dir(out_dir)

    # load wards
    wards = None
    if DEFAULT_WARDS.exists():
        wards = load_wards(DEFAULT_WARDS)
        log.info("Loaded wards: %d features from %s", len(wards), DEFAULT_WARDS)
    else:
        log.warning("Wards file not found at %s", DEFAULT_WARDS)

    # generate H3 cells
    if wards is not None and not wards.empty:
        h3_cells = generate_h3_cells_for_wards(wards, res=h3_res, expand_k=2)
    else:
        if h3 is None:
            raise RuntimeError("h3 library required")
        center_h = h3.geo_to_h3(12.9716, 77.5946, h3_res)
        h3_cells = list(h3.k_ring(center_h, k_ring))

    if len(h3_cells) == 0:
        center_h = h3.geo_to_h3(12.9716, 77.5946, h3_res)
        h3_cells = list(h3.k_ring(center_h, k_ring))

    n_cells = len(h3_cells)
    log.info("Using %d H3 cells.", n_cells)

    # H3 GDF
    h3_rows=[]
    for hcell in h3_cells:
        lat, lon = h3.h3_to_geo(hcell)
        poly = h3_boundary_poly(hcell)
        h3_rows.append({"h3_id": hcell, "geometry": poly, "centroid_lat": lat, "centroid_lon": lon})
    h3_gdf = gpd.GeoDataFrame(h3_rows, geometry="geometry", crs="EPSG:4326")

    # ward assignment
    if wards is not None and not wards.empty:
        h3_gdf = assign_wards_to_h3(h3_gdf, wards)
    else:
        h3_gdf["ward_id"] = [f"ward_{(i%100)+1}" for i in range(len(h3_gdf))]

    # static proxies
    built_density = np.clip(np.linspace(50,300,n_cells) + rng.normal(0,8,n_cells), 5, 1000)
    green_cover = np.clip(np.linspace(0.2,0.05,n_cells) + rng.normal(0,0.02,n_cells), 0, 1)

    # time index
    timestamps = generate_time_index(days, freq=freq)
    n_steps = len(timestamps)
    log.info("Generating %d time steps (%s)", n_steps, freq)

    aqi_series = synth_pollution_series(n_cells, n_steps)
    traffic_series = synth_traffic_series(n_cells, n_steps)

    # per-day parquet
    daily_buffer=[]; current_day=None; written=[]
    for t_idx, ts in enumerate(tqdm(timestamps, desc="time steps")):
        ts_iso = pd.Timestamp(ts).isoformat()
        aqi_row = aqi_series[t_idx]; traffic_row = traffic_series[t_idx]

        for i,hcell in enumerate(h3_cells):
            daily_buffer.append({
                "timestamp": ts_iso,
                "ward_id": h3_gdf.at[i, "ward_id"],
                "h3_id": hcell,
                "aqi": int(aqi_row[i]),
                "traffic_speed_kmph": float(traffic_row[i]),
                "built_density_score": float(built_density[i]),
                "green_cover_ratio": float(green_cover[i]),
            })

        day_str = pd.Timestamp(ts).date().isoformat()
        if current_day is None:
            current_day = day_str
        next_ts = timestamps[t_idx+1] if t_idx+1 < n_steps else None
        next_day = pd.Timestamp(next_ts).date().isoformat() if next_ts is not None else None

        # roll day
        if next_day != current_day or (t_idx == n_steps-1):
            df_day = pd.DataFrame(daily_buffer)
            df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
            out_path = out_dir / f"temporal_{current_day}.parquet"

            try:
                df_day.set_index(["h3_id","timestamp"]).to_parquet(out_path, compression="snappy")
                log.info("WROTE day parquet: %s rows=%d", out_path, df_day.shape[0])
                written.append(out_path)
            except Exception as e:
                log.warning("Failed to write %s: %s", out_path, e)

            daily_buffer=[]; current_day = next_day

    # validate & combine
    daily_files = sorted(out_dir.glob("temporal_*.parquet"))
    valid=[]; skipped=[]
    for p in daily_files:
        if not p.exists() or p.stat().st_size < 20:
            skipped.append(str(p)); continue
        if not is_valid_parquet(p):
            skipped.append(str(p)); continue
        valid.append(p)

    combined_path = out_dir / f"temporal_{days}d_combined.parquet"
    if len(valid) == 0:
        raise RuntimeError("No valid parquet files to combine!")

    # streaming
    if pa is not None and pq is not None:
        writer=None
        try:
            for p in valid:
                try:
                    chunk = pd.read_parquet(p).reset_index()
                except Exception as e:
                    log.warning("Failed read %s: %s", p, e)
                    continue

                for col in chunk.columns:
                    if chunk[col].dtype == object:
                        if chunk[col].apply(lambda x: isinstance(x,(dict,list))).any():
                            chunk[col] = chunk[col].apply(lambda x: json.dumps(x) if isinstance(x,(dict,list)) else x)

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(combined_path), table.schema, compression="snappy")
                writer.write_table(table)

            if writer is not None:
                writer.close()
        finally:
            try:
                if writer is not None:
                    writer.close()
            except Exception:
                pass

        log.info("WROTE combined parquet: %s", combined_path)
    else:
        # fallback memory
        dfs=[pd.read_parquet(p).reset_index() for p in valid]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        df_all.set_index(["h3_id","timestamp"]).to_parquet(combined_path, compression="snappy")
        log.info("WROTE combined parquet (fallback): %s rows=%d", combined_path, df_all.shape[0])

    # write geojsons
    out_h3 = out_dir / "h3_cells.geojson"
    safe_write_geojson(h3_gdf, out_h3)

    # latest-h3 summary
    try:
        df_comb = pd.read_parquet(combined_path).reset_index()
        latest = df_comb.groupby("h3_id")["timestamp"].max().reset_index()
        latest_rows=[]
        for _, r in latest.iterrows():
            hcell = r["h3_id"]; ts = r["timestamp"]
            row = df_comb[(df_comb["h3_id"]==hcell) & (df_comb["timestamp"]==ts)].iloc[0].to_dict()
            latest_rows.append(row)
        df_latest = pd.DataFrame(latest_rows)
        h3_latest_gdf = h3_gdf.merge(df_latest, on="h3_id", how="left")
        safe_write_geojson(h3_latest_gdf, out_dir / "temporal_latest_h3.geojson")
    except Exception as e:
        log.warning("Failed to write temporal_latest_h3.geojson: %s", e)

    log.info("Pipeline finished. Outputs in %s", out_dir)
    return combined_path, out_h3


# ----------------- CLI -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--h3-res", type=int, default=8)
    p.add_argument("--k-ring", type=int, default=6)
    p.add_argument("--freq", default="1H")
    p.add_argument("--out", default="./outputs")
    p.add_argument("--use-real-files", action="store_true")
    args = p.parse_args()

    generate_temporal_dataset(days=args.days, h3_res=args.h3_res, k_ring=args.k_ring,
                              freq=args.freq, outdir=args.out, use_real_files=args.use_real_files)
