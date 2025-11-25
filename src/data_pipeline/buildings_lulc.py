# fixed_buildings_lulc.py
"""
Fixed pipeline combining wards, H3 polyfill, and building -> H3 aggregation.

Fixes applied:
 - import csv (fix NameError)
 - compute centroids/areas in projected CRS to avoid geographic centroid warnings
 - use shapely.ops.unary_union instead of deprecated unary_union attr
 - robust H3 hex area handling across bindings
 - ensure single geometry column when writing GeoJSON
"""

import os
import csv
import json
import math
import logging
from collections import defaultdict, Counter

import fiona
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from pyproj import Transformer
from tqdm import tqdm

# H3
import h3

# Optional raster support
try:
    import rasterio
    from rasterio.features import geometry_mask
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

# CONFIG - adjust paths as needed
BUILDINGS_GEOJSON = "data/geo/buildings.geojson"
WARDS_GEOJSON = "data/geo/wards/wards_master_enriched.geojson"
OUT_DIR = "data/geo"
H3_RES = 8
BATCH_SIZE = 50000
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# transformers
to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

def geom_area_m2(geom):
    """Compute area in m^2 by projecting to Web Mercator (EPSG:3857)."""
    try:
        proj = transform(to_merc, geom)
        return abs(proj.area)
    except Exception:
        return float("nan")

def geom_centroid_wgs84(geom):
    """Compute centroid robustly: centroid of projected geometry, then back to WGS84 point."""
    proj = transform(to_merc, geom)
    cent_proj = proj.centroid
    pt_wgs = transform(to_wgs, Point(cent_proj.x, cent_proj.y))
    lon, lat = pt_wgs.x, pt_wgs.y

    return Point(lon, lat)

# H3 helpers (robust across bindings)
def hex_area_sqm_from_h3(h):
    """Return hex area (m^2) robustly for h3 index `h`."""
    try:
        # prefer cell_area (per-cell) if available
        if hasattr(h3, "cell_area"):
            return float(h3.cell_area(h, unit="m^2"))
    except Exception:
        pass
    try:
        # fallback: compute resolution from cell and use hex_area(res)
        if hasattr(h3, "h3_get_resolution") and hasattr(h3, "hex_area"):
            res = int(h3.h3_get_resolution(h))
            return float(h3.hex_area(res, unit="m^2"))
    except Exception:
        pass
    try:
        # older binding name
        if hasattr(h3, "h3_get_resolution"):
            res = int(h3.h3_get_resolution(h))
            if hasattr(h3, "cell_area"):
                return float(h3.cell_area(h, unit="m^2"))
    except Exception:
        pass
    return None

def polyfill_geom_to_h3(geom, res):
    """Return set of h3 ids that intersect shape geom. Robust: tries multiple binding APIs then sampling fallback."""
    # convert to geojson-like
    gj = mapping(geom)
    # produce lat-lng rings
    rings = []
    if gj["type"] == "Polygon":
        rings.append([(pt[1], pt[0]) for pt in gj["coordinates"][0]])
    elif gj["type"] == "MultiPolygon":
        for poly in gj["coordinates"]:
            rings.append([(pt[1], pt[0]) for pt in poly[0]])
    else:
        return set()

    # Try direct polyfill functions (many variants across h3 bindings)
    try:
        if hasattr(h3, "polyfill"):
            # some bindings accept list of (lat,lng) for polyfill
            out = h3.polyfill(rings[0], res)
            return set(out)
    except Exception:
        pass

    try:
        if hasattr(h3, "polyfill_geojson"):
            # expects geojson-like
            out = h3.polyfill_geojson(gj, res)
            return set(out)
    except Exception:
        pass

    try:
        # some bindings have polyfill_polygon
        if hasattr(h3, "polyfill_polygon"):
            out = h3.polyfill_polygon(gj, res)
            return set(out)
    except Exception:
        pass

    try:
        # experimental polygon_to_cells_experimental (build LatLngPoly if available)
        if hasattr(h3, "polygon_to_cells_experimental") and hasattr(h3, "LatLngPoly"):
            latlngpoly = h3.LatLngPoly()
            for ring in rings:
                latlngpoly.add_ring([h3.LatLng(lat, lon) for lat, lon in ring])
            out = h3.polygon_to_cells_experimental(latlngpoly, res, contain="center")
            return set(out)
    except Exception:
        pass

    # Sampling fallback (robust but slower)
    try:
        minx, miny, maxx, maxy = geom.bounds  # lon/lat
        step = 0.0012  # ~120m - adjust if needed
        collected = set()
        x = minx
        while x <= maxx:
            y = miny
            while y <= maxy:
                pt = Point(x, y)
                if geom.contains(pt):
                    try:
                        h = h3.geo_to_h3(y, x, res)  # geo_to_h3 often expects (lat,lng)
                        collected.add(h)
                    except Exception:
                        try:
                            h = h3.geo_to_h3(x, y, res)
                            collected.add(h)
                        except Exception:
                            pass
                y += step
            x += step
        return collected
    except Exception:
        return set()

def centroid_h3(geom, res):
    c = geom.centroid
    lat, lon = c.y, c.x
    try:
        return h3.geo_to_h3(lat, lon, res)
    except Exception:
        try:
            return h3.geo_to_h3(lon, lat, res)
        except Exception:
            return None

# --- Load wards, compute centroids/areas properly ---
logging.info(f"Loading wards from: {WARDS_GEOJSON}")
gdf_wards = gpd.read_file(WARDS_GEOJSON)
if "ward_id" not in gdf_wards.columns:
    if "ward_num" in gdf_wards.columns:
        gdf_wards["ward_id"] = gdf_wards["ward_num"].apply(lambda x: f"ward_{int(x)}")
    elif "ward_name" in gdf_wards.columns:
        gdf_wards["ward_id"] = gdf_wards["ward_name"]
    else:
        raise RuntimeError("wards file missing ward_id column and no fallback found.")

# Compute projected centroids and areas to avoid the geographic centroid warning
gdf_wards = gdf_wards.to_crs("EPSG:4326")  # ensure base
# compute centroid in mercator then convert back to WGS84 for robust centroid
centroids = []
areas = []
for geom in gdf_wards.geometry:
    try:
        a = geom_area_m2(geom)
        areas.append(a)
    except Exception:
        areas.append(None)
    try:
        c = geom_centroid_wgs84(geom)
        centroids.append(c)
    except Exception:
        # fallback to shapely centroid
        centroids.append(geom.centroid)
gdf_wards["area_sqm"] = areas
gdf_wards["area_sqkm"] = [a/1e6 if a is not None else None for a in areas]
gdf_wards["ward_centroid"] = centroids

# Spatial index
ward_sindex = gdf_wards.sindex
logging.info(f"Wards loaded: {len(gdf_wards)}")

# union geometry using shapely.ops.unary_union (avoid deprecated unary_union attr)
union_geom = unary_union(list(gdf_wards.geometry))
logging.info("Union geometry built.")

# Polyfill H3 for union
logging.info(f"Polyfilling H3 resolution {H3_RES} ...")
h3_set = polyfill_geom_to_h3(union_geom, H3_RES)
h3_list = sorted(list(h3_set))
logging.info(f"Total hexes covering union: {len(h3_list)}")

# Build H3 CSV + GeoJSON (single geometry column)
h3_rows = []
h3_features = []
for h in h3_list:
    # centroid lat lon
    try:
        lat, lon = h3.h3_to_geo(h)
    except Exception:
        lat, lon = None, None
    area_sqm = hex_area_sqm_from_h3(h)
    h3_rows.append({"h3_id": h, "centroid_lat": lat, "centroid_lon": lon, "area_sqm": area_sqm})
    # polygon
    try:
        boundary = h3.h3_to_geo_boundary(h, geo_json=True)
        coords = [[(pt[1], pt[0]) for pt in boundary] + [(boundary[0][1], boundary[0][0])]]
        feat = {"type":"Feature", "properties":{"h3_id":h}, "geometry":{"type":"Polygon", "coordinates": coords}}
        h3_features.append(feat)
    except Exception:
        pass

pd.DataFrame(h3_rows).to_csv(os.path.join(OUT_DIR, f"h3_grid_res{H3_RES}.csv"), index=False)
with open(os.path.join(OUT_DIR, f"h3_grid_res{H3_RES}.geojson"), "w", encoding="utf8") as f:
    json.dump({"type":"FeatureCollection", "features": h3_features}, f)

# --- Process buildings stream ---
out_buildings_csv = os.path.join(OUT_DIR, "buildings_processed.csv")
parquet_base = os.path.join(OUT_DIR, "buildings_batch_")

csv_cols = ["building_id","geometry","centroid_lat","centroid_lon","footprint_m2",
            "ward_id","h3_id_centroid","h3_ids","building_type","floors","height_m",
            "year_built","roof_type","source_tags"]

# write CSV header
with open(out_buildings_csv, "w", newline="", encoding="utf8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_cols)

# aggregates per h3
h3_agg = defaultdict(lambda: {"total_buildings":0, "total_footprint":0.0, "ward_counts":Counter(), "floors_sum":0.0, "height_sum":0.0, "height_count":0})

def assign_ward_for_point(pt: Point):
    # search spatial index by point bbox
    possible = list(ward_sindex.intersection(pt.bounds))
    for idx in possible:
        try:
            if gdf_wards.loc[idx, "geometry"].contains(pt):
                return gdf_wards.loc[idx, "ward_id"]
        except Exception:
            continue
    # fallback: nearest centroid
    dists = gdf_wards["ward_centroid"].distance(pt)
    idx = dists.idxmin()
    return gdf_wards.loc[idx, "ward_id"]

def infer_type(props):
    # same heuristics as earlier
    for k in ("building:use","building","landuse","amenity","shop"):
        v = props.get(k)
        if v:
            s = str(v).lower()
            if any(x in s for x in ("house","resid","apartment","flat")):
                return "residential"
            if any(x in s for x in ("shop","retail","market","mall","commercial","supermarket","office")):
                return "commercial"
            if any(x in s for x in ("industrial","factory","warehouse")):
                return "industrial"
            return s
    return "unknown"

def safe_get(props, keys):
    for k in keys:
        v = props.get(k)
        if v is not None:
            return v
    return None

logging.info("Processing buildings (stream)...")
with fiona.open(BUILDINGS_GEOJSON, "r") as src:
    total = len(src) if hasattr(src, "__len__") else None
    it = tqdm(src, total=total)
    batch = []
    parquet_count = 0
    for i, feat in enumerate(it):
        props = feat.get("properties", {}) or {}
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        footprint = geom_area_m2(geom)
        centroid = geom_centroid_wgs84(geom)
        centroid_lat, centroid_lon = centroid.y, centroid.x
        h3_cent = centroid_h3(centroid, H3_RES)
        try:
            h3_ids = sorted(list(polyfill_geom_to_h3(geom, H3_RES)))
        except Exception:
            h3_ids = []
        if not h3_ids and h3_cent:
            h3_ids = [h3_cent]

        building_type = infer_type(props)
        floors = safe_get(props, ("building:levels","levels","floors","max_level"))
        try:
            floors = float(floors) if floors is not None else None
        except:
            floors = None
        height = safe_get(props, ("height","building:height","roof:height"))
        try:
            if height is not None:
                height = float(str(height).replace("m","").strip())
        except:
            height = None
        year_built = safe_get(props, ("start_date","opening_date","year_built"))
        roof = safe_get(props, ("roof:shape","roof","building:roof"))

        # ward assignment using centroid
        ward_id = assign_ward_for_point(centroid)

        row = {
            "building_id": i,
            "geometry": json.dumps(mapping(geom)),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "footprint_m2": footprint,
            "ward_id": ward_id,
            "h3_id_centroid": h3_cent,
            "h3_ids": json.dumps(h3_ids),
            "building_type": building_type,
            "floors": floors,
            "height_m": height,
            "year_built": year_built,
            "roof_type": roof,
            "source_tags": json.dumps({k:props.get(k) for k in ("building","amenity","shop","name","source") if k in props}, ensure_ascii=False)
        }
        batch.append(row)

        # update h3 aggregates (divide footprint among intersecting hexes)
        if h3_ids and footprint and footprint>0:
            per = footprint / len(h3_ids)
            for h in h3_ids:
                rec = h3_agg[h]
                rec["total_buildings"] += 1
                rec["total_footprint"] += per
                if ward_id:
                    rec["ward_counts"][ward_id] += 1
                if floors is not None:
                    rec["floors_sum"] += floors
                if height is not None:
                    rec["height_sum"] += height
                    rec["height_count"] += 1

        # flush
        if len(batch) >= BATCH_SIZE:
            df = pd.DataFrame(batch)[csv_cols]
            df.to_csv(out_buildings_csv, mode="a", header=False, index=False)
            # parquet chunk
            pfile = f"{parquet_base}{parquet_count}.parquet"
            df.to_parquet(pfile, index=False)
            parquet_count += 1
            batch = []

    # final flush
    if batch:
        df = pd.DataFrame(batch)[csv_cols]
        df.to_csv(out_buildings_csv, mode="a", header=False, index=False)
        pfile = f"{parquet_base}{parquet_count}.parquet"
        df.to_parquet(pfile, index=False)
        parquet_count += 1
        batch = []

logging.info("Buildings streaming complete.")

# Build h3_buildings_summary CSV
h3_rows = []
for h, stats in h3_agg.items():
    hex_area = hex_area_sqm_from_h3(h)
    builtup_pct = None
    if hex_area and stats["total_footprint"] is not None:
        builtup_pct = min(100.0, 100.0 * stats["total_footprint"] / hex_area)
    dominant_ward = stats["ward_counts"].most_common(1)[0][0] if stats["ward_counts"] else None
    avg_floors = (stats["floors_sum"]/stats["total_buildings"]) if stats["total_buildings"]>0 and stats["floors_sum"] else None
    avg_height = (stats["height_sum"]/stats["height_count"]) if stats["height_count"]>0 else None
    try:
        lat, lon = h3.h3_to_geo(h)
    except:
        lat, lon = (None, None)
    h3_rows.append({
        "h3_id": h,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "ward_id": dominant_ward,
        "hex_area_sqm": hex_area,
        "total_buildings": stats["total_buildings"],
        "total_footprint_m2": stats["total_footprint"],
        "builtup_pct": builtup_pct,
        "avg_floors": avg_floors,
        "avg_height_m": avg_height
    })
df_h3 = pd.DataFrame(h3_rows)
df_h3.to_csv(os.path.join(OUT_DIR, f"h3_buildings_summary_res{H3_RES}.csv"), index=False)

logging.info("Wrote H3 buildings summary.")

# LULC fallback: built-up percentage from building footprints is already in df_h3 (builtup_pct).
# If you have raster LULC inputs, run zonal stats separately (code omitted for brevity but can be added).
logging.info("Pipeline finished successfully. Outputs in outputs/ directory.")
