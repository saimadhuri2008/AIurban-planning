# src/data_pipeline/feature_aggregation.py
"""
Final H3 feature aggregation with robust pathing and auto-generation of missing inputs.

Reads preferred:
  src/data_pipeline/spatial/nodes_bengaluru.geojson
  src/data_pipeline/spatial/edges_bengaluru.geojson
  src/data_pipeline/remote_sensing/{ntl_bengaluru_mock, ndvi_bengaluru_mock}.geojson
  src/data_pipeline/policy/policy_bengaluru_mock.geojson
  src/data_pipeline/mobility/mobility_bengaluru_final.geojson
  src/data_pipeline/utilities/utilities_bengaluru_final.geojson

If missing, generates them (simple, synthetic) and continues.

Writes:
  data/processed/spatial/h3_gdf_final_static.geojson
  data/processed/spatial/edges_gdf_final_static.geojson
  data/processed/temporal/temporal_data_h3_48h.csv
  data/processed/temporal/temporal_traffic_48h.csv
"""

import warnings
from pathlib import Path
from datetime import datetime
import os
import random

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Optional: OSMnx for auto-generation ----------
try:
    import osmnx as ox
    HAS_OSMNX = True
except Exception:
    HAS_OSMNX = False

# ---------- H3 ----------
try:
    import h3
except ImportError:
    raise ImportError("The 'h3' package is required. Install with: pip install h3")


# =========================
# Config & Paths
# =========================
np.random.seed(42)
random.seed(42)

# Project root = two levels up from this file (…/urban planning)
ROOT = Path(__file__).resolve().parents[2]

# INPUT folders (pipeline sources)
PIPELINE_DIR = ROOT / "src" / "data_pipeline"
SPATIAL_IN   = PIPELINE_DIR / "spatial"
REMOTE_IN    = PIPELINE_DIR / "remote_sensing"
POLICY_IN    = PIPELINE_DIR / "policy"
MOBILITY_IN  = PIPELINE_DIR / "mobility"
UTILITIES_IN = PIPELINE_DIR / "utilities"
for p in (SPATIAL_IN, REMOTE_IN, POLICY_IN, MOBILITY_IN, UTILITIES_IN):
    p.mkdir(parents=True, exist_ok=True)

# OUTPUT folders
DATA_PROCESSED = ROOT / "data" / "processed"
SPATIAL_OUT    = DATA_PROCESSED / "spatial"
TEMPORAL_OUT   = DATA_PROCESSED / "temporal"
SPATIAL_OUT.mkdir(parents=True, exist_ok=True)
TEMPORAL_OUT.mkdir(parents=True, exist_ok=True)

# City & H3
CITY_NAME = "Bengaluru, India"   # change if needed
H3_RESOLUTION = 9
TIMESTEPS = 48  # half-hourly for 2 days

print(f"--- Starting Final H3 Feature Aggregation (Resolution: {H3_RESOLUTION}) ---")


# =========================
# Helpers: H3 robust funcs
# =========================
def get_h3_cells(geom_wgs84, res: int):
    geo = geom_wgs84.__geo_interface__ if hasattr(geom_wgs84, "__geo_interface__") else geom_wgs84
    try:
        if hasattr(h3, "polygon_to_cells"):
            return set(h3.polygon_to_cells(geo, res))
    except Exception:
        pass
    try:
        if hasattr(h3, "polygon_to_cells_experimental"):
            return set(h3.polygon_to_cells_experimental(geo, res))
    except Exception:
        pass
    try:
        if hasattr(h3, "polyfill"):
            return set(h3.polyfill(geo, res))
    except Exception:
        pass
    if hasattr(h3, "api"):
        for subname in dir(h3.api):
            sub = getattr(h3.api, subname)
            if hasattr(sub, "polygon_to_cells"):
                try:
                    return set(sub.polygon_to_cells(geo, res))
                except Exception:
                    pass
            if hasattr(sub, "polyfill"):
                try:
                    return set(sub.polyfill(geo, res))
                except Exception:
                    pass
    # last resort approximate sampler
    from shapely.geometry import shape
    poly = shape(geo) if not hasattr(geo, "geom_type") else geom_wgs84
    minx, miny, maxx, maxy = poly.bounds
    step = 0.005
    cells = set()
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            pt = Point(x, y)
            if poly.contains(pt) or poly.touches(pt):
                try:
                    hid = h3.latlng_to_cell(y, x, res)
                except Exception:
                    hid = h3.geo_to_h3(y, x, res)
                cells.add(hid)
            y += step
        x += step
    if cells:
        return cells
    raise RuntimeError("H3 polyfill failure.")


def h3_cell_to_polygon(hid):
    try:
        if hasattr(h3, "cell_to_boundary"):
            boundary = h3.cell_to_boundary(hid)  # [(lat, lon)]
            return Polygon([(lon, lat) for lat, lon in boundary])
    except Exception:
        pass
    if hasattr(h3, "api"):
        for subname in dir(h3.api):
            sub = getattr(h3.api, subname)
            if hasattr(sub, "cell_to_boundary"):
                try:
                    boundary = sub.cell_to_boundary(hid)
                    return Polygon([(lon, lat) for lat, lon in boundary])
                except Exception:
                    pass
    return None


# =========================
# Spatial inputs: find or build
# =========================
def find_or_build_spatial_inputs():
    nodes_path = SPATIAL_IN / "nodes_bengaluru.geojson"
    edges_path = SPATIAL_IN / "edges_bengaluru.geojson"

    if nodes_path.exists() and edges_path.exists():
        print(f" -> Found spatial inputs at {SPATIAL_IN}")
        return gpd.read_file(nodes_path).set_index("osmid"), gpd.read_file(edges_path)

    # Fallbacks?
    fallbacks = [
        (ROOT / "data" / "processed" / "spatial" / "nodes_bengaluru.geojson",
         ROOT / "data" / "processed" / "spatial" / "edges_bengaluru.geojson"),
        (ROOT / "data" / "raw" / "nodes_bengaluru.geojson",
         ROOT / "data" / "raw" / "edges_bengaluru.geojson"),
    ]
    for npth, epth in fallbacks:
        if npth.exists() and epth.exists():
            print(f" -> Found spatial inputs at fallback: {npth.parent}")
            nodes_gdf = gpd.read_file(npth).set_index("osmid")
            edges_gdf = gpd.read_file(epth)
            nodes_gdf.reset_index().to_file(nodes_path, driver="GeoJSON")
            edges_gdf.to_file(edges_path, driver="GeoJSON")
            return nodes_gdf.set_index("osmid"), edges_gdf

    if not HAS_OSMNX:
        raise FileNotFoundError(
            "Missing nodes/edges and OSMnx not installed. Install: pip install osmnx\n"
            f"Or place GeoJSONs under {SPATIAL_IN}"
        )

    print(f" -> Spatial inputs not found. Generating from OSM for '{CITY_NAME}' ...")
    ox.settings.use_cache = True
    G = ox.graph_from_place(CITY_NAME, network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G)
    nodes.to_file(nodes_path, driver="GeoJSON")
    edges.to_file(edges_path, driver="GeoJSON")
    print(f" -> Saved generated nodes/edges to {SPATIAL_IN}")
    return gpd.read_file(nodes_path).set_index("osmid"), gpd.read_file(edges_path)


# =========================
# Mock-layer builders (only if missing)
# =========================
def ensure_mock_layers(study_area_poly_wgs84, crs):
    """Create simple synthetic layers if they don't exist."""
    # --- Remote sensing (NTL & NDVI) as coarse H3 polygons with values ---
    ntl_p = REMOTE_IN / "ntl_bengaluru_mock.geojson"
    ndvi_p = REMOTE_IN / "ndvi_bengaluru_mock.geojson"
    if not ntl_p.exists() or not ndvi_p.exists():
        # use coarser H3 to create polygons
        res_coarse = max(6, H3_RESOLUTION - 2)
        cells = list(get_h3_cells(study_area_poly_wgs84, res_coarse))
        polys = []
        for hid in cells:
            poly = h3_cell_to_polygon(hid)
            if poly and poly.is_valid:
                polys.append(poly)
        # random-ish values (but spatially smooth by sorting)
        n = len(polys)
        vals_ntl  = np.clip(np.linspace(0.1, 1.0, n) + np.random.normal(0, 0.05, n), 0.05, 1.2)
        vals_ndvi = np.clip(np.linspace(0.2, 0.8, n)[::-1] + np.random.normal(0, 0.05, n), 0.0, 1.0)
        ntl = gpd.GeoDataFrame({"avg_ntl_intensity": vals_ntl}, geometry=polys, crs="EPSG:4326").to_crs(crs)
        ndv = gpd.GeoDataFrame({"avg_ndvi": vals_ndvi}, geometry=polys, crs="EPSG:4326").to_crs(crs)
        ntl.to_file(ntl_p, driver="GeoJSON")
        ndv.to_file(ndvi_p, driver="GeoJSON")
        print(f" -> Mock NTL/NDVI generated at {REMOTE_IN}")

    # --- Policy zoning polygons: split bounding box into grid with FAR classes ---
    pol_p = POLICY_IN / "policy_bengaluru_mock.geojson"
    if not pol_p.exists():
        minx, miny, maxx, maxy = gpd.GeoSeries([study_area_poly_wgs84], crs="EPSG:4326").to_crs(crs).total_bounds
        cols, rows = 4, 4
        w = (maxx - minx) / cols
        h = (maxy - miny) / rows
        cells = []
        fars = [1.5, 2.0, 3.0, 4.0]
        zones = ["R", "C", "M", "I"]
        for i in range(cols):
            for j in range(rows):
                cells.append(box(minx + i*w, miny + j*h, minx + (i+1)*w, miny + (j+1)*h))
        df = gpd.GeoDataFrame({
            "base_zoning": [random.choice(zones) for _ in cells],
            "permissible_far": [random.choice(fars) for _ in cells]
        }, geometry=cells, crs=crs)
        df.to_file(pol_p, driver="GeoJSON")
        print(f" -> Mock policy zoning generated at {POLICY_IN}")

    # --- Mobility POIs: take midpoints of a sample of edges ---
    mob_p = MOBILITY_IN / "mobility_bengaluru_final.geojson"
    if not mob_p.exists():
        edges_path = SPATIAL_IN / "edges_bengaluru.geojson"
        egdf = gpd.read_file(edges_path).to_crs(crs)
        egdf = egdf[egdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        egdf["midpoint"] = egdf.geometry.interpolate(0.5, normalized=True)
        pts = egdf.sample(min(250, len(egdf)), random_state=42)["midpoint"].reset_index(drop=True)
        mob = gpd.GeoDataFrame({
            "daily_ridership_mock": np.random.randint(500, 5000, len(pts))
        }, geometry=pts, crs=crs)
        mob.to_file(mob_p, driver="GeoJSON")
        print(f" -> Mock mobility POIs generated at {MOBILITY_IN}")

    # --- Utilities POIs: another sample of edge midpoints with capacity/reliability ---
    util_p = UTILITIES_IN / "utilities_bengaluru_final.geojson"
    if not util_p.exists():
        edges_path = SPATIAL_IN / "edges_bengaluru.geojson"
        egdf = gpd.read_file(edges_path).to_crs(crs)
        egdf = egdf[egdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        egdf["midpoint"] = egdf.geometry.interpolate(0.7, normalized=True)
        pts = egdf.sample(min(150, len(egdf)), random_state=24)["midpoint"].reset_index(drop=True)
        util = gpd.GeoDataFrame({
            "capacity_mock": np.random.randint(5, 50, len(pts)),
            "reliability_score_mock": np.round(np.random.uniform(0.6, 0.99, len(pts)), 3)
        }, geometry=pts, crs=crs)
        util.to_file(util_p, driver="GeoJSON")
        print(f" -> Mock utilities POIs generated at {UTILITIES_IN}")


# =========================
# Load inputs (or build)
# =========================
nodes_gdf, edges_gdf = find_or_build_spatial_inputs()

# Study area (WGS84) for H3 + mock generation
try:
    study_area_poly_wgs84 = nodes_gdf.to_crs(epsg=4326).geometry.union_all().convex_hull
except Exception:
    study_area_poly_wgs84 = nodes_gdf.to_crs(epsg=4326).geometry.unary_union.convex_hull

# Create mock layers if missing
ensure_mock_layers(study_area_poly_wgs84, crs=nodes_gdf.crs)

# Now read all layers (they exist for sure)
ntl_gdf      = gpd.read_file(REMOTE_IN / "ntl_bengaluru_mock.geojson").to_crs(nodes_gdf.crs)
ndvi_gdf     = gpd.read_file(REMOTE_IN / "ndvi_bengaluru_mock.geojson").to_crs(nodes_gdf.crs)
policy_gdf   = gpd.read_file(POLICY_IN / "policy_bengaluru_mock.geojson").to_crs(nodes_gdf.crs)
mobility_gdf = gpd.read_file(MOBILITY_IN / "mobility_bengaluru_final.geojson").to_crs(nodes_gdf.crs)
utilities_gdf= gpd.read_file(UTILITIES_IN / "utilities_bengaluru_final.geojson").to_crs(nodes_gdf.crs)

print(" -> Inputs ready (including generated mocks if needed).")


# =========================
# H3 Grid generation
# =========================
print("\n[STEP] H3 grid")
h3_ids = get_h3_cells(study_area_poly_wgs84, H3_RESOLUTION)

cells, kept = [], []
for hid in h3_ids:
    poly = h3_cell_to_polygon(hid)
    if poly is not None and poly.is_valid:
        cells.append(poly)
        kept.append(hid)

h3_gdf = gpd.GeoDataFrame({"h3_index": kept}, geometry=cells, crs="EPSG:4326").to_crs(nodes_gdf.crs).set_index("h3_index")
print(f" -> {len(h3_gdf):,} H3 cells generated.")


# =========================
# Static Feature Fusion
# =========================
print("\n[STEP] Static fusion")

ntl_join = gpd.sjoin(h3_gdf[["geometry"]], ntl_gdf[["avg_ntl_intensity", "geometry"]], how="left", predicate="contains")
h3_gdf["static_ntl_intensity"] = ntl_join.groupby(ntl_join.index)["avg_ntl_intensity"].mean().fillna(0.1)

ndvi_join = gpd.sjoin(h3_gdf[["geometry"]], ndvi_gdf[["avg_ndvi", "geometry"]], how="left", predicate="contains")
h3_gdf["static_avg_ndvi"] = ndvi_join.groupby(ndvi_join.index)["avg_ndvi"].mean().fillna(0.1)

policy_join = gpd.sjoin(
    h3_gdf[["geometry"]],
    policy_gdf[["base_zoning", "permissible_far", "geometry"]],
    how="left",
    predicate="intersects",
)
h3_gdf["policy_zone_far"] = policy_join.groupby(policy_join.index)["permissible_far"].mean().fillna(2.0)
h3_gdf = pd.get_dummies(h3_gdf, columns=["policy_zone_far"], prefix="zoning_far")

mob_join = gpd.sjoin(h3_gdf[["geometry"]], mobility_gdf, how="left", predicate="contains")
mob_agg = mob_join.groupby(mob_join.index).agg(
    poi_transit_count=("daily_ridership_mock", "count"),
    h3_avg_ridership=("daily_ridership_mock", "mean"),
).fillna(0)
h3_gdf = h3_gdf.join(mob_agg)

util_join = gpd.sjoin(h3_gdf[["geometry"]], utilities_gdf, how="left", predicate="contains")
util_agg = util_join.groupby(util_join.index).agg(
    poi_utility_count=("capacity_mock", "count"),
    h3_avg_capacity=("capacity_mock", "mean"),
    h3_avg_reliability=("reliability_score_mock", "mean"),
).fillna(0)
h3_gdf = h3_gdf.join(util_agg)

# Synthetic socioeconomics
if "zoning_far_4.0" not in h3_gdf.columns:
    h3_gdf["zoning_far_4.0"] = 0
h3_gdf["sim_pop_density"] = (
    h3_gdf["static_ntl_intensity"] * 1500
    + h3_gdf["zoning_far_4.0"] * 5000
    + np.random.normal(500, 500, len(h3_gdf))
)
h3_gdf["static_median_income_INR"] = 50000 + h3_gdf["static_ntl_intensity"] * 10000
h3_gdf["static_dependency_ratio"] = np.clip(
    h3_gdf["sim_pop_density"].rank(pct=True).apply(lambda x: 1 - x) * 0.3 + 0.4, 0.3, 0.9
)

print(" -> Static features fused.")


# =========================
# Temporal Simulation
# =========================
print("\n[STEP] Temporal simulation")

timestamps = pd.to_datetime(pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=int(TIMESTEPS), freq="30min"))
h3_idx_list = h3_gdf.index.tolist()

temporal_h3_records = []
for ts in timestamps:
    hour = ts.hour
    base_aqi = 50 + 20 * np.sin(2 * np.pi * (hour - 8) / 24)
    aqi = base_aqi + np.random.normal(0, 5, len(h3_gdf)) + h3_gdf["sim_pop_density"].values * 0.001
    temp_c = 25 + 5 * np.sin(2 * np.pi * (hour - 14) / 24) + np.random.normal(0, 1, len(h3_gdf))
    temporal_h3_records.append(pd.DataFrame({
        "h3_index": h3_idx_list, "timestamp": ts,
        "aqi": np.clip(aqi, 20, 150), "temp_c": np.clip(temp_c, 15, 40)
    }))
temporal_h3_df = pd.concat(temporal_h3_records, ignore_index=True)

# Edge policy + targets
edges_gdf = edges_gdf.copy()
edges_gdf.rename(columns={"u": "source", "v": "target", "key": "key_id"}, inplace=True)
edges_gdf.set_index(["source", "target", "key_id"], inplace=True)

np.random.seed(43)
edges_gdf["is_low_emissions_zone"] = np.random.choice([0, 1], len(edges_gdf), p=[0.9, 0.1])
edges_gdf["max_speed_policy"] = np.random.choice([30, 40, 50, 60], len(edges_gdf), p=[0.1, 0.3, 0.4, 0.2])
edges_gdf["is_public_transit_route"] = np.random.choice([0, 1], len(edges_gdf), p=[0.8, 0.2])
if "length" not in edges_gdf.columns:
    edges_gdf["length"] = 50.0

temporal_edge_records = []
h3_temp_indexed = temporal_h3_df.set_index(["h3_index", "timestamp"])

for ts in timestamps:
    avg_aqi = h3_temp_indexed.xs(ts, level="timestamp")["aqi"].mean()
    hour = ts.hour
    base_flow = 50 + 100 * np.sin(2 * np.pi * (hour - 8) / 24)
    sim_vehicle_count = np.clip(base_flow * np.random.normal(1, 0.1, len(edges_gdf)), 20, 500)
    base_speed_mps = edges_gdf["max_speed_policy"] * 1000 / 3600.0
    congestion_factor = 1 + (sim_vehicle_count / 300.0) + (avg_aqi / 100.0)
    avg_travel_time_s = (edges_gdf["length"] / base_speed_mps) * congestion_factor
    temporal_edge_records.append(pd.DataFrame({
        "u": edges_gdf.index.get_level_values("source"),
        "v": edges_gdf.index.get_level_values("target"),
        "key": edges_gdf.index.get_level_values("key_id"),
        "timestamp": ts,
        "sim_vehicle_count": sim_vehicle_count,
        "avg_travel_time_s": np.clip(avg_travel_time_s, 5, 1200)
    }))
temporal_traffic_df = pd.concat(temporal_edge_records, ignore_index=True)

# =========================
# Save outputs
# =========================
print("\n[STEP] Save outputs → data/processed/**")
(h3_gdf.reset_index()).to_file(SPATIAL_OUT / "h3_gdf_final_static.geojson", driver="GeoJSON")
(edges_gdf.reset_index()).to_file(SPATIAL_OUT / "edges_gdf_final_static.geojson", driver="GeoJSON")
temporal_h3_df.to_csv(TEMPORAL_OUT / "temporal_data_h3_48h.csv", index=False)
temporal_traffic_df.to_csv(TEMPORAL_OUT / "temporal_traffic_48h.csv", index=False)

print("\n--- Complete. Spatial + temporal artifacts written. ---")
print(f"Spatial → {SPATIAL_OUT}")
print(f"Temporal → {TEMPORAL_OUT}")
