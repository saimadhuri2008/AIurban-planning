"""
Robust pipeline for osmnx (auto-detect), POIs, H3 (using polygon_to_cells),
and H3 cell polygon construction using cell_to_boundary (works with your h3).
"""

import osmnx as ox
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import h3
import shapely.geometry as sg
import numpy as np
import traceback

warnings.filterwarnings('ignore')

# --- CONFIG ---
PLACE_NAME = "Bengaluru, India"
NETWORK_TYPE = "drive"
H3_RESOLUTION = 8
BUFFER_DIST = 1000  # meters

print("osmnx version detected:", ox.__version__)
print("h3 module info (using polygon_to_cells & cell_to_boundary).")

# --- Robust H3 polyfill using your h3 API ---
def h3_polyfill_with_polygon_to_cells(geo_json_polygon, res):
    """
    Use h3.polygon_to_cells (preferred for your h3 package) to return a set of H3 indexes.
    geo_json_polygon: a GeoJSON-like mapping (.__geo_interface__ from shapely/geopandas)
    res: integer resolution
    """
    # Try direct polygon_to_cells
    if hasattr(h3, "polygon_to_cells"):
        try:
            # Many builds accept (geo_json_polygon, res)
            ids = h3.polygon_to_cells(geo_json_polygon, res)
            # polygon_to_cells sometimes returns list - convert to set
            return set(ids)
        except Exception as e:
            # try experimental variant name if available
            pass

    # experimental variant
    if hasattr(h3, "polygon_to_cells_experimental"):
        try:
            ids = h3.polygon_to_cells_experimental(geo_json_polygon, res)
            return set(ids)
        except Exception:
            pass

    # If polygon_to_cells isn't available/working, try the 'api' namespace
    try:
        if hasattr(h3, "api"):
            # some h3 builds have api.*; attempt to discover polygon_to_cells
            api_mod = h3.api
            # try basic
            if hasattr(api_mod, "basic") and hasattr(api_mod.basic, "polygon_to_cells"):
                return set(api_mod.basic.polygon_to_cells(geo_json_polygon, res))
            # try other nested modules
            for attr in dir(api_mod):
                sub = getattr(api_mod, attr)
                if hasattr(sub, "polygon_to_cells"):
                    try:
                        return set(sub.polygon_to_cells(geo_json_polygon, res))
                    except Exception:
                        pass
    except Exception:
        pass

    # Last resort: approximate by sampling points and using latlng_to_cell
    try:
        from shapely.geometry import shape, Point
        poly = shape(geo_json_polygon)
        minx, miny, maxx, maxy = poly.bounds
        # pick a step guided by resolution (smaller res => larger cells => coarser step)
        # conservative default:
        step = 0.01
        h3_ids = set()
        x = minx
        while x <= maxx:
            y = miny
            while y <= maxy:
                pt = Point(x, y)
                if poly.contains(pt) or poly.touches(pt):
                    # h3 expects lat, lon for latlng_to_cell
                    try:
                        hid = h3.latlng_to_cell(y, x, res)
                        h3_ids.add(hid)
                    except Exception:
                        # many builds also expose geo_to_cells / geo_to_h3 but we try latlng_to_cell
                        pass
                y += step
            x += step
        if len(h3_ids) > 0:
            return h3_ids
    except Exception:
        pass

    raise AttributeError("No usable polygon_to_cells-like function found in this h3 build.")

# --- H3 cell boundary builder using cell_to_boundary ---
def h3_cell_boundary_to_polygon(h3_index):
    """
    Build a shapely Polygon for a single h3 index using h3.cell_to_boundary or h3.cell_to_vertex
    Returns shapely.geometry.Polygon in (lon, lat) order.
    """
    # many builds have cell_to_boundary(h3_index) -> list of (lat, lon) pairs
    if hasattr(h3, "cell_to_boundary"):
        try:
            boundary = h3.cell_to_boundary(h3_index)  # often returns list of (lat, lon)
            # Convert to (lon, lat) before creating Polygon
            boundary_lonlat = [(pt[1], pt[0]) for pt in boundary]
            return sg.Polygon(boundary_lonlat)
        except Exception:
            pass

    # try alternative names
    for fname in ["cell_to_vertexes", "cell_to_vertex", "cell_to_latlng", "cell_to_boundary_geo"]:
        if hasattr(h3, fname):
            try:
                boundary = getattr(h3, fname)(h3_index)
                boundary_lonlat = [(pt[1], pt[0]) for pt in boundary]
                return sg.Polygon(boundary_lonlat)
            except Exception:
                pass

    # fallback: try api namespace
    try:
        if hasattr(h3, "api"):
            api_mod = h3.api
            for attr in dir(api_mod):
                sub = getattr(api_mod, attr)
                if hasattr(sub, "cell_to_boundary"):
                    try:
                        boundary = sub.cell_to_boundary(h3_index)
                        boundary_lonlat = [(pt[1], pt[0]) for pt in boundary]
                        return sg.Polygon(boundary_lonlat)
                    except Exception:
                        pass
    except Exception:
        pass

    # if nothing works, return None
    return None

# ----------------------------------------------------------------------
# PART 1: OSMNX NETWORK EXTRACTION (auto-detect & robust)
# ----------------------------------------------------------------------
print("\n--- Part 1: OSMnx Network Extraction and Projection ---")
try:
    place_gdf = ox.geocode_to_gdf(PLACE_NAME)
    minx, miny, maxx, maxy = place_gdf.total_bounds
    west, south, east, north = minx, miny, maxx, maxy

    lat_margin = BUFFER_DIST / 111000.0
    lon_margin = lat_margin / np.cos(np.deg2rad((north + south) / 2.0))

    north_b = north + lat_margin
    south_b = south - lat_margin
    east_b = east + lon_margin
    west_b = west - lon_margin

    buffered_polygon_unproj = sg.box(west_b, south_b, east_b, north_b)

    # Choose method based on osmnx version; tolerant fallbacks
    ox_ver = ox.__version__
    major_version = int(ox_ver.split('.')[0]) if ox_ver and ox_ver.split('.')[0].isdigit() else None

    G_nx = None
    used_method = None

    if major_version is not None and major_version >= 2:
        try:
            # Try keyword bbox first (OSMnx >=2.0 expects keywords)
            G_nx = ox.graph_from_bbox(north=north_b, south=south_b, east=east_b, west=west_b,
                                      network_type=NETWORK_TYPE, retain_all=True)
            used_method = "graph_from_bbox (keyword args for osmnx >=2)"
        except Exception as e:
            print("Keyword bbox call failed for osmnx >=2. Trying fallbacks. Error:", e)

    if G_nx is None:
        # Try positional (older behavior)
        try:
            G_nx = ox.graph_from_bbox(north_b, south_b, east_b, west_b, NETWORK_TYPE, retain_all=True)
            used_method = "graph_from_bbox (positional args)"
        except Exception as e:
            print("Positional bbox call failed. Error:", e)

    if G_nx is None:
        # Most robust fallback
        print("Falling back to graph_from_place(PLACE_NAME) ...")
        G_nx = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE, retain_all=True)
        used_method = "graph_from_place (fallback)"

    print(f"Graph extraction succeeded using: {used_method}")
    print(f"Nodes (raw graph): {len(G_nx.nodes):,}, Edges (raw graph): {len(G_nx.edges):,}")

except Exception as e:
    print("Error during Network Extraction:", e)
    traceback.print_exc()
    raise

# Project graph & convert to gdfs
G_proj = ox.project_graph(G_nx)
proj_crs = G_proj.graph.get('crs', None)
print("Projected graph CRS:", proj_crs)

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
print(f"Nodes GDF shape: {nodes_gdf.shape}, Edges GDF shape: {edges_gdf.shape}")

# ----------------------------------------------------------------------
# PART 2: POI FEATURE GENERATION
# ----------------------------------------------------------------------
print("\n--- Part 2: POI Feature Generation ---")
poi_tags = {
    "amenity": ["hospital", "school", "university", "bank", "restaurant", "cafe"],
    "leisure": ["park", "garden", "playground", "stadium"],
    "public_transport": ["metro_station", "bus_station", "stop_position"],
    "building": ["commercial", "office", "residential"]
}

pois_gdf = ox.features.features_from_polygon(buffered_polygon_unproj, tags=poi_tags)
pois_gdf = pois_gdf[pois_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].reset_index(drop=True)
if pois_gdf.empty:
    pois_gdf = gpd.GeoDataFrame(columns=['amenity', 'leisure', 'public_transport', 'building', 'geometry'], geometry='geometry', crs='EPSG:4326')

pois_proj = pois_gdf.to_crs(edges_gdf.crs)
print(f"POI GeoDataFrame shape (Points only): {pois_proj.shape}")

# ----------------------------------------------------------------------
# PART 3: H3 GRID CREATION & FEATURE FUSION (using polygon_to_cells & cell_to_boundary)
# ----------------------------------------------------------------------
print("\n--- Part 3: H3 Area Node Generation & Feature Fusion ---")
minx_p, miny_p, maxx_p, maxy_p = edges_gdf.total_bounds
bbox_polygon = sg.box(minx_p, miny_p, maxx_p, maxy_p)

# convert projected bbox to WGS84 geojson-like polygon (for polygon_to_cells)
bbox_poly_unproj = gpd.GeoSeries([bbox_polygon], crs=edges_gdf.crs).to_crs('EPSG:4326').iloc[0]

# use polygon_to_cells to get H3 indexes (robust wrapper)
try:
    h3_indexes = h3_polyfill_with_polygon_to_cells(bbox_poly_unproj.__geo_interface__, res=H3_RESOLUTION)
    print(f"H3 polygon_to_cells produced {len(h3_indexes)} cells.")
except Exception as exc:
    print("H3 polygon_to_cells failed:", exc)
    traceback.print_exc()
    raise

# Build H3 polygons using cell_to_boundary
h3_polygons = []
skipped = 0
for hid in h3_indexes:
    poly = h3_cell_boundary_to_polygon(hid)
    if poly is None or not poly.is_valid:
        skipped += 1
        continue
    h3_polygons.append({'h3_index': hid, 'geometry': poly})

if skipped > 0:
    print(f"Skipped {skipped} H3 cells due to inability to build polygon boundaries.")

h3_gdf = gpd.GeoDataFrame(h3_polygons, crs='EPSG:4326')
h3_gdf = h3_gdf[h3_gdf.intersects(bbox_poly_unproj)].reset_index(drop=True)
h3_gdf = h3_gdf.to_crs(edges_gdf.crs)
print(f"H3 Area Nodes GeoDataFrame created. Shape: {h3_gdf.shape}")

# Simulated socio-demographic features
if not nodes_gdf.empty:
    city_center = nodes_gdf.geometry.unary_union.centroid
else:
    city_center = h3_gdf.geometry.unary_union.centroid

h3_centroids = h3_gdf.geometry.centroid
distance_to_center = h3_centroids.distance(city_center)
max_dist = distance_to_center.max() if len(distance_to_center) > 0 else 1.0

h3_gdf['sim_pop_density'] = ((max_dist - distance_to_center) / max_dist).clip(0) ** 2 * 20000 + 1000
h3_gdf['sim_median_income'] = ((max_dist - distance_to_center) / max_dist).clip(0) * 100000 + np.random.normal(50000, 10000, size=len(h3_gdf))
zoning_categories = ['Residential_High', 'Residential_Low', 'Commercial', 'Industrial', 'Green_Belt']
h3_gdf['sim_zoning'] = np.random.choice(zoning_categories, size=len(h3_gdf), p=[0.4, 0.3, 0.15, 0.1, 0.05])
h3_gdf = pd.concat([h3_gdf.reset_index(drop=True), pd.get_dummies(h3_gdf['sim_zoning'], prefix='zoning').reset_index(drop=True)], axis=1)

# POI counts (spatial join)
if not pois_proj.empty and not h3_gdf.empty:
    for col in ['amenity', 'leisure', 'public_transport']:
        if col not in pois_proj.columns:
            pois_proj[col] = None

    h3_nodes_with_poi = gpd.sjoin(h3_gdf, pois_proj[['amenity', 'leisure', 'public_transport', 'geometry']], how="left", predicate="contains", lsuffix='h3', rsuffix='poi')

    poi_counts = h3_nodes_with_poi.groupby('h3_index').agg(
        poi_park_count=('leisure', lambda x: (x.isin(['park', 'garden'])).sum() if len(x) > 0 else 0),
        poi_hospital_count=('amenity', lambda x: (x == 'hospital').sum() if len(x) > 0 else 0),
        poi_transit_stop_count=('public_transport', lambda x: x.count() if len(x) > 0 else 0)
    ).reset_index()

    h3_gdf = h3_gdf.merge(poi_counts, on='h3_index', how='left').fillna({'poi_park_count': 0, 'poi_hospital_count': 0, 'poi_transit_stop_count': 0})
else:
    h3_gdf['poi_park_count'] = 0
    h3_gdf['poi_hospital_count'] = 0
    h3_gdf['poi_transit_stop_count'] = 0

print(f"H3 Area Nodes now have {h3_gdf.drop(columns=['geometry']).shape[1]} static features (including one-hot zoning & POI counts).")

# ----------------------------------------------------------------------
# PART 4: EDGE FEATURE ENRICHMENT
# ----------------------------------------------------------------------
print("\n--- Part 4: Edge Feature Enrichment ---")
if 'highway' not in edges_gdf.columns:
    edges_gdf['highway'] = None

highway_types = ['motorway', 'primary', 'secondary', 'tertiary', 'residential', 'service']
def highway_flag(val, hwy):
    if isinstance(val, str):
        return 1 if val == hwy else 0
    if isinstance(val, (list, tuple)):
        return 1 if hwy in val else 0
    return 0

for hwy in highway_types:
    edges_gdf[f'hwy_{hwy}'] = edges_gdf['highway'].apply(lambda x, h=hwy: highway_flag(x, h))

edges_gdf['has_bus_lane'] = np.random.choice([0, 1], size=len(edges_gdf), p=[0.9, 0.1])
if 'oneway' not in edges_gdf.columns:
    edges_gdf['oneway'] = False
edges_gdf['is_one_way'] = edges_gdf['oneway'].apply(lambda x: 1 if x is True else 0)
edges_gdf['edge_length_m'] = edges_gdf.get('length', pd.Series(np.nan, index=edges_gdf.index))

final_edge_features = ['edge_length_m', 'is_one_way', 'has_bus_lane'] + [f'hwy_{hwy}' for hwy in highway_types]
print(f"Edges GDF now has {len(final_edge_features)} static features (listed).")

# ----------------------------------------------------------------------
# FINAL OUTPUT & VISUAL VERIFICATION
# ----------------------------------------------------------------------
print("\n--- Final Data Structures ---")
print("1. Nodes GDF (Traffic Intersections):", nodes_gdf.shape)
print("2. Edges GDF (Road Segments):", edges_gdf.shape)
print("3. H3 GDF (Area Nodes for Multimodal Features):", h3_gdf.shape)

# Plot projected graph + H3 + hospitals
fig, ax = ox.plot_graph(G_proj, node_size=0, edge_linewidth=0.5, edge_color="gray", show=False, close=False, bgcolor="w")
h3_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.1, alpha=0.5, label='H3 Area Nodes')
if 'amenity' in pois_proj.columns:
    hosp = pois_proj[pois_proj['amenity'] == 'hospital']
    if not hosp.empty:
        hosp.plot(ax=ax, color='red', marker='X', markersize=20, label='Hospitals')

plt.title(f"Hybrid Graph Foundation (Road Network + H3 Grid) for {PLACE_NAME}")
plt.legend()
plt.show()

print("\n--- Phase 1, Step 1 Complete ---")
