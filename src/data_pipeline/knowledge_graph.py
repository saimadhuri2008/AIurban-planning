#!/usr/bin/env python3
"""
build_knowledge_graph.py

Construct a Bengaluru Urban Knowledge Graph from your processed master files.

Outputs:
 - graph_graphml:   data/processed/master/bengaluru_urban_graph.graphml
 - neo4j CSVs:      data/processed/master/neo4j_nodes.csv, neo4j_rels.csv
 - summary JSON:    data/processed/master/graph_summary.json

Run:
    python src/build_knowledge_graph.py
"""

import os
from pathlib import Path
import json
import logging
import math

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("kg_builder")

# ---------- UPDATE FILE PATHS IF NEEDED ----------
BASE = Path(r"C:\AIurban-planning")
MASTER_DIR = BASE / "data" / "processed" / "master"

PATHS = {
    "h3_master": MASTER_DIR / "h3_master.parquet",
    "wards_master": MASTER_DIR / "wards_master.parquet",
    "fused": MASTER_DIR / "master_fused.parquet",
    # fallback sources
    "roads": BASE / "outputs" / "roads_enriched.geojson",
    "buildings": BASE / "data" / "processed" / "buildings_processed.ndjson",
    "schools": BASE / "outputs" / "schools_h3_res8.geojson",
    "health": BASE / "outputs" / "health_h3_res8.geojson",
    "metro": BASE / "data" / "processed" / "metro_stations_enriched.csv",
    "electricity": BASE / "data" / "processed" / "electricty" / "electricity_assets.geojson",
    "od": BASE / "data" / "processed" / "od_flows.csv",
}

OUT_DIR = MASTER_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def safe_read_parquet(p):
    if p.exists():
        return pd.read_parquet(p)
    LOG.warning("Missing parquet: %s", p)
    return None

def safe_read_geo(p):
    if p.exists():
        return gpd.read_file(p)
    LOG.warning("Missing geo: %s", p)
    return None

def safe_read_table(p):
    if not p.exists():
        LOG.warning("Missing table: %s", p)
        return None

    ext = p.suffix.lower()

    # NDJSON / JSONLINES
    if ext in [".ndjson", ".jsonl"]:
        try:
            return pd.read_json(p, lines=True)
        except Exception as e:
            LOG.error("Failed reading NDJSON %s: %s", p, e)
            return None

    # CSV
    if ext == ".csv":
        try:
            return pd.read_csv(p)
        except Exception as e:
            LOG.error("Failed reading CSV %s: %s", p, e)
            return None

    # Parquet
    if ext == ".parquet":
        try:
            return pd.read_parquet(p)
        except Exception as e:
            LOG.error("Failed reading Parquet %s: %s", p, e)
            return None

    # Fallback: attempt JSON
    if ext in [".json", ".geojson"]:
        try:
            return gpd.read_file(p)
        except Exception:
            try:
                return pd.read_json(p)
            except Exception as e:
                LOG.error("Failed reading JSON %s: %s", p, e)
                return None

    # If unknown extension, fallback to CSV
    try:
        return pd.read_csv(p)
    except Exception as e:
        LOG.error("Unknown format %s: %s", p, e)
        return None


# ---------- load data ----------
LOG.info("Loading inputs...")
h3_df = safe_read_parquet(PATHS["h3_master"])
wards_df = safe_read_parquet(PATHS["wards_master"])
fused_df = safe_read_parquet(PATHS["fused"])
roads_gdf = safe_read_geo(PATHS["roads"])
buildings_gdf = safe_read_table(PATHS["buildings"])  # may be ndjson - pandas can read it as JSON lines if necessary
schools_gdf = safe_read_geo(PATHS["schools"])
health_gdf = safe_read_geo(PATHS["health"])
elec_gdf = safe_read_geo(PATHS["electricity"])
od_df = safe_read_table(PATHS["od"])
metro_df = safe_read_table(PATHS["metro"])

# convert any GeoDataFrames to GeoPandas if needed
if isinstance(buildings_gdf, pd.DataFrame) and "geometry" in buildings_gdf.columns:
    try:
        buildings_gdf = gpd.GeoDataFrame(buildings_gdf, geometry="geometry", crs="EPSG:4326")
    except Exception:
        pass

# Ensure basic columns exist
if h3_df is not None:
    h3_df["h3_index"] = h3_df["h3_index"].astype(str)

if wards_df is not None:
    wards_df["ward_id"] = wards_df["ward_id"].astype(str)

# ---------- Build Graph ----------
LOG.info("Initializing graphs...")
G = nx.Graph()          # spatial & attribute graph (undirected)
G_flow = nx.DiGraph()   # directed graph for OD flows (weights)

# Helper to add Ward nodes
if wards_df is not None:
    LOG.info("Adding Ward nodes...")
    for _, r in wards_df.iterrows():
        wid = r.get("ward_id")
        attrs = {
            "node_type": "Ward",
            "ward_num": r.get("ward_num") or r.get("ward_num_x") or None,
            "ward_name": r.get("ward_name") or r.get("ward_name_x") or None,
            "zone_name": r.get("zone_name") or r.get("zone_name_x") or None,
            "population_est": float(r.get("population_est")) if pd.notna(r.get("population_est")) else None,
            "area_sqkm": float(r.get("area_sqkm") or r.get("area_sqkm_x") or r.get("area_sqkm_y") or None),
        }
        # optionally include centroid if present
        lat = r.get("centroid_lat") or r.get("centroid_lat_x") or r.get("centroid_lat_y")
        lon = r.get("centroid_lon") or r.get("centroid_lon_x") or r.get("centroid_lon_y")
        if pd.notna(lat) and pd.notna(lon):
            attrs["centroid_lat"] = float(lat); attrs["centroid_lon"] = float(lon)
        G.add_node(wid, **attrs)

# H3 nodes
if h3_df is not None:
    LOG.info("Adding H3 nodes...")
    for _, r in h3_df.iterrows():
        h = str(r.get("h3_index"))
        attrs = {
            "node_type": "H3Cell",
            "ward_id": r.get("ward_id"),
            "population_h3": float(r.get("population_h3")) if pd.notna(r.get("population_h3")) else None,
            "built_area_m2": float(r.get("built_area_m2")) if pd.notna(r.get("built_area_m2")) else None,
            "it_job_density": float(r.get("it_job_density")) if pd.notna(r.get("it_job_density")) else None,
            "income_index": float(r.get("income_index")) if pd.notna(r.get("income_index")) else None,
        }
        # geometry centroid from fused or h3 df if present in geometry column
        try:
            if "geometry" in h3_df.columns:
                geom = r.geometry
                if geom is not None and not geom.is_empty:
                    attrs["centroid_lat"] = float(geom.centroid.y)
                    attrs["centroid_lon"] = float(geom.centroid.x)
        except Exception:
            pass
        G.add_node(h, **attrs)
        # H3 -> Ward relation (if ward_id exists)
        if r.get("ward_id") and pd.notna(r.get("ward_id")):
            G.add_edge(h, r.get("ward_id"), relation="IN_WARD")
            G_flow.add_node(h, node_type="H3Cell")  # ensure exists in flow graph

# Add counts-type nodes (schools, health, parks as nodes), linking to ward & h3 if available
def add_point_nodes(gdf, node_label):
    if gdf is None:
        return
    LOG.info("Adding %s nodes...", node_label)
    # if gdf has h3_index or ward_id, use them; otherwise, use geometry centroid to map
    for idx, r in gdf.iterrows():
        nid = f"{node_label}:{idx}"
        attrs = {"node_type": node_label}
        # common attributes
        for col in ["name", "type", "capacity", "address"]:
            if col in r.index:
                attrs[col] = r.get(col)
        # geometry centroids
        if hasattr(r, "geometry") and r.geometry is not None and not r.geometry.is_empty:
            pt = r.geometry.centroid
            attrs["centroid_lat"] = float(pt.y)
            attrs["centroid_lon"] = float(pt.x)
        # add node
        G.add_node(nid, **attrs)
        # try connect to h3 or ward
        if "h3_index" in r.index and pd.notna(r.get("h3_index")):
            G.add_edge(nid, str(r.get("h3_index")), relation=f"IN_{node_label}_H3")
        if "ward_id" in r.index and pd.notna(r.get("ward_id")):
            G.add_edge(nid, r.get("ward_id"), relation=f"IN_{node_label}_WARD")

add_point_nodes(schools_gdf, "School")
add_point_nodes(health_gdf, "HealthFacility")
add_point_nodes(elec_gdf, "ElectricityAsset")

# Roads: create road nodes and optionally connect road adjacency via spatial touches or endpoints
if roads_gdf is not None:
    LOG.info("Adding RoadSegment nodes...")
    # ensure an id
    roads_gdf = roads_gdf.reset_index().rename(columns={"index": "road_idx"})
    for _, r in roads_gdf.iterrows():
        rid = f"Road:{int(r['road_idx'])}"
        attrs = {"node_type": "RoadSegment"}
        # attach useful attributes if exist
        for c in ["road_type", "length_m", "speed_kmph", "congestion_index", "name"]:
            if c in r.index:
                val = r.get(c)
                if pd.notna(val):
                    attrs[c] = val
        # geometry centroid if present
        if hasattr(r, "geometry") and r.geometry is not None and not r.geometry.is_empty:
            try:
                pt = r.geometry.centroid
                attrs["centroid_lat"] = float(pt.y)
                attrs["centroid_lon"] = float(pt.x)
            except Exception:
                pass
        G.add_node(rid, **attrs)
        # connect road to ward(s) by spatial join; try ward membership if ward geometry exists
        if "ward_id" in r.index and pd.notna(r.get("ward_id")):
            G.add_edge(rid, r.get("ward_id"), relation="OVERLAPS_WARD")
        # if h3 index exists
        if "h3_index" in r.index and pd.notna(r.get("h3_index")):
            G.add_edge(rid, str(r.get("h3_index")), relation="IN_H3")

    # optional: connect adjacent roads by spatial intersection (costly for many segments)
    LOG.info("Connecting nearby road segments (spatial k-nearest, approximate)...")
    try:
        # build centroids
        roads_gdf["centroid"] = roads_gdf.geometry.centroid
        centroids = list(roads_gdf["centroid"].values)
        ids = [f"Road:{int(i)}" for i in roads_gdf["road_idx"].values]
        # naive O(n^2) for small city; if too big, use rtree / spatial index
        for i in range(len(ids)):
            for j in range(i+1, min(i+6, len(ids))):  # cheaply connect a few closest by index (fast heuristic)
                a = centroids[i]; b = centroids[j]
                try:
                    d = a.distance(b)
                    if d < 0.01:  # ~ about 1km threshold depends on CRS
                        G.add_edge(ids[i], ids[j], relation="CONNECTS_TO", distance=float(d))
                except Exception:
                    continue
    except Exception as e:
        LOG.warning("Road adjacency linking skipped: %s", e)

# OD Flows: build directed edges between H3 cells
if od_df is not None:
    LOG.info("Adding OD flow edges to flow graph...")
    # guess columns: origin_h3, destination_h3, od_flow / flow
    cols = set(od_df.columns.str.lower())
    possible_orig = next((c for c in od_df.columns if "origin" in c.lower() and "h3" in c.lower()), None)
    possible_dest = next((c for c in od_df.columns if "dest" in c.lower() and "h3" in c.lower()), None)
    possible_flow = next((c for c in od_df.columns if "flow" in c.lower()), None)
    if possible_orig and possible_dest and possible_flow:
        for _, r in od_df.iterrows():
            o = str(r[possible_orig]); d = str(r[possible_dest]); w = r[possible_flow]
            try:
                w = float(w)
            except Exception:
                w = 1.0
            if o and d:
                G_flow.add_edge(o, d, weight=w, relation="OD_FLOW")
    else:
        LOG.warning("OD table lacks origin/destination/flow columns; skipping OD edges")

# Ward adjacency (NEARBY) by centroid distance (threshold)
def connect_wards_by_distance(max_km=1.5):
    # compute centroid coords
    if wards_df is None:
        return
    coords = []
    ids = []
    for _, r in wards_df.iterrows():
        wid = r.get("ward_id")
        lat = r.get("centroid_lat") or r.get("centroid_lat_x") or r.get("centroid_lat_y")
        lon = r.get("centroid_lon") or r.get("centroid_lon_x") or r.get("centroid_lon_y")
        if pd.notna(lat) and pd.notna(lon):
            coords.append((float(lat), float(lon))); ids.append(wid)
    # simple O(n^2) for wards (~200); okay
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            lat1, lon1 = coords[i]; lat2, lon2 = coords[j]
            # haversine approx in km
            R = 6371.0
            phi1 = math.radians(lat1); phi2 = math.radians(lat2)
            dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
            dist = 2*R*math.asin(math.sqrt(a))
            if dist <= max_km:
                G.add_edge(ids[i], ids[j], relation="NEARBY", distance_km=dist)

connect_wards_by_distance(max_km=1.5)

# ---------- Compute some metrics ----------
LOG.info("Computing basic graph metrics...")
summary = {}
summary['num_nodes'] = int(G.number_of_nodes())
summary['num_edges'] = int(G.number_of_edges())
# degree distribution (for wards & h3)
deg = dict(G.degree())
summary['avg_degree'] = sum(deg.values()) / len(deg) if len(deg) > 0 else 0

# centrality for H3 cells (small selection)
try:
    h3_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "H3Cell"]
    if len(h3_nodes) > 0:
        sub = G.subgraph(h3_nodes)
        pr = nx.pagerank_numpy(sub) if len(sub) < 5000 else {}
        # annotate top 10
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
        summary['top_h3_pagerank'] = top_pr
        # set attribute on nodes if desired
        for n, v in pr.items():
            G.nodes[n]['pagerank'] = float(v)
except Exception as e:
    LOG.warning("Pagerank skipped: %s", e)

# Ward degree (for adjacency)
try:
    ward_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Ward"]
    degs = {w: G.degree(w) for w in ward_nodes}
    top_wards = sorted(degs.items(), key=lambda x: x[1], reverse=True)[:10]
    summary['top_wards_by_degree'] = top_wards
except Exception:
    pass

# ---------- Export graph files ----------
LOG.info("Exporting GraphML and Neo4j CSVs...")

# GraphML (preserves node attributes but can be large)
graphml_path = OUT_DIR / "bengaluru_urban_graph.graphml"
try:
    nx.write_graphml(G, graphml_path)
    LOG.info("Wrote GraphML: %s", graphml_path)
except Exception as e:
    LOG.exception("GraphML write failed: %s", e)

# Export Neo4j-friendly CSVs: nodes, relationships
nodes_out = OUT_DIR / "neo4j_nodes.csv"
rels_out = OUT_DIR / "neo4j_rels.csv"

# Prepare node rows: id, labels, properties as JSON string or as separate columns
node_rows = []
for n, attrs in G.nodes(data=True):
    label = attrs.get("node_type", "Entity")
    props = {k: v for k,v in attrs.items() if k != "node_type"}
    # flatten common scalar props into columns, and store the rest in props_json
    node_rows.append({
        "node_id": n,
        "label": label,
        "props_json": json.dumps(props)
    })
pd.DataFrame(node_rows).to_csv(nodes_out, index=False)
LOG.info("Wrote Neo4j nodes csv: %s", nodes_out)

# relationships
rel_rows = []
for u, v, attrs in G.edges(data=True):
    rel_rows.append({
        "start_id": u,
        "end_id": v,
        "type": attrs.get("relation", "RELATED_TO"),
        "props_json": json.dumps({k:v for k,v in attrs.items() if k != "relation"})
    })
pd.DataFrame(rel_rows).to_csv(rels_out, index=False)
LOG.info("Wrote Neo4j rels csv: %s", rels_out)

# flow graph export (OD) — separate file
od_out = OUT_DIR / "od_edges.csv"
if G_flow.number_of_edges() > 0:
    flow_rows = []
    for u, v, a in G_flow.edges(data=True):
        flow_rows.append({"origin": u, "destination": v, "flow": a.get("weight", 1.0)})
    pd.DataFrame(flow_rows).to_csv(od_out, index=False)
    LOG.info("Wrote OD flows csv: %s", od_out)
else:
    LOG.info("No OD flow edges to export")

# summary
summary_path = OUT_DIR / "graph_summary.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
LOG.info("Wrote summary: %s", summary_path)

LOG.info("Knowledge graph build complete. Nodes: %d Edges: %d", G.number_of_nodes(), G.number_of_edges())
