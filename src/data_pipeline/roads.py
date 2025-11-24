#!/usr/bin/env python3
"""
roads.py

Build road network edges GeoDataFrame, compute approximate centralities (fast),
and export GeoJSON + CSV.

Usage:
    python roads.py --wards <wards.geojson> --kml1 <bengaluru-roadways.kml> --kml2 <complete_roads.kml> --outdir ./outputs --k-sample 400

Defaults attempt to use OSMnx; fallback to user KMLs if OSMnx not available / failing.
"""

import argparse
from pathlib import Path
import logging
import sys
import math
import json
import time
import fiona
from fiona.crs import from_epsg
import math, gc

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely import force_2d      # <-- Added for flattening 3D KML
import networkx as nx
import numpy as np

# OSMnx optional
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except Exception:
    OSMNX_AVAILABLE = False

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("roads")

# default paths
DEFAULT_WARDS = Path(r"C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson")
# DEFAULT_KML1 = Path("data/raw/bengaluru-roadways.kml")
DEFAULT_KML2 = Path("/data/raw/complete_roads.kml")

# ------------------------
# helpers
# ------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_read_wards(path: Path):
    if not path.exists():
        log.warning("Wards file not found: %s", path)
        return None
    g = gpd.read_file(str(path))
    try:
        g = g.to_crs(epsg=4326)
    except Exception:
        pass
    # pick ward id
    ward_col = next((c for c in g.columns if c.lower().startswith("ward")), None)
    if ward_col is None:
        ward_col = g.columns[0]
        g = g.rename(columns={ward_col: "ward_id"})
        ward_col = "ward_id"
    else:
        if ward_col != "ward_id":
            g = g.rename(columns={ward_col: "ward_id"})
            ward_col = "ward_id"
    g["ward_id"] = g["ward_id"].astype(str)
    return g

def read_kml_layers(paths):
    """Read KML(s) and return concatenated GeoDataFrame of LineString geometries where possible."""
    frames = []
    for p in paths:
        if p is None or not Path(p).exists():
            continue
        p = Path(p)
        try:
            g = gpd.read_file(str(p))
            if g is None or g.empty:
                continue

            # Keep only line geometries
            g = g[g.geometry.type.isin(["LineString", "MultiLineString", "GeometryCollection", "MultiGeometry"]) |
                  g.geometry.type.str.contains("Line")]

            if g.empty:
                continue

            # ---- FIX: force 2D coordinates everywhere ----
            g["geometry"] = g.geometry.apply(force_2d)

            frames.append(g)

        except Exception as e:
            log.warning("Failed to read KML %s: %s", p, e)

    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    out = pd.concat(frames, ignore_index=True)
    try:
        out = out.to_crs(epsg=4326)
    except Exception:
        pass

    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def linestrings_from_gdf(gdf):
    """Explode MultiLine to single LineStrings and return GeoDataFrame (lon/lat order)."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    gdf = gdf.reset_index(drop=True)

    def to_ls(geom):
        if geom.geom_type == "MultiLineString":
            coords = []
            for part in geom.geoms:
                coords.extend(list(part.coords))
            return LineString(coords)
        return geom

    # ---- FIX: ensure 2D after explode ----
    gdf["geometry"] = gdf.geometry.apply(force_2d).apply(to_ls)

    gdf = gdf.set_geometry("geometry")
    gdf.crs = "EPSG:4326"
    return gdf

def project_to_metric(gdf):
    if gdf is None or gdf.empty:
        return gdf
    try:
        return gdf.to_crs(epsg=3857)
    except Exception:
        try:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
            return gdf.to_crs(epsg=3857)
        except Exception:
            return gdf

def line_length_m(ls_geom):
    try:
        return float(ls_geom.length)
    except Exception:
        return float(0.0)

# ------------------------
# Graph build / fallback
# ------------------------
def build_graph_from_osm(wards_gdf, buffer_m=500):
    if not OSMNX_AVAILABLE:
        log.info("OSMnx not available -> skipping OSM extraction.")
        return None
    try:
        union = wards_gdf.unary_union if hasattr(wards_gdf, "unary_union") else wards_gdf.geometry.unary_union
        poly = union.buffer(0.0001)
        log.info("Attempting OSMnx graph extraction for bounding polygon...")
        G = ox.graph_from_polygon(poly, network_type="drive", simplify=True)
        Gu = ox.utils_graph.get_undirected(G) if hasattr(ox.utils_graph, "get_undirected") else nx.Graph(G)
        return Gu
    except Exception as e:
        log.warning("OSMnx extraction failed: %s", e)
        return None


def build_graph_from_kml_lines(lines_gdf, snap_tol_m=50):
    """Construct graph from KML lines (fallback)."""
    log.info("Building fallback graph from KML lines...")

    lines_proj = project_to_metric(lines_gdf.copy())
    nodes = {}
    edges = []

    # ---- FIXED round_xy to support 2D/3D ----
    def round_xy(xy):
        """
        Accepts (x, y) or (x, y, z) and returns a snapped (x, y).
        """
        x, y = xy[:2]    # ignore z if present
        snap = max(1.0, float(snap_tol_m))
        rx = round(x / snap) * snap
        ry = round(y / snap) * snap
        return (rx, ry)

    for idx, row in lines_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.coords)
        except Exception:
            try:
                coords = []
                for part in geom.geoms:
                    coords.extend(list(part.coords))
            except Exception:
                continue

        if len(coords) < 2:
            continue

        for i in range(len(coords)-1):
            a = coords[i]
            b = coords[i+1]
            an = round_xy(a)
            bn = round_xy(b)

            edges.append({
                "u_xy": an,
                "v_xy": bn,
                "coords": (a, b),
                "orig_idx": idx
            })

    uniq = {}
    id_counter = 0

    for e in edges:
        for k in ("u_xy", "v_xy"):
            xy = e[k]
            if xy not in uniq:
                uniq[xy] = id_counter
                id_counter += 1

    G = nx.Graph()

    nodes_list = [{"nid": nid, "x": xy[0], "y": xy[1]} for xy, nid in uniq.items()]
    nodes_df = pd.DataFrame(nodes_list)

    nodes_gdf = gpd.GeoDataFrame(nodes_df,
                                 geometry=gpd.points_from_xy(nodes_df["x"], nodes_df["y"]),
                                 crs="EPSG:3857")

    try:
        nodes_gdf = nodes_gdf.to_crs(epsg=4326)
    except Exception:
        pass

    for _, r in nodes_gdf.iterrows():
        nid = int(r["nid"])
        G.add_node(nid,
                   x_proj=float(r["x"]),
                   y_proj=float(r["y"]),
                   lon=float(r.geometry.x),
                   lat=float(r.geometry.y))

    edge_id = 0
    for e in edges:
        u = uniq[e["u_xy"]]; v = uniq[e["v_xy"]]
        ux, uy = e["u_xy"]; vx, vy = e["v_xy"]
        length = math.hypot(vx - ux, vy - uy)

        urow = nodes_gdf[nodes_gdf["nid"] == u].iloc[0]
        vrow = nodes_gdf[nodes_gdf["nid"] == v].iloc[0]
        geom = LineString([(urow.geometry.x, urow.geometry.y),
                           (vrow.geometry.x, vrow.geometry.y)])

        G.add_edge(u, v, key=edge_id, length_m=float(length), geometry=geom)
        edge_id += 1

    log.info("Built fallback graph: nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G



# ------------------------
# Centrality computations (approx)
# ------------------------
def compute_approx_centralities(G, k_sample=400, weight_attr="length_m"):
    """
    Compute approximate betweenness centrality using k random sources.
    Also computes degree centrality and sampled closeness (approx).
    Returns:
      node_betweenness (dict), node_closeness (dict), degree_centrality (dict)
    """
    n_nodes = G.number_of_nodes()
    k = int(min(k_sample, max(10, n_nodes // 10)))
    log.info("Computing approximate centralities with k_sample=%d (effective k=%d) on %d nodes", k_sample, k, n_nodes)

    # choose random seed nodes (deterministic)
    nodes = list(G.nodes())
    rng = np.random.RandomState(42)
    if k >= n_nodes:
        sources = nodes
    else:
        sources = list(rng.choice(nodes, size=k, replace=False))

    # Use NetworkX betweenness_centrality with k sources (fast approx)
    try:
        start = time.time()
        # if weight_attr present in edges, pass weight; else None
        weight = weight_attr if any(weight_attr in d for u,v,d in G.edges(data=True)) else None
        bet = nx.betweenness_centrality(G, k=sources, normalized=True, weight=weight, endpoints=False, seed=42)
        dur = time.time() - start
        log.info("Betweenness computed (approx) in %.1fs", dur)
    except Exception as e:
        log.warning("Approx betweenness computation failed: %s. Falling back to zero array.", e)
        bet = {n: 0.0 for n in G.nodes()}

    # degree centrality (fast)
    deg = nx.degree_centrality(G)

    # approximate closeness: compute shortest path lengths from k sample nodes and invert avg distance
    try:
        start = time.time()
        # run multi-source shortest_path_length and approximate closeness per node
        sample_nodes = sources if len(sources) >= 5 else list(G.nodes())[:min(50, n_nodes)]
        # accumulate distances
        dist_acc = {n: 0.0 for n in G.nodes()}
        count_acc = {n: 0 for n in G.nodes()}
        for s in sample_nodes:
            lengths = nx.single_source_dijkstra_path_length(G, s, weight=weight) if weight else nx.single_source_shortest_path_length(G, s)
            for n, d in lengths.items():
                dist_acc[n] += d
                count_acc[n] += 1
        clos = {}
        for n in G.nodes():
            if count_acc[n] == 0:
                clos[n] = 0.0
            else:
                avgd = dist_acc[n] / count_acc[n]
                clos[n] = 0.0 if avgd == 0 else 1.0 / avgd
        dur = time.time() - start
        log.info("Sampled closeness computed in %.1fs", dur)
        # normalize clos to 0..1
        vals = np.array(list(clos.values()), dtype=float)
        if np.nanmax(vals) - np.nanmin(vals) > 0:
            mn = np.nanmin(vals); mx = np.nanmax(vals)
            for n in clos:
                clos[n] = (clos[n] - mn) / (mx - mn)
        else:
            for n in clos:
                clos[n] = 0.0
    except Exception as e:
        log.warning("Approx closeness failed: %s", e)
        clos = {n: 0.0 for n in G.nodes()}

    return bet, clos, deg

# ------------------------
# Export functions
# ------------------------
def graph_edges_to_gdf(G):
    """Convert graph edges to GeoDataFrame with useful attributes."""
    rows = []
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry", None)
        if geom is None:
            # try construct geometry from node coords stored in node attrs
            nu = G.nodes[u]
            nv = G.nodes[v]
            try:
                geom = LineString([(nu["lon"], nu["lat"]), (nv["lon"], nv["lat"])])
            except Exception:
                geom = None
        length_m = float(data.get("length_m", 0.0))
        road_type = data.get("highway", data.get("road_type", None))
        lanes = data.get("lanes", None)
        speed = data.get("maxspeed", data.get("speed_kmph", None))
        one_way = data.get("oneway", data.get("is_oneway", False))
        rows.append({
            "u": u,
            "v": v,
            "length_m": length_m,
            "road_type": road_type,
            "lanes": lanes,
            "speed_limit": speed,
            "oneway": one_way,
            "geometry": geom
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf

# ------------------------
# Main
# ------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wards", default=str(DEFAULT_WARDS))
   # p.add_argument("--kml1", default=str(DEFAULT_KML1))
    p.add_argument("--kml2", default=str(DEFAULT_KML2))
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--k-sample", type=int, default=400, help="k sample size for approx betweenness")
    p.add_argument("--snap-tol-m", type=float, default=50.0, help="snap tolerance (meters) when building graph from KML")
    args = p.parse_args()

    wards_path = Path(args.wards)
    kml_paths = [Path(args.kml2)]
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    wards_gdf = None
    if wards_path.exists():
        wards_gdf = safe_read_wards(wards_path)
        log.info("Loaded wards: %d features from %s", 0 if wards_gdf is None else len(wards_gdf), wards_path)

    # 1) Try OSMnx first (fast if available and internet is ok)
    G = None
    if OSMNX_AVAILABLE and wards_gdf is not None:
        G = build_graph_from_osm(wards_gdf)
        if G is None:
            log.info("OSMnx graph not built (falling back to KML).")

    # 2) If no OSM graph, read KMLs and build fallback graph
    if G is None:
        lines_gdf = read_kml_layers(kml_paths)
        lines_gdf = linestrings_from_gdf(lines_gdf)
        if lines_gdf.empty:
            log.error("No lines found in provided KMLs. Exiting.")
            sys.exit(1)
        G = build_graph_from_kml_lines(lines_gdf, snap_tol_m=args.snap_tol_m)

    # 3) compute approximate centralities
    bet, clos, deg = compute_approx_centralities(G, k_sample=args.k_sample, weight_attr="length_m")

    # attach node centralities as node attributes
    nx.set_node_attributes(G, bet, "bet_centrality_node")  # for nodes (betweenness)
    nx.set_node_attributes(G, clos, "closeness_node")
    nx.set_node_attributes(G, deg, "degree_centrality_node")

    # convert edges to gdf
    edges_gdf = graph_edges_to_gdf(G)
    if edges_gdf.empty:
        log.error("No edges extracted from graph. Exiting.")
        sys.exit(1)

    # compute centroids and lengths properly in metric CRS
    edges_proj = project_to_metric(edges_gdf.copy())
    edges_gdf["length_m"] = edges_proj.geometry.length.fillna(edges_gdf["length_m"]).astype(float)
    # compute centroids in lon/lat for output
    try:
        edges_centroid = edges_proj.geometry.centroid.to_crs(epsg=4326)
        edges_gdf["centroid_lon"] = edges_centroid.x
        edges_gdf["centroid_lat"] = edges_centroid.y
    except Exception:
        # fallback: compute centroid in current crs and hope it's lon/lat
        try:
            cent = edges_gdf.geometry.centroid
            edges_gdf["centroid_lon"] = cent.x
            edges_gdf["centroid_lat"] = cent.y
        except Exception:
            edges_gdf["centroid_lon"] = None
            edges_gdf["centroid_lat"] = None

    # map node centrality to edges: average of endpoint node betweenness/closeness/degree
    def node_attr_avg(u, v, attr_dict):
        a = attr_dict.get(u, 0.0)
        b = attr_dict.get(v, 0.0)
        return float((a + b) / 2.0)

    # node bet and clos dictionaries (some keys may be absent)
    node_bet = bet
    node_clos = clos
    node_deg = deg

    avg_bet = []
    avg_clos = []
    avg_deg = []
    for _, r in edges_gdf.iterrows():
        u = r["u"]; v = r["v"]
        avg_bet.append(node_attr_avg(u, v, node_bet))
        avg_clos.append(node_attr_avg(u, v, node_clos))
        avg_deg.append(node_attr_avg(u, v, node_deg))

    edges_gdf["betweenness_centrality"] = avg_bet
    edges_gdf["closeness_centrality"] = avg_clos
    edges_gdf["degree_centrality"] = avg_deg

    # Add road_id
    edges_gdf = edges_gdf.reset_index(drop=True)
    edges_gdf["road_id"] = edges_gdf.index.map(lambda x: f"road_{x+1}")

    # Reorder columns
    cols = ["road_id","u","v","length_m","road_type","lanes","speed_limit","oneway","betweenness_centrality","closeness_centrality","degree_centrality","centroid_lon","centroid_lat","geometry"]
    for c in cols:
        if c not in edges_gdf.columns:
            edges_gdf[c] = None
    edges_out = edges_gdf[cols]

    # ensure geometry column exists and is proper
    if "geometry" not in edges_out.columns:
        raise RuntimeError("No geometry column extracted for edges")

    # final export: GeoJSON + CSV
    out_geo = Path(outdir) / "roads_edges.geojson"
    out_csv = Path(outdir) / "roads_edges.csv"
        # -----------------------------
    # Memory-safe writer: validate daily parquet (if any) and stream GeoJSON/CSV creation
    # -----------------------------
    # Prepare WGS84 GeoDataFrame for GeoJSON output (avoid copying huge frames where possible)
    try:
        edges_out_wgs = edges_out.set_geometry("geometry")
        edges_out_wgs = edges_out_wgs.to_crs(epsg=4326)
    except Exception:
        # best-effort: assume geometry already lon/lat
        edges_out_wgs = edges_out.set_geometry("geometry")

    # Attempt straightforward GeoJSON write first (may fail on low-memory)
    try:
        edges_out_wgs.to_file(out_geo, driver="GeoJSON")
        # write CSV without geometry (keep centroid columns)
        edges_out_wgs.drop(columns=["geometry"]).to_csv(out_csv, index=False)
        log.info("Saved outputs: %s  %s", out_geo, out_csv)
    except Exception as write_err:
        log.warning("Direct GeoJSON/CSV write failed (%s). Falling back to streaming-safe write.", write_err)

        # Safe CSV fallback: convert geometry to WKT and write CSV
        try:
            edges_out2 = edges_out.copy()
            edges_out2["geometry_wkt"] = edges_out2.geometry.apply(lambda g: g.wkt if g is not None else None)
            edges_out2.drop(columns=["geometry"], inplace=True)
            edges_out2.to_csv(out_csv, index=False)
            log.info("WROTE roads CSV (geometry as WKT): %s", out_csv)
        except Exception as csv_err:
            log.warning("Fallback CSV write failed: %s", csv_err)

        # Attempt streaming GeoJSON write using Fiona (stream features) - lower memory footprint
        try:
            import fiona
            from fiona.crs import from_epsg
            schema = {
                "geometry": "Polygon" if edges_out_wgs.geometry.geom_type.mode == "Polygon" else "LineString",
                "properties": {}
            }
            # build properties schema heuristically (strings or numbers)
            for col in edges_out_wgs.columns:
                if col == "geometry":
                    continue
                # simplified typing
                dtype = edges_out_wgs[col].dtype
                if np.issubdtype(dtype, np.integer):
                    schema["properties"][col] = "int"
                elif np.issubdtype(dtype, np.floating):
                    schema["properties"][col] = "float"
                else:
                    schema["properties"][col] = "str"

            crs = from_epsg(4326)
            # open sink and stream features
            with fiona.open(str(out_geo), "w", driver="GeoJSON", crs=crs, schema=schema) as sink:
                for _, row in edges_out_wgs.iterrows():
                    props = {k: (row[k] if not pd.isna(row[k]) else None) for k in schema["properties"].keys()}
                    geom = row.geometry.__geo_interface__ if row.geometry is not None else None
                    sink.write({"geometry": geom, "properties": props})
            log.info("WROTE GeoJSON (streamed): %s", out_geo)
        except Exception as fiona_err:
            log.warning("Streaming GeoJSON write failed: %s", fiona_err)
            log.warning("As last resort, saving a small sample GeoJSON and full CSV (WKT).")

            # write small geojson sample to inspect
            try:
                sample = edges_out_wgs.head(100).copy()
                sample.to_file(out_geo, driver="GeoJSON")
                log.info("WROTE sample GeoJSON: %s (first 100 rows)", out_geo)
            except Exception as samp_err:
                log.warning("Failed to write sample geojson: %s", samp_err)

    # explicit GC to free memory
    try:
        del edges_out2
    except Exception:
        pass
    gc.collect()


    # save graph for future re-use
    try:
        import pickle
        gx_path = Path(outdir) / "roads_graph.pkl"
        with open(gx_path, "wb") as f:
            pickle.dump(G, f)
        log.info("Saved graph pickle: %s", gx_path)
    except Exception as e:
        log.warning("Failed to pickle graph: %s", e)

    log.info("Done.")

if __name__ == "__main__":
    main()
