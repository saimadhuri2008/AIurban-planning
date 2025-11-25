# utils_geo.py
"""
Spatial utilities and robust IO helpers used by the master merge pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import h3

LOG = logging.getLogger("merge.utils")
LOG.setLevel(logging.INFO)


# ---------- IO helpers ----------
def read_geo(path: str, sample: Optional[int] = None) -> Optional[gpd.GeoDataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        LOG.warning("Geo file not found: %s", path)
        return None
    try:
        gdf = gpd.read_file(str(p))
        if sample and len(gdf) > sample:
            gdf = gdf.sample(sample, random_state=42).reset_index(drop=True)
        # ensure CRS set
        if gdf.crs is None:
            LOG.warning("Input %s has no CRS, assuming EPSG:4326", path)
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            try:
                gdf = gdf.to_crs("EPSG:4326")
            except Exception:
                pass
        return gdf
    except Exception as e:
        LOG.exception("Failed reading geo file %s: %s", path, e)
        return None


def read_table(path: str, sample: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        LOG.warning("Table file not found: %s", path)
        return None
    try:
        if p.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(str(p))
        elif p.suffix.lower() in [".ndjson", ".jsonl"]:
            df = pd.read_json(str(p), lines=True)
        elif p.suffix.lower() in [".geojson"]:
            df = gpd.read_file(str(p))
        else:
            df = pd.read_csv(str(p))
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        LOG.exception("Failed reading table %s: %s", path, e)
        return None


def save_gdf(gdf: gpd.GeoDataFrame, path: str):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(p), driver="GeoJSON")
        LOG.info("Saved geojson: %s", path)
    except Exception:
        try:
            gdf.to_file(str(p))
        except Exception as e:
            LOG.exception("Failed to save gdf to %s: %s", path, e)


def save_df(df: pd.DataFrame, path: str):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if str(p).lower().endswith(".parquet"):
            df.to_parquet(str(p), index=False)
        else:
            df.to_csv(str(p), index=False)
        LOG.info("Saved table: %s", path)
    except Exception:
        LOG.exception("Failed to save dataframe to %s", path)


# ---------- H3 helpers ----------
def h3_index_from_point(lat: float, lon: float, res: int) -> Optional[str]:
    try:
        return h3.geo_to_h3(lat, lon, res)
    except Exception:
        return None


def add_h3_centroid_cols(gdf: gpd.GeoDataFrame, h3_col: str = "h3_index",
                         lat_col: str = "centroid_lat", lon_col: str = "centroid_lon") -> gpd.GeoDataFrame:
    # If h3 index exists, add centroid lat/lon columns using h3 lib
    if h3_col not in gdf.columns:
        return gdf
    def _c(row):
        try:
            c = h3.h3_to_geo(row[h3_col])
            return c[0], c[1]
        except Exception:
            return None, None
    latlons = gdf.apply(lambda r: _c(r), axis=1)
    gdf[lat_col] = [x[0] for x in latlons]
    gdf[lon_col] = [x[1] for x in latlons]
    return gdf


# ---------- spatial joins ----------
def spatial_join_points_to_polygons(points_gdf: gpd.GeoDataFrame, polys_gdf: gpd.GeoDataFrame,
                                    how: str = "left", predicate: str = "within",
                                    poly_key: str = "ward_id", keep_geom: bool = True) -> gpd.GeoDataFrame:
    """
    Join points -> polygons. Returns points_gdf with a column poly_key (value from polys_gdf).
    """
    if points_gdf is None or polys_gdf is None:
        return points_gdf
    try:
        polys = polys_gdf[[poly_key, "geometry"]].copy() if poly_key in polys_gdf.columns else polys_gdf[["geometry"]].copy()
        pts = points_gdf.copy()
        # ensure same crs
        try:
            pts = pts.to_crs("EPSG:4326")
            polys = polys.to_crs("EPSG:4326")
        except Exception:
            pass
        joined = gpd.sjoin(pts, polys, how=how, predicate=predicate)
        # sjoin places poly_key into joined (if poly_key existed)
        if poly_key in joined.columns:
            pts[poly_key] = joined[poly_key]
        else:
            # fallback: if polys had index - use index_right
            if "index_right" in joined.columns:
                pts[poly_key] = joined["index_right"]
        if not keep_geom:
            pts = pts.drop(columns=["geometry"], errors="ignore")
        return pts
    except Exception:
        LOG.exception("Spatial join failed")
        return points_gdf


def nearest_node_join(src_gdf: gpd.GeoDataFrame, target_gdf: gpd.GeoDataFrame,
                      target_key: str = "id", return_dist_col: str = None) -> gpd.GeoDataFrame:
    """
    For each row in src_gdf, find nearest geometry in target_gdf and attach target_key.
    """
    if src_gdf is None or target_gdf is None:
        return src_gdf
    try:
        src = src_gdf.copy()
        tgt = target_gdf.copy()
        # ensure same crs
        src = src.to_crs("EPSG:4326")
        tgt = tgt.to_crs("EPSG:4326")
        tgt = tgt.reset_index().rename(columns={"index": "_tindex"})
        # build kd tree via shapely STRtree if available (fast)
        from shapely.strtree import STRtree
        geom_to_idx = {geom: idx for idx, geom in enumerate(tgt.geometry)}
        tree = STRtree(list(geom_to_idx.keys()))
        tgt_geoms = list(geom_to_idx.keys())
        tgt_indices = list(geom_to_idx.values())

        nearest_ids = []
        dists = []
        for geom in src.geometry:
            try:
                cand = tree.nearest(geom)
                cand_idx = geom_to_idx.get(cand)
                row = tgt.iloc[cand_idx]
                # compute approximate distance (degrees)
                d = geom.distance(cand)
                nearest_ids.append(row.get(target_key) if target_key in row else row.get("_tindex"))
                dists.append(float(d))
            except Exception:
                nearest_ids.append(None)
                dists.append(None)
        src[f"nearest_{target_key}"] = nearest_ids
        if return_dist_col:
            src[return_dist_col] = dists
        return src
    except Exception:
        LOG.exception("Nearest join failed")
        return src_gdf


# ---------- geometry helpers ----------
def ensure_point_gdf_from_table(df: pd.DataFrame, lon_candidates: List[str], lat_candidates: List[str],
                                lon_col: str = "lon", lat_col: str = "lat") -> Optional[gpd.GeoDataFrame]:
    """
    Take a tabular dataframe and create a GeoDataFrame by detecting lat/lon columns.
    """
    if df is None:
        return None
    cols = df.columns.tolist()
    lonc = next((c for c in lon_candidates if c in cols), None)
    latc = next((c for c in lat_candidates if c in cols), None)
    if not lonc or not latc:
        LOG.warning("No lat/lon columns found among candidates %s / %s", lon_candidates, lat_candidates)
        return None
    try:
        tmp = df.copy()
        tmp[lon_col] = pd.to_numeric(tmp[lonc], errors="coerce")
        tmp[lat_col] = pd.to_numeric(tmp[latc], errors="coerce")
        tmp = tmp.dropna(subset=[lon_col, lat_col])
        gdf = gpd.GeoDataFrame(tmp, geometry=[Point(xy) for xy in zip(tmp[lon_col], tmp[lat_col])], crs="EPSG:4326")
        return gdf
    except Exception:
        LOG.exception("Failed creating point gdf")
        return None
