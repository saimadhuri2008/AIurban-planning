#!/usr/bin/env python3
"""
generate_bda.py

Generate synthetic BDA Master Plan-style policy zones from a city boundary,
optionally partition into radial sectors, attach ward names (spatial join),
and aggregate zones to H3 cells using h3 4.x style APIs (no "polyfill" call).

Usage (PowerShell example):
    python src/data_pipeline/generate_bda.py --boundary data/spatial/boundary.geojson \
      --wards data/spatial/wards_bengaluru_from_kml.geojson --n_zones 8 --do_h3 True --h3_res 8

This script intentionally avoids calling `h3.polyfill` (deprecated/renamed).
It will use h3.geo_to_cells / polygon_to_cells_experimental / polygon_to_cells
depending on what's available in the installed h3 package.
"""
import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Polygon, Point, MultiPolygon
from shapely.ops import unary_union

# -------------------- logging --------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "generate_bda.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)


# -------------------- H3 wrapper for v4.x --------------------
def try_import_h3() -> Optional[Dict[str, Any]]:
    """
    Return wrapper dict with:
      - 'geo_to_cells'(geom_geojson, res) -> set(h3_index)
      - 'polygon_to_cells'(geom_geojson, res) -> set(h3_index)
      - 'polygon_to_cells_experimental'(geom_geojson, res, contain) -> set(h3_index)
      - 'cell_to_latlng'(h) -> (lat, lon)
      - 'module' -> raw h3 module
    Or None if not available.
    """
    try:
        import h3 as _h3
    except Exception as e:
        logging.warning("h3 import failed: %s", e)
        return None

    wrapper = {"module": _h3}

    # centroid function
    def _cell_to_latlng(h):
        # prefer cell_to_latlng (v4.x)
        if hasattr(_h3, "cell_to_latlng"):
            try:
                latlon = _h3.cell_to_latlng(h)
                return float(latlon[0]), float(latlon[1])
            except Exception:
                pass
        # fallback names
        for path in ("cell_to_latlng", "cell_to_geo", "h3_to_geo", "cell_to_latlon"):
            fn = getattr(_h3, path, None)
            if callable(fn):
                try:
                    latlon = fn(h)
                    return float(latlon[0]), float(latlon[1])
                except Exception:
                    continue
        return None

    wrapper["cell_to_latlng"] = _cell_to_latlng

    # polygon/geo -> cells adapters (v4.x has geo_to_cells, polygon_to_cells, polygon_to_cells_experimental)
    def _geo_to_cells(geom_geojson, res):
        """Use h3.geo_to_cells if available (accepts geojson-like)."""
        fn = getattr(_h3, "geo_to_cells", None)
        if not callable(fn):
            raise RuntimeError("h3.geo_to_cells not available")
        return set(fn(geom_geojson, res))

    def _polygon_to_cells(geom_geojson, res):
        """Use polygon_to_cells if available (tries with coordinates/lon-lat order)."""
        fn = getattr(_h3, "polygon_to_cells", None)
        if not callable(fn):
            raise RuntimeError("h3.polygon_to_cells not available")
        # try direct call; h3 implementation may expect lon-lat or a specific mapping
        return set(fn(geom_geojson, res))

    def _polygon_to_cells_experimental(geom_geojson, res, contain="center"):
        """Use polygon_to_cells_experimental if available (returns set)."""
        fn = getattr(_h3, "polygon_to_cells_experimental", None)
        if not callable(fn):
            raise RuntimeError("h3.polygon_to_cells_experimental not available")
        # Many variants return list; convert to set
        return set(fn(geom_geojson, res, contain))

    # populate wrapper with available functions
    if callable(getattr(_h3, "geo_to_cells", None)):
        wrapper["geo_to_cells"] = _geo_to_cells
    if callable(getattr(_h3, "polygon_to_cells", None)):
        wrapper["polygon_to_cells"] = _polygon_to_cells
    if callable(getattr(_h3, "polygon_to_cells_experimental", None)):
        wrapper["polygon_to_cells_experimental"] = _polygon_to_cells_experimental

    # sanity check: require at least one polygon->cells function
    if not any(k in wrapper for k in ("geo_to_cells", "polygon_to_cells", "polygon_to_cells_experimental")):
        logging.warning("h3 installed but no geo/polygon->cells functions detected; H3 disabled.")
        return None

    logging.info("h3 imported; version=%s; available functions=%s",
                 getattr(_h3, "__version__", "unknown"),
                 [k for k in wrapper.keys() if k != "module"])
    return wrapper


# -------------------- helpers --------------------
def safe_to_geojson_mapping(geom):
    """Return a geojson-like mapping for a shapely geometry with lon-lat coordinates."""
    gm = mapping(geom)
    # ensure coordinates are floats (lon, lat)
    return gm


def write_diag(out_dir: Path, index: int, geom, errors: list):
    """Write diagnostics for a failing geometry to out_dir/h3_diag"""
    diag_dir = out_dir / "h3_diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    geo_path = diag_dir / f"bda_h3_fail_{index}_{ts}.geojson"
    json_path = diag_dir / f"bda_h3_fail_{index}_{ts}.json"
    # save geometry geojson (single feature)
    try:
        gdf = gpd.GeoDataFrame([{"idx": index}], geometry=[geom], crs="EPSG:4326")
        gdf.to_file(str(geo_path), driver="GeoJSON")
    except Exception:
        pass
    # save JSON diagnostics
    diag = {"index": index, "errors": errors, "timestamp": ts}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)


# -------------------- core: create radial partitions --------------------
def radial_partition(boundary_gdf: gpd.GeoDataFrame, n_zones: int) -> gpd.GeoDataFrame:
    """
    Partition the unary_union of the boundary into `n_zones` radial wedges around its centroid.
    Returns GeoDataFrame with zone polygons and metadata.
    """
    logging.info("Partitioning boundary into %d radial zones...", n_zones)
    union = unary_union(boundary_gdf.geometry)
    centroid = union.centroid
    cx, cy = centroid.x, centroid.y

    # build wedges
    wedges = []
    total = 360.0
    step = total / n_zones
    for i in range(n_zones):
        a1 = np.deg2rad(i * step)
        a2 = np.deg2rad((i + 1) * step)
        # create a large triangular wedge that covers bounding box extent
        # pick a radius large enough to cover the union bbox
        minx, miny, maxx, maxy = union.bounds
        maxd = max(maxx - minx, maxy - miny) * 4.0 + 0.01
        pts = [
            (cx, cy),
            (cx + maxd * np.cos(a1), cy + maxd * np.sin(a1)),
            (cx + maxd * np.cos(a2), cy + maxd * np.sin(a2)),
        ]
        wedge = Polygon(pts)
        # intersect wedge with union to get actual sector within the boundary
        zone = union.intersection(wedge)
        if zone.is_empty:
            continue
        wedges.append({"zone_id": f"zone_{i+1}", "geometry": zone})

    zones_gdf = gpd.GeoDataFrame(wedges, crs=boundary_gdf.crs if boundary_gdf.crs else "EPSG:4326")
    # compute centroids (warning: geographic CRS may give degree units)
    zones_gdf["centroid_lon"] = zones_gdf.geometry.centroid.x
    zones_gdf["centroid_lat"] = zones_gdf.geometry.centroid.y
    logging.info("Created %d radial zones.", len(zones_gdf))
    return zones_gdf


# -------------------- spatial join with wards --------------------
def spatial_join_wards(zones_gdf: gpd.GeoDataFrame, wards_path: Optional[str]):
    if not wards_path:
        logging.info("No wards path provided; skipping ward join.")
        return zones_gdf
    p = Path(wards_path)
    if not p.exists():
        logging.warning("Wards file not found: %s", wards_path)
        return zones_gdf
    try:
        wards = gpd.read_file(str(p)).to_crs("EPSG:4326")
        # attempt to pick ward name column
        ward_cols = [c for c in wards.columns if "ward" in c.lower() or "name" in c.lower()]
        ward_col = ward_cols[0] if ward_cols else wards.columns[0]
        joined = gpd.sjoin(zones_gdf, wards[[ward_col, "geometry"]], how="left", predicate="intersects")
        joined = joined.rename(columns={ward_col: "ward_name"})
        joined = joined.drop(columns=["index_right"], errors="ignore")
        logging.info("Spatial join with wards complete. Sample ward names: %s", joined["ward_name"].dropna().unique()[:5])
        return joined
    except Exception as e:
        logging.exception("Spatial join with wards failed: %s", e)
        return zones_gdf


# -------------------- H3 aggregation using v4.x style functions --------------------
def h3_aggregate_zones(zones_gdf: gpd.GeoDataFrame, h3_wrapper: Dict[str, Any], res: int, out_dir: Path):
    """
    For each zone geometry, call the available h3 function(s) to get H3 cells.
    Writes diagnostics for failures and returns pandas DataFrame of aggregated mappings.
    """
    rows = []
    diagnostics = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in zones_gdf.reset_index(drop=True).iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            logging.warning("Skipping empty geometry at idx %s", idx)
            continue

        gm = safe_to_geojson_mapping(geom)
        errors = []
        cells = set()

        # Try preferred functions in order:
        # 1) geo_to_cells (v4.x)
        if "geo_to_cells" in h3_wrapper:
            try:
                cells = h3_wrapper["geo_to_cells"](gm, res)
            except Exception as e:
                errors.append(("geo_to_cells", str(e)))

        # 2) polygon_to_cells_experimental with 'center' containment
        if not cells and "polygon_to_cells_experimental" in h3_wrapper:
            try:
                cells = h3_wrapper["polygon_to_cells_experimental"](gm, res, "center")
            except Exception as e:
                errors.append(("polygon_to_cells_experimental", str(e)))

        # 3) polygon_to_cells (fallback)
        if not cells and "polygon_to_cells" in h3_wrapper:
            try:
                cells = h3_wrapper["polygon_to_cells"](gm, res)
            except Exception as e:
                errors.append(("polygon_to_cells", str(e)))

        # 4) Try simplified geometry (reduce vertex count) then retry polygon_to_cells_experimental
        if not cells:
            try:
                simp = geom.simplify(0.0001, preserve_topology=True)
                gm_simp = safe_to_geojson_mapping(simp)
                if "polygon_to_cells_experimental" in h3_wrapper:
                    try:
                        cells = h3_wrapper["polygon_to_cells_experimental"](gm_simp, res, "center")
                    except Exception as e:
                        errors.append(("polygon_to_cells_experimental_simplified", str(e)))
                if not cells and "polygon_to_cells" in h3_wrapper:
                    try:
                        cells = h3_wrapper["polygon_to_cells"](gm_simp, res)
                    except Exception as e:
                        errors.append(("polygon_to_cells_simplified", str(e)))
            except Exception as e:
                errors.append(("simplify", str(e)))

        if not cells:
            logging.warning("polyfill failed for geometry idx %s; diagnostics written.", idx)
            write_diag(out_dir, idx, geom, errors)
            diagnostics.append({"idx": idx, "errors": errors})
            continue

        # append mapping rows
        for h in cells:
            rows.append({
                "zone_idx": idx,
                "zone_id": row.get("zone_id"),
                "h3_index": h,
                "permissible_far": row.get("permissible_far"),
                "height_limit_m": row.get("height_limit_m"),
                "setback_ratio": row.get("setback_ratio"),
            })

    h3_df = pd.DataFrame(rows)
    # compute centroids if possible
    if not h3_df.empty and "cell_to_latlng" in h3_wrapper:
        try:
            latlons = [h3_wrapper["cell_to_latlng"](h) for h in h3_df["h3_index"].tolist()]
            h3_df["centroid_lat"] = [ll[0] if ll else None for ll in latlons]
            h3_df["centroid_lon"] = [ll[1] if ll else None for ll in latlons]
        except Exception:
            h3_df["centroid_lat"] = None
            h3_df["centroid_lon"] = None

    # save aggregated H3 table
    out_parquet = out_dir / f"bda_policy_h3_res{res}.parquet"
    out_geo = out_dir / f"bda_policy_h3_res{res}.geojson"
    try:
        h3_df.to_parquet(out_parquet, index=False)
        logging.info("Saved H3 aggregated parquet: %s", out_parquet)
    except Exception as e:
        logging.warning("Failed to write H3 parquet: %s", e)
        try:
            h3_df.to_csv(str(out_dir / f"bda_policy_h3_res{res}.csv"), index=False)
            logging.info("Wrote H3 CSV fallback.")
        except Exception:
            logging.exception("Failed to write H3 CSV fallback.")

    # save centroids geojson for quick viz if centroid fields available
    try:
        if not h3_df.empty and h3_df["centroid_lon"].notna().any():
            g = gpd.GeoDataFrame(h3_df,
                                 geometry=gpd.points_from_xy(h3_df["centroid_lon"], h3_df["centroid_lat"]),
                                 crs="EPSG:4326")
            g.to_file(str(out_geo), driver="GeoJSON")
            logging.info("Saved H3 centroids geojson: %s", out_geo)
    except Exception:
        logging.exception("Failed to write H3 centroids geojson.")

    # diagnostics summary
    diag_path = out_dir / f"bda_h3_diag_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump({"n_mappings": len(rows), "n_failures": len(diagnostics), "failures": diagnostics}, f, indent=2)
    logging.info("H3 diagnostics saved: %s", diag_path)

    return h3_df, diag_path


# -------------------- save zones --------------------
def save_policy_outputs(policy_gdf: gpd.GeoDataFrame, out_dir: Path, prefix="bda_masterplan_zones"):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    geo_path = out_dir / f"{prefix}_{ts}.geojson"
    parquet_path = out_dir / f"{prefix}_{ts}.parquet"
    preview_path = out_dir / f"{prefix}_{ts}_preview.json"
    meta_path = out_dir / f"{prefix}_{ts}_metadata.json"

    # GeoJSON
    try:
        g = policy_gdf.copy()
        try:
            g = g.to_crs("EPSG:4326")
        except Exception:
            pass
        g.to_file(str(geo_path), driver="GeoJSON")
        logging.info("Saved GeoJSON: %s", geo_path)
    except Exception:
        logging.exception("Failed to save GeoJSON")

    # parquet / csv fallback
    try:
        policy_gdf.to_parquet(str(parquet_path), index=False)
        logging.info("Saved Parquet: %s", parquet_path)
    except Exception as e:
        logging.warning("Parquet write failed (%s); writing CSV fallback.", e)
        try:
            csv_path = out_dir / f"{prefix}_{ts}.csv"
            df_csv = policy_gdf.copy()
            if "geometry" in df_csv.columns:
                df_csv["geometry_wkt"] = df_csv.geometry.apply(lambda g: g.wkt if g is not None else None)
                df_csv = df_csv.drop(columns=["geometry"])
            df_csv.to_csv(str(csv_path), index=False)
            logging.info("Saved CSV fallback: %s", csv_path)
        except Exception:
            logging.exception("CSV fallback failed")

    # preview JSON (no geometry)
    try:
        preview_df = policy_gdf.copy()
        if "geometry" in preview_df.columns:
            preview_df = preview_df.drop(columns=["geometry"])
        preview_records = json.loads(preview_df.head(50).to_json(orient="records"))
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview_records, f, indent=2)
        logging.info("Saved preview JSON: %s", preview_path)
    except Exception:
        # fallback robust conversion
        try:
            preview_records = preview_df.head(50).to_dict(orient="records")
            with open(preview_path, "w", encoding="utf-8") as f:
                json.dump(preview_records, f, indent=2)
            logging.info("Saved preview JSON fallback.")
        except Exception:
            logging.exception("Failed to save preview JSON")

    # metadata
    try:
        metadata = {"created_at": datetime.datetime.now().isoformat(),
                    "n_zones": int(len(policy_gdf)),
                    "attributes": list(policy_gdf.columns)}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logging.info("Saved metadata: %s", meta_path)
    except Exception:
        logging.exception("Failed to save metadata")

    return {"geojson": str(geo_path), "parquet": str(parquet_path), "preview": str(preview_path), "metadata": str(meta_path)}


# -------------------- CLI main --------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate BDA policy zones and H3 aggregation (h3 v4.x compatible).")
    parser.add_argument("--boundary", type=str, required=True, help="Boundary GeoJSON")
    parser.add_argument("--wards", type=str, default=None, help="Wards GeoJSON (optional)")
    parser.add_argument("--out_dir", type=str, default="data/policy", help="Output folder")
    parser.add_argument("--n_zones", type=int, default=8, help="Number of radial zones")
    parser.add_argument("--do_h3", type=lambda s: s.lower() in ("true", "1", "yes"), default=False, help="Aggregate to H3")
    parser.add_argument("--h3_res", type=int, default=8, help="H3 resolution")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Starting BDA policy zone generation ===")
    logging.info("Inputs: boundary=%s wards=%s n_zones=%s do_h3=%s h3_res=%s",
                 args.boundary, args.wards, args.n_zones, args.do_h3, args.h3_res)

    # load boundary
    bpath = Path(args.boundary)
    if not bpath.exists():
        logging.error("Boundary file not found: %s", args.boundary)
        sys.exit(1)
    boundary_gdf = gpd.read_file(str(bpath))
    if boundary_gdf.crs is None:
        boundary_gdf.set_crs("EPSG:4326", inplace=True)
    else:
        try:
            boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
        except Exception:
            pass
    logging.info("Loaded boundary: %s (features=%d)", args.boundary, len(boundary_gdf))

    # partition
    zones_gdf = radial_partition(boundary_gdf, args.n_zones)

    # spatial join with wards
    zones_gdf = spatial_join_wards(zones_gdf, args.wards)

    # add some mock regulatory attributes (if missing)
    if "permissible_far" not in zones_gdf.columns:
        # simple synthetic assignment for demo
        zones_gdf["permissible_far"] = [round(1.0 + (i % 4) * 0.5, 2) for i in range(len(zones_gdf))]
        zones_gdf["height_limit_m"] = [20 + (i % 4) * 10 for i in range(len(zones_gdf))]
        zones_gdf["setback_ratio"] = [0.1 + (i % 3) * 0.05 for i in range(len(zones_gdf))]

    # optional H3 aggregation (v4.x style)
    h3_outputs = None
    if args.do_h3:
        h3_wrapper = try_import_h3()
        if h3_wrapper is None:
            logging.warning("H3 wrapper not available; skipping H3 aggregation.")
        else:
            h3_df, diag = h3_aggregate_zones(zones_gdf, h3_wrapper, args.h3_res, out_dir)
            h3_outputs = {"parquet": str(out_dir / f"bda_policy_h3_res{args.h3_res}.parquet"),
                          "geojson": str(out_dir / f"bda_policy_h3_res{args.h3_res}.geojson"),
                          "diag": str(diag)}

    outputs = save_policy_outputs(zones_gdf, out_dir, prefix="bda_masterplan_zones")
    logging.info("Saved policy outputs: %s", outputs)
    if h3_outputs:
        logging.info("H3 outputs: %s", h3_outputs)

    logging.info("Pipeline complete. Outputs saved to: %s", outputs)
    print("Pipeline complete. Outputs saved to:", outputs)


if __name__ == "__main__":
    main()
