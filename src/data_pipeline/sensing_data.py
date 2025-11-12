#!/usr/bin/env python3
"""
remote_sensing_pipeline.py

Process local VIIRS Nighttime Lights (NTL) GeoTIFFs and Sentinel/Landsat bands to:
 - compute NDVI
 - produce H3-aggregated features (mean/median)
 - output GeoJSON / Parquet and metadata

Usage:
    python src/data_pipeline/remote_sensing/remote_sensing_pipeline.py \
       --ntl_dir "data/raw/ntl/" \
       --sentinel_dir "data/raw/sentinel/" \
       --boundary "data/socioeconomic/socioeconomic_bengaluru_final.geojson" \
       --h3_res 8
"""

import argparse
from pathlib import Path
import logging
import json
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
from rasterstats import zonal_stats
import h3
from shapely.geometry import shape, mapping, Point, Polygon
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
log = logging.getLogger("remote_sensing")

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_boundary(boundary_path: Path):
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)
    gdf = gpd.read_file(boundary_path)
    # ensure WGS84
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

def list_geotiffs_in_dir(d: Path):
    return sorted([p for p in d.glob("**/*.tif")] + [p for p in d.glob("**/*.tiff")])

def compute_ndvi_from_red_nir(red_path: Path, nir_path: Path, out_path: Path, resampling=Resampling.nearest):
    """
    Compute NDVI = (NIR - RED) / (NIR + RED)
    Expects single-band red and nir GeoTIFFs (or uses first band).
    Saves NDVI GeoTIFF to out_path and returns the xarray.DataArray object.
    """
    log.info("Computing NDVI from RED=%s and NIR=%s", red_path, nir_path)
    red = rxr.open_rasterio(red_path, masked=True).squeeze()  # (y,x)
    nir = rxr.open_rasterio(nir_path, masked=True).squeeze()

    # Reproject/resample NIR to RED grid if needed
    if not red.rio.crs == nir.rio.crs or red.rio.transform() != nir.rio.transform() or red.shape != nir.shape:
        nir = nir.rio.reproject_match(red, resampling=resampling)

    # cast to float
    red = red.astype("float32")
    nir = nir.astype("float32")

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = ndvi.clip(-1, 1)

    # save
    ndvi.rio.set_crs(red.rio.crs)
    ndvi.rio.to_raster(out_path)
    log.info("Saved NDVI raster to %s", out_path)
    return ndvi

def aggregate_raster_to_h3(raster_path: Path, h3_res: int, boundary_gdf: gpd.GeoDataFrame, stat: str = "mean", sample_fraction: float = 1.0):
    """
    Aggregate a raster to H3 cells defined inside the boundary_gdf extent.
    Returns a DataFrame with columns: h3_index, stat_value, centroid_lat, centroid_lon.
    Uses rasterstats.zonal_stats under the hood for simplicity and robustness.
    """
    log.info("Aggregating raster %s to H3 res %d (stat=%s)", raster_path, h3_res, stat)
    da = rxr.open_rasterio(raster_path, masked=True).squeeze()
    da = da.rio.reproject("EPSG:4326")  # ensure WGS84
    bounds = boundary_gdf.total_bounds  # minx, miny, maxx, maxy

    # Build list of H3 cells covering bounding box (a conservative range)
    # find center points of candidate H3 cells by scanning bbox: instead, use polygon cover approach:
    # We'll rasterize the H3 hex polygons that intersect bounds
    minx, miny, maxx, maxy = bounds
    # generate grid of sample points (coarse) to find relevant hexes
    lon_steps = np.linspace(minx, maxx, 300)  # adjust density if needed based on bbox size
    lat_steps = np.linspace(miny, maxy, 300)
    # create sample points (subsample for speed)
    pts = []
    for i, lon in enumerate(lon_steps):
        if i % 30 != 0:
            continue
        for j, lat in enumerate(lat_steps):
            if j % 30 != 0:
                continue
            pts.append((lat, lon))
    # compute h3 indices
    h3_set = set()
    for lat, lon in pts:
        try:
            h = h3.latlng_to_cell(float(lat), float(lon), h3_res)
            h3_set.add(h)
        except Exception:
            continue

    # expand each hex to polygon and intersect with boundary to filter
    hex_polys = []
    hex_ids = []
    for h in h3_set:
        poly = Polygon(h3.h3_to_geo_boundary(h, geo_json=True))
        hex_polys.append(poly)
        hex_ids.append(h)

    hex_gdf = gpd.GeoDataFrame({"h3_index": hex_ids}, geometry=hex_polys, crs="EPSG:4326")
    # join with study area to keep only hexes overlapping boundary
    hex_gdf = hex_gdf[hex_gdf.intersects(boundary_gdf.unary_union)]

    if hex_gdf.empty:
        log.warning("No H3 hexes found overlapping study area. Try increasing sampling resolution or use polygon_to_cells approach.")
        return pd.DataFrame(columns=["h3_index", f"{stat}_value", "centroid_lat", "centroid_lon"])

    # use rasterstats zonal_stats to compute the stat for each hex polygon
    zs = zonal_stats(hex_gdf.geometry, str(raster_path), stats=[stat], all_touched=False, nodata=None, geojson_out=False)
    rows = []
    for idx, h in enumerate(hex_gdf["h3_index"].values):
        val = zs[idx].get(stat, None)
        centroid = hex_gdf.geometry.iloc[idx].centroid
        rows.append({"h3_index": h, f"{stat}_value": None if val is None else float(val),
                     "centroid_lat": centroid.y, "centroid_lon": centroid.x})
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Pipeline main
# -------------------------
def main(args):
    ntl_dir = Path(args.ntl_dir) if args.ntl_dir else None
    sentinel_dir = Path(args.sentinel_dir) if args.sentinel_dir else None
    boundary_path = Path(args.boundary)
    h3_res = args.h3_res
    out_dir = ensure_dir(Path(args.output_dir))

    boundary_gdf = read_boundary(boundary_path)

    outputs = {"ntl": None, "ndvi": None}

    # --- process VIIRS NTL files ---
    if ntl_dir and ntl_dir.exists():
        tifs = list_geotiffs_in_dir(ntl_dir)
        if not tifs:
            log.warning("No NTL GeoTIFFs found in %s", ntl_dir)
        else:
            # For many monthly files you might want to average them; here we take the mean across files
            # If only one file, we'll use it as-is.
            if len(tifs) == 1:
                ntl_raster = tifs[0]
            else:
                # simple averaging mosaic: open with rioxarray, stack and compute mean
                rasters = [rxr.open_rasterio(p).squeeze().rio.reproject("EPSG:4326") for p in tifs]
                # reproject/resample to same grid as first
                base = rasters[0]
                regridded = []
                for r in rasters:
                    if r.rio.transform() != base.rio.transform() or r.shape != base.shape:
                        r = r.rio.reproject_match(base)
                    regridded.append(r)
                stacked = xr.concat(regridded, dim="time") if "xr" in globals() else None
                if stacked is not None:
                    mean_da = stacked.mean(dim="time")
                    ntl_raster_path = out_dir / "ntl_mean_mosaic.tif"
                    mean_da.rio.to_raster(ntl_raster_path)
                    ntl_raster = ntl_raster_path
                else:
                    # fallback: use first file
                    ntl_raster = tifs[0]

            df_ntl_h3 = aggregate_raster_to_h3(ntl_raster, h3_res, boundary_gdf, stat="mean")
            if not df_ntl_h3.empty:
                ntl_out = out_dir / f"ntl_h3_res{h3_res}.parquet"
                df_ntl_h3.to_parquet(ntl_out, index=False)
                outputs["ntl"] = str(ntl_out)
                log.info("Wrote NTL H3 parquet: %s", ntl_out)
            else:
                log.warning("NTL aggregation returned empty result.")

    # --- process NDVI from sentinel dir (expect red & nir or precomputed NDVI tifs) ---
    if sentinel_dir and sentinel_dir.exists():
        # look for precomputed ndvi or raw bands
        ndvi_candidates = list(sentinel_dir.glob("*ndvi*.tif")) + list(sentinel_dir.glob("*NDVI*.tif"))
        if ndvi_candidates:
            ndvi_raster = ndvi_candidates[0]
        else:
            # try find red and nir
            red_candidates = list(sentinel_dir.glob("*_B4*.tif")) + list(sentinel_dir.glob("*red*.tif"))
            nir_candidates = list(sentinel_dir.glob("*_B8*.tif")) + list(sentinel_dir.glob("*nir*.tif"))
            if red_candidates and nir_candidates:
                red_path = red_candidates[0]
                nir_path = nir_candidates[0]
                ndvi_out_path = out_dir / "ndvi_computed.tif"
                compute_ndvi_from_red_nir(red_path, nir_path, ndvi_out_path)
                ndvi_raster = ndvi_out_path
            else:
                ndvi_raster = None

        if ndvi_raster:
            df_ndvi_h3 = aggregate_raster_to_h3(ndvi_raster, h3_res, boundary_gdf, stat="mean")
            if not df_ndvi_h3.empty:
                ndvi_out = out_dir / f"ndvi_h3_res{h3_res}.parquet"
                df_ndvi_h3.to_parquet(ndvi_out, index=False)
                outputs["ndvi"] = str(ndvi_out)
                log.info("Wrote NDVI H3 parquet: %s", ndvi_out)
            else:
                log.warning("NDVI aggregation returned empty result.")
    # metadata
    meta = {
        "boundary": str(boundary_path),
        "h3_res": h3_res,
        "outputs": outputs
    }
    with open(out_dir / "remote_sensing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata: %s", out_dir / "remote_sensing_metadata.json")
    log.info("Remote sensing pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntl_dir", type=str, default="data/raw/ntl", help="Dir with VIIRS GeoTIFF(s)")
    parser.add_argument("--sentinel_dir", type=str, default="data/raw/sentinel", help="Dir with sentinel/landsat bands")
    parser.add_argument("--boundary", type=str, default="data/socioeconomic/socioeconomic_bengaluru_final.geojson", help="Boundary GeoJSON (wards)")
    parser.add_argument("--h3_res", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="data/remote_sensing")
    args = parser.parse_args()
    main(args)
