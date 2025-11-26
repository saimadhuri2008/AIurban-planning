# fix_wards_geometry_v2.py
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
from shapely import wkt, wkb
from shapely.geometry import shape, Point, Polygon
import sys

INPUT = Path(r"C:\AIurban-planning\data\processed\master\wards_master.parquet")
OUT_PARQUET = INPUT.with_name("wards_master_fixed.parquet")
OUT_GPKG = INPUT.with_name("wards_master_fixed.gpkg")

def convert_value_to_geom(v):
    """Try WKB -> WKT -> GeoJSON -> list coords -> None"""
    if v is None:
        return None
    # already shapely
    try:
        from shapely.geometry.base import BaseGeometry
        if isinstance(v, BaseGeometry):
            return v
    except Exception:
        pass

    # WKB bytes or memoryview
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return wkb.loads(bytes(v))
        except Exception:
            pass

    # WKT string
    if isinstance(v, str):
        s = v.strip()
        # quick WKT detection
        if s.upper().startswith(("POINT","LINESTRING","POLYGON","MULTI","GEOMETRYCOLLECTION")):
            try:
                return wkt.loads(s)
            except Exception:
                pass
        # try parse as JSON/GeoJSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and "type" in parsed:
                return shape(parsed)
        except Exception:
            # not JSON
            pass

    # GeoJSON dict
    if isinstance(v, dict) and "type" in v:
        try:
            return shape(v)
        except Exception:
            pass

    # list-like coords -> create Point/Polygon heuristics
    if isinstance(v, (list, tuple)):
        # point [x,y] or [lon,lat]
        if len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            return Point(v[0], v[1])
        # polygon-like (list of lists)
        if len(v) and isinstance(v[0], (list, tuple)):
            try:
                return Polygon(v)
            except Exception:
                pass

    return None

def ensure_geometry_column(df, geom_col_candidate="geometry", crs_epsg=4326):
    """Return GeoDataFrame with active 'geometry' column (overwrites safely)."""
    df = df.copy()
    # If already GeoDataFrame with a geometry dtype and active geometry set -> just ensure CRS
    if isinstance(df, gpd.GeoDataFrame):
        try:
            _ = df.geometry  # will raise if active geometry not set
            if df.geometry.dtype == "geometry":
                if df.crs is None:
                    df = df.set_crs(epsg=crs_epsg, allow_override=True)
                return df
        except Exception:
            pass

    # If there is a column named geom_col_candidate, try convert that series
    if geom_col_candidate in df.columns:
        s = df[geom_col_candidate]
        # quick sample test to decide conversion type
        sample = s.dropna().iloc[0] if s.dropna().shape[0] > 0 else None
        # If sample already shapely -> promote to GeoDataFrame
        try:
            from shapely.geometry.base import BaseGeometry
            if isinstance(sample, BaseGeometry):
                gdf = gpd.GeoDataFrame(df, geometry=geom_col_candidate, crs=f"EPSG:{crs_epsg}")
                return gdf
        except Exception:
            pass

        # Convert entire series using convert_value_to_geom
        print(f"Converting column '{geom_col_candidate}' -> shapely geometries ...")
        converted = s.apply(convert_value_to_geom)
        nonnull = converted.dropna().shape[0]
        print(f"  converted non-null geometries: {nonnull} / {len(s)}")
        if nonnull == 0:
            raise RuntimeError(f"Could not convert any geometries from column '{geom_col_candidate}'. Inspect sample: {sample!r}")

        # Overwrite (or create) 'geometry' column safely
        df["geometry"] = converted
        # Build GeoDataFrame and set CRS
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=crs_epsg, allow_override=True)
        return gdf

    # If no candidate column exists, try to detect columns with object dtype that may hold geom
    for c in df.columns:
        if df[c].dtype == object:
            sample_vals = df[c].dropna().iloc[:5] if df[c].dropna().shape[0] else []
            # crude presence of WKB/WKT/GeoJSON indicators
            if any(isinstance(v, (bytes, bytearray, memoryview)) for v in sample_vals) or \
               any(isinstance(v, str) and ("POLYGON" in v.upper() or "POINT" in v.upper() or "type" in v.lower()) for v in sample_vals) or \
               any(isinstance(v, dict) and "type" in v for v in sample_vals):
                print(f"Trying fallback column '{c}' for geometry conversion.")
                converted = df[c].apply(convert_value_to_geom)
                if converted.dropna().shape[0] > 0:
                    df["geometry"] = converted
                    gdf = gpd.GeoDataFrame(df, geometry="geometry")
                    if gdf.crs is None:
                        gdf = gdf.set_crs(epsg=crs_epsg, allow_override=True)
                    return gdf

    raise RuntimeError("No geometry column found/convertible. Provide sample cell to debug.")

def main():
    print("Loading parquet:", INPUT)
    df = pd.read_parquet(INPUT)
    print("Columns:", df.columns.tolist())
    # Show a tiny sample to help debugging
    print("Sample first row (keys):", list(df.iloc[0].keys()) if len(df)>0 else "empty")
    # ensure geometry
    try:
        gdf = ensure_geometry_column(df, geom_col_candidate="geometry", crs_epsg=4326)
    except Exception as e:
        print("Failed to ensure geometry:", e)
        raise

    # Validate geometries and attempt fix if invalid
    total = len(gdf)
    valid_count = int(gdf.geometry.is_valid.sum())
    print(f"GeoDataFrame rows: {total}, valid geometries: {valid_count}")
    if valid_count < total:
        print("Attempting to fix invalid geometries with buffer(0) for those rows.")
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = gdf.loc[~gdf.geometry.is_valid, "geometry"].buffer(0)
        print("Rechecking validity:", int(gdf.geometry.is_valid.sum()), "valid geometries after fix.")

    # Save outputs
    print("Writing fixed parquet to:", OUT_PARQUET)
    gdf.to_parquet(OUT_PARQUET, index=False)
    try:
        print("Also writing geopackage for quick verification:", OUT_GPKG)
        gdf.to_file(OUT_GPKG, layer="wards_master_fixed", driver="GPKG")
    except Exception as e:
        print("Could not write GPKG (this is non-fatal). Error:", e)

    print("Done. You can now point your analysis script to:", OUT_PARQUET)
    print("Preview row 0:")
    print(gdf.head(1).T)

if __name__ == "__main__":
    main()
