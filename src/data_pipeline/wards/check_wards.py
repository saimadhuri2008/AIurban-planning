import geopandas as gpd
from shapely.ops import unary_union
import shapely
import json

WARD_PATH = r"C:\AIurban-planning\data\geo\wards\wards_master_enriched.geojson"

print("Loading wards...")
gdf = gpd.read_file(WARD_PATH)
print("Rows:", len(gdf))

print("Columns:", gdf.columns.tolist())
print("CRS:", gdf.crs)

# Fix invalid geometries
gdf["geometry"] = gdf["geometry"].buffer(0)

invalid = gdf[~gdf.is_valid]
print("Invalid geometries:", len(invalid))

# Compute bounding box
minx, miny, maxx, maxy = gdf.total_bounds
print("\nBounding box:")
print("  minx:", minx)
print("  miny:", miny)
print("  maxx:", maxx)
print("  maxy:", maxy)

# Compute total area
gdf_3857 = gdf.to_crs(3857)
total_area_sqkm = gdf_3857.geometry.area.sum() / 1e6
print("\nTotal ward area (sqkm):", total_area_sqkm)

# Check expected H3 counts
import h3
hex_area = h3.cell_area("87754e1acffffff", unit='km^2')  # example res7 cell area

expected_res7 = int(total_area_sqkm / hex_area)
print("\nExpected H3 RES7 hex count:", expected_res7)

# Quick union geometry preview
union = unary_union(gdf.geometry)
print("\nUnion polygon type:", union.geom_type)

# Save union for checking
with open("/data/wards/wards_union_preview.geojson", "w") as f:
    json.dump({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": shapely.to_geojson(union)
        }]
    }, f)

print("\nSaved union preview: wards_union_preview.geojson")
