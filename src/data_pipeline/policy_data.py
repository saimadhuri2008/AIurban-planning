import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# --- 1. CONFIGURATION AND SETUP ---
print("--- Starting Policy / Counterfactual Data Extraction Framework ---")

# Define target data directory
output_dir_policy = "../data/policy/"
os.makedirs(output_dir_policy, exist_ok=True)
print(f"   -> Saving files to: {output_dir_policy}")

# --- 2. MOCK POLICY ZONING DATA GENERATION ---
# This step simulates the process of converting complex BDA Master Plan 2031
# zoning maps (e.g., PDF/raster) into usable vector GeoJSON data.
print("\n[STEP 2: Generating Mock BDA Zoning Data]")

def generate_mock_policy_data():
    """
    Generates mock policy variables (Zoning, FAR/FSI) aggregated by Planning Zones.

    In a real project, this data would be manually digitized or extracted from
    BDA planning shapefiles and documents.
    """
    # -------------------------------------------------------------------------
    # Mock Policy Zone Boundaries (Simplistic Polygons around Bengaluru)
    # -------------------------------------------------------------------------
    zone_geometries = [
        # Zone 1: Older Core Area (High FAR/FSI to allow redevelopment)
        Polygon([(77.5, 12.95), (77.55, 12.95), (77.55, 13.0), (77.5, 13.0)]),
        # Zone 2: Green Belt / Low-Density Zone (Low FAR/FSI)
        Polygon([(77.6, 12.9), (77.65, 12.9), (77.65, 12.95), (77.6, 12.95)]),
        # Zone 3: Major Commercial Corridor (High FAR/FSI, Mixed Use)
        Polygon([(77.7, 13.05), (77.75, 13.05), (77.75, 13.1), (77.7, 13.1)]),
        # Zone 4: Residential Layout (Medium FAR/FSI)
        Polygon([(77.45, 12.8), (77.5, 12.8), (77.5, 12.85), (77.45, 12.85)]),
    ]

    # -------------------------------------------------------------------------
    # Mock Regulatory Attributes (Key Policy Variables)
    # -------------------------------------------------------------------------
    data = {
        'policy_zone_id': ['Core_R', 'Green_B', 'Commercial_C', 'Residential_L'],
        'base_zoning': ['Residential', 'Green_Belt', 'Commercial', 'Residential'],
        'permissible_far': [3.5, 0.75, 4.0, 2.5], # Floor Area Ratio (FSI)
        'height_limit_m': [40, 15, 60, 25],
        'setback_ratio': [0.15, 0.30, 0.10, 0.20], # Front/side setback requirement
        'geometry': zone_geometries
    }

    # Create the GeoDataFrame
    policy_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    return policy_gdf

policy_gdf = generate_mock_policy_data()


# --- 3. DATA SAVING ---
print("\n[STEP 3: Saving Policy Data]")

# Save the mock GeoDataFrame
file_path = os.path.join(output_dir_policy, "policy_bengaluru_mock.geojson")
policy_gdf.to_file(file_path, driver='GeoJSON')

print(f"   ✅ Policy data framework established.")
print(f"   Mock Policy Zone data saved to: {file_path}")
print(f"   Data attributes: {list(policy_gdf.columns)}")
print("      -> ACTION REQUIRED: Replace mock data with real BDA Master Plan boundary data.")


print("\n--- Step 5 Complete: ALL Multimodal Data Layers Now Extracted/Framed ---")
print("You now have all five required data layers (Spatial, Temporal, Remote Sensing, Socioeconomic, Policy) saved locally.")
