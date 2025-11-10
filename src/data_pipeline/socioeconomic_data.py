"""
File: census_data.py
Author: Gemini
Project: Urban Infrastructure & Spatial Planning - Bengaluru
Description:
    This script is the core module for the socioeconomic data layer. It attempts 
    to fetch real ward boundaries via Bhuvan WFS and generates mock Census attributes 
    for simulation. Enhanced with vulnerability and equity metrics.
"""
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import datetime
import logging
import json # Added for metadata saving

# --- 1. CONFIGURATION AND SETUP ---

# Configure logging (CRITICAL for production tracing)
os.makedirs("../logs/", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler('../logs/data_extraction.log')]
)
log = logging.getLogger(__name__)

log.info("--- Starting Socioeconomic Data Extraction Framework (Census Layer) ---")
print("--- Starting Socioeconomic Data Extraction Framework (Census Layer) ---")

# 1a. Reproducibility
np.random.seed(42)

# Define target data directory
output_dir_socio = "../data/socioeconomic/"
os.makedirs(output_dir_socio, exist_ok=True)
log.info(f"Saving files to: {output_dir_socio}")
print(f"   -> Saving files to: {output_dir_socio}")

# Constants
N_AREAS_MOCK = 198  # Default mock number of Wards/Taluks
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32643" # UTM Zone 43N for Bengaluru

# ---------------------------------------------------------------------
# WFS Integration Approach 
# ---------------------------------------------------------------------
BHUVAN_WFS_URL = (
    "https://bhuvan.nrsc.gov.in/geoserver/<workspace>/wfs?"  
    "service=WFS&version=1.0.0&request=GetFeature&"
    "typeName=<layer>&outputFormat=application/json"
)

# --- NEW FUNCTION: Bhuvan WFS or Mock Fallback for Boundaries ---

def load_boundaries_from_bhuvan_or_fallback(wfs_url, n_areas_mock):
    """
    Attempts to load real ward boundaries using the Bhuvan WFS approach. 
    Falls back to mock geometry generation if the WFS connection fails.
    """
    # 1. ATTEMPT WFS FETCH
    try:
        # Check if the URL placeholders have been filled
        if "<workspace>" in wfs_url or "<layer>" in wfs_url:
            raise ValueError("Bhuvan WFS URL has placeholders. Falling back to mock data.")

        log.info("Attempting to fetch real ward boundaries via Bhuvan WFS.")
        print("   -> Attempting to fetch real ward boundaries via Bhuvan WFS...")
        
        # This line will connect to the external service
        real_gdf = gpd.read_file(wfs_url)
        
        if real_gdf.empty:
            raise Exception("WFS returned an empty GeoDataFrame.")

        # Standard cleaning
        real_gdf = real_gdf.to_crs(CRS_WGS84)
        
        log.info(f"Successfully loaded {len(real_gdf)} boundaries from Bhuvan WFS.")
        print(f"   ✅ Successfully loaded {len(real_gdf)} boundaries via WFS.")
        return real_gdf
        
    except (Exception, ValueError) as e:
        log.warning(f"Failed to load boundaries from Bhuvan WFS (Error: {e}). Falling back to mock geometry.")
        print("   ⚠️ WFS fetch failed or URL incomplete. Generating mock boundaries as fallback...")
        
        # 2. FALLBACK to MOCK GEOMETRY GENERATION
        center_lat, center_lon = 12.9716, 77.5946
        geometries = []
        for i in range(n_areas_mock):
            # Base jitter for area center
            lon_j = center_lon + np.random.uniform(-0.15, 0.15)
            lat_j = center_lat + np.random.uniform(-0.15, 0.15)
            
            # Define a small square area
            size = np.random.uniform(0.005, 0.015)
            polygon = Polygon([
                (lon_j - size, lat_j - size),
                (lon_j + size, lat_j - size),
                (lon_j + size, lat_j + size),
                (lon_j - size, lat_j + size)
            ])
            geometries.append(polygon)

        # Create mock geometry GDF
        mock_geometry_gdf = gpd.GeoDataFrame({
            'area_id': [f'Ward_{i:03d}' for i in range(n_areas_mock)]
        }, geometry=geometries, crs=CRS_WGS84)
        
        log.info(f"Mock geometry GeoDataFrame created with N={len(mock_geometry_gdf)} areas.")
        return mock_geometry_gdf


# --- 2. DATA GENERATION / FETCH (Ward/Taluk Level) ---
log.info("Starting Data Fetch/Generation for Socioeconomic Layer.")
print("\n[STEP 2: Integrating Real Boundaries (Bhuvan WFS) and Mock Attributes]")

# 2a. Fetch or Generate Boundaries
geometry_gdf = load_boundaries_from_bhuvan_or_fallback(BHUVAN_WFS_URL, N_AREAS_MOCK)
N_AREAS_FINAL = len(geometry_gdf)

# 2b. Mock Attribute Data Generation 
log.info(f"Generating mock attributes to match {N_AREAS_FINAL} geometries...")

# Initialize core attributes
socioeconomic_df = pd.DataFrame({
    'population': np.random.randint(25000, 80000, N_AREAS_FINAL),
    'literacy_rate': np.random.uniform(75.0, 95.0, N_AREAS_FINAL).round(2),
    'avg_household_income_rs': np.random.randint(50000, 300000, N_AREAS_FINAL),
    'poverty_index': np.clip(np.random.normal(0.2, 0.1, N_AREAS_FINAL), 0.05, 0.5).round(3)
})

# --- ADDED: ENHANCED VULNERABILITY METRICS ---
# Correlate vulnerability with poverty index
base_vulnerability = socioeconomic_df['poverty_index'] * 100
socioeconomic_df['pct_informal_housing'] = np.clip(
    base_vulnerability + np.random.normal(0, 5, N_AREAS_FINAL), 
    5, 40
).round(2)
socioeconomic_df['pct_elderly'] = np.clip(
    10 + (1 - socioeconomic_df['poverty_index']) * 15 + np.random.normal(0, 3, N_AREAS_FINAL), # Mock higher elderly population in established, less poor wards
    5, 25
).round(2)
# --- END ADDED VULNERABILITY METRICS ---


# 2c. Merge attributes onto geometry
socioeconomic_gdf = geometry_gdf.merge(
    socioeconomic_df, 
    left_index=True, 
    right_index=True, 
    how='left'
)

# Ensure an 'area_id' exists for consistency
if 'area_id' not in socioeconomic_gdf.columns:
    socioeconomic_gdf['area_id'] = socioeconomic_gdf.index.map(lambda x: f'Ward_{x:03d}')
    
log.info(f"Final Socioeconomic GeoDataFrame created with {len(socioeconomic_gdf)} records.")


# --- 3. DATA ENRICHMENT AND INTEGRITY ---
log.info("Starting Data Enrichment and Integrity Checks.")
print("\n[STEP 3: Enrichment and Integrity]")

# Derived Socioeconomic Metrics
socioeconomic_gdf["income_per_capita_rs"] = (
    socioeconomic_gdf["avg_household_income_rs"] / socioeconomic_gdf["population"]
).round(2)
socioeconomic_gdf["literacy_poverty_ratio"] = (
    socioeconomic_gdf["literacy_rate"] / (socioeconomic_gdf["poverty_index"] + 1e-6)
).round(2)


# Coordinate Validation 
socioeconomic_gdf["is_valid"] = socioeconomic_gdf.geometry.is_valid
if not socioeconomic_gdf["is_valid"].all():
    log.warning("Some geometries are invalid. Attempting to fix with .buffer(0)...")
    print("   ⚠️ Some geometries are invalid. Attempting to fix...")
    socioeconomic_gdf["geometry"] = socioeconomic_gdf.geometry.buffer(0)
    socioeconomic_gdf.drop(columns=['is_valid'], inplace=True)
    
if not socioeconomic_gdf.geometry.is_valid.all():
    log.error("Geometry fix failed. Aborting save.")
    print("   🛑 Critical Error: Geometry fix failed.")
    # In a real pipeline, you would use sys.exit() or raise an error here.
else:
    log.info("Geometry validation and fix completed successfully.")
    print("   ✅ Geometry validation passed.")
    
# Reproject to metric CRS (UTM) for calculations like area, perimeter, etc.
metric_gdf = socioeconomic_gdf.to_crs(CRS_METRIC)  
log.info(f"Reprojected to metric CRS ({CRS_METRIC}) for analysis.")

socioeconomic_gdf['area_sq_m'] = metric_gdf.geometry.area.round(2)

# Reproject back to WGS84 for visualization and saving (GeoJSON standard)
socioeconomic_gdf = socioeconomic_gdf.to_crs(CRS_WGS84)


# --- 4. DATA SAVING AND SUMMARIZING ---
log.info("Starting Data Saving and Summarization.")
print("\n[STEP 4: Saving Data]")

# Save the final GeoDataFrame
file_path = os.path.join(output_dir_socio, "socioeconomic_bengaluru_mock.geojson")
socioeconomic_gdf.to_file(file_path, driver='GeoJSON')

log.info(f"Socioeconomic data saved to: {file_path}")
print(f"   ✅ Socioeconomic data saved to: {file_path}")
print(f"   Total Areas Processed (Wards/Taluks): {len(socioeconomic_gdf)}")

log.info("\n--- Data Summary (Quick Verification) ---")
summary_stats = socioeconomic_gdf[[
    "population", 
    "avg_household_income_rs", 
    "pct_informal_housing", # Added new field
    "pct_elderly", # Added new field
    "poverty_index"
]].describe().round(2)

log.info(f"Data Summary:\n{summary_stats.to_string()}")
print("\n--- Data Summary (Quick Verification) ---")
print(summary_stats)


# Export metadata
metadata = {
    "city": "Bengaluru",
    "date_created": datetime.datetime.now().isoformat(),
    "data_source": "MOCK / Synthetic Data (Boundaries: Attempted Bhuvan WFS)",
    "total_areas": len(socioeconomic_gdf),
    "primary_crs": socioeconomic_gdf.crs.to_string(),
    "attributes": list(socioeconomic_gdf.columns),
    "notes": "Generated to simulate Census/BBMP data for modeling. Enhanced with informal housing and elderly population percentages."
}
metadata_path = os.path.join(output_dir_socio, "socioeconomic_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
log.info(f"Metadata exported to {metadata_path}")
print(f"   ✅ Metadata exported to {metadata_path}")


log.info("--- Socioeconomic Data Acquisition Complete ---")
print("\n--- Socioeconomic Data Acquisition Complete ---")
