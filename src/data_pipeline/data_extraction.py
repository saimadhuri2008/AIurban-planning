"""
File: 03_Multimodal_Data_Extraction_Master.py
Author: Gemini
Project: Urban Infrastructure & Spatial Planning - Bengaluru
Description:
    Master script to frame, extract, and mock all advanced multimodal data layers 
    (Remote Sensing, Policy, Mobility, Utilities) for subsequent H3 aggregation.
"""
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import osmnx as ox
import datetime
import warnings
import logging
# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. CONFIGURATION AND SETUP ---
np.random.seed(42) 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

REMOTE_DIR = "../data/remote_sensing/"
POLICY_DIR = "../data/policy/"
MOBILITY_DIR = "../data/mobility/"
UTILITIES_DIR = "../data/utilities/"

os.makedirs(REMOTE_DIR, exist_ok=True)
os.makedirs(POLICY_DIR, exist_ok=True)
os.makedirs(MOBILITY_DIR, exist_ok=True)
os.makedirs(UTILITIES_DIR, exist_ok=True)

logging.info("--- Starting Multimodal Data Extraction Master Script ---")
print("--- Starting Multimodal Data Extraction Master Script ---")

# Define BBOX for consistent OSMnx extraction (Same as in your code)
NORTH, SOUTH, EAST, WEST = 13.15, 12.8, 77.75, 77.45 
PLACE_NAME = "Bengaluru, Karnataka, India"


# --- 2. REMOTE SENSING FRAMEWORK (NTL & NDVI) ---
print("\n[STEP 2: Remote Sensing Framework (NTL & NDVI Mock)]")

# Base coordinates for mocking, centralized around Bengaluru
mock_coords = [(77.5 + np.random.uniform(-0.15, 0.15), 12.9 + np.random.uniform(-0.15, 0.15)) for _ in range(500)]
mock_points = [Point(xy) for xy in mock_coords]

# 2.1. NTL Mocking
data_ntl = {
    'grid_id': np.arange(500),
    'avg_ntl_intensity': np.clip(np.random.normal(3.5, 2.0, 500), 0.1, 10.0),
    'geometry': mock_points
}
ntl_gdf = gpd.GeoDataFrame(data_ntl, crs="EPSG:4326")
ntl_gdf.to_file(os.path.join(REMOTE_DIR, "ntl_bengaluru_mock.geojson"), driver='GeoJSON')
logging.info("NTL mock data saved.")

# 2.2. NDVI Mocking
data_ndvi = {
    'grid_id': np.arange(500),
    'avg_ndvi': np.clip(np.random.normal(0.45, 0.2, 500), 0.05, 0.85),
    'geometry': mock_points
}
ndvi_gdf = gpd.GeoDataFrame(data_ndvi, crs="EPSG:4326")
ndvi_gdf.to_file(os.path.join(REMOTE_DIR, "ndvi_bengaluru_mock.geojson"), driver='GeoJSON')
logging.info("NDVI mock data saved.")
print(" ✅ NTL & NDVI mock feature files saved to ../data/remote_sensing/")


# --- 3. POLICY / ZONING FRAMEWORK ---
print("\n[STEP 3: Policy / Zoning Framework (Mock)]")

# Zone definitions (simplified polygons covering parts of the BBOX)
zone_geometries = [
    Polygon([(77.5, 12.95), (77.55, 12.95), (77.55, 13.0), (77.5, 13.0)]),
    Polygon([(77.6, 12.9), (77.65, 12.9), (77.65, 12.95), (77.6, 12.95)]),
    Polygon([(77.7, 13.05), (77.75, 13.05), (77.75, 13.1), (77.7, 13.1)]),
    Polygon([(77.45, 12.8), (77.5, 12.8), (77.5, 12.85), (77.45, 12.85)]),
]
data_policy = {
    'policy_zone_id': ['Core_R', 'Green_B', 'Commercial_C', 'Residential_L'],
    'base_zoning': ['Residential', 'Green_Belt', 'Commercial', 'Residential'],
    'permissible_far': [3.5, 0.75, 4.0, 2.5], 
    'geometry': zone_geometries
}
policy_gdf = gpd.GeoDataFrame(data_policy, crs="EPSG:4326")
policy_gdf.to_file(os.path.join(POLICY_DIR, "policy_bengaluru_mock.geojson"), driver='GeoJSON')
logging.info("Policy mock data saved.")
print(" ✅ Policy/Zoning mock feature file saved to ../data/policy/")


# --- 4. MOBILITY / TRANSPORT FRAMEWORK (Real OSM + Mock Operational) ---
print("\n[STEP 4: Mobility / Transport (Real OSM + Mock Metrics)]")

transit_tags = {'highway': ['bus_stop'], 'public_transport': ['station', 'platform']}
try:
    mobility_gdf = ox.features_from_bbox(NORTH, SOUTH, EAST, WEST, tags=transit_tags)
    mobility_gdf = mobility_gdf[mobility_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()
    mobility_gdf.set_crs("EPSG:4326", inplace=True)

    # Mock Operational Metrics
    is_bus = mobility_gdf.apply(lambda x: 'Bus Stop' in str(x.get('highway')), axis=1)
    mobility_gdf['daily_ridership_mock'] = np.where(is_bus, 
        np.clip(np.random.normal(3500, 1000, len(mobility_gdf)), 500, 8000), 
        np.clip(np.random.normal(18000, 4000, len(mobility_gdf)), 5000, 30000)
    ).astype(int)
    mobility_gdf['route_count_mock'] = np.where(is_bus, 
        np.random.randint(5, 25, len(mobility_gdf)), 
        np.random.choice([1, 2], len(mobility_gdf), p=[0.7, 0.3])
    ).astype(int)
    
    # Select final columns
    mobility_gdf = mobility_gdf[['daily_ridership_mock', 'route_count_mock', 'geometry']].copy()

except Exception as e:
    logging.warning(f"OSMnx Mobility failed, using fallback mock: {e}")
    # Fallback to pure mock
    N_FALLBACK = 500
    coords = [(77.59 + np.random.uniform(-0.15, 0.15), 12.97 + np.random.uniform(-0.15, 0.15)) for _ in range(N_FALLBACK)]
    mobility_gdf = gpd.GeoDataFrame({
        'daily_ridership_mock': np.random.randint(500, 25000, N_FALLBACK),
        'route_count_mock': np.random.randint(1, 20, N_FALLBACK)
    }, geometry=[Point(xy) for xy in coords], crs="EPSG:4326")

mobility_gdf.to_file(os.path.join(MOBILITY_DIR, "mobility_bengaluru_final.geojson"), driver='GeoJSON')
logging.info("Mobility data saved.")
print(f" ✅ Mobility/Transport stops (N={len(mobility_gdf)}) saved to ../data/mobility/")


# --- 5. ENERGY / UTILITIES FRAMEWORK (Real OSM + Mock Operational) ---
print("\n[STEP 5: Energy / Utilities (Real OSM + Mock Metrics)]")

utility_tags = {'power': ['substation', 'transformer'], 'water': ['water_tower', 'reservoir']}
try:
    utilities_gdf = ox.features_from_bbox(NORTH, SOUTH, EAST, WEST, tags=utility_tags)
    utilities_gdf = utilities_gdf[utilities_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()
    utilities_gdf.set_crs("EPSG:4326", inplace=True)

    # Mock Operational Metrics
    utilities_gdf['capacity_mock'] = utilities_gdf.apply(
        lambda x: np.random.randint(50, 500) if 'power' in x else np.random.randint(1, 100), axis=1
    ).astype(int)
    utilities_gdf['reliability_score_mock'] = np.clip(np.random.normal(0.8, 0.15, size=len(utilities_gdf)), 0.5, 0.99).round(2)
    
    # Select final columns
    utilities_gdf = utilities_gdf[['capacity_mock', 'reliability_score_mock', 'geometry']].copy()

except Exception as e:
    logging.warning(f"OSMnx Utilities failed, using fallback mock: {e}")
    # Fallback to pure mock
    N_FALLBACK = 150
    coords = [(77.59 + np.random.uniform(-0.15, 0.15), 12.97 + np.random.uniform(-0.15, 0.15)) for _ in range(N_FALLBACK)]
    utilities_gdf = gpd.GeoDataFrame({
        'capacity_mock': np.random.randint(5, 500, N_FALLBACK),
        'reliability_score_mock': np.clip(np.random.normal(0.8, 0.15, N_FALLBACK), 0.5, 0.99).round(2)
    }, geometry=[Point(xy) for xy in coords], crs="EPSG:4326")

utilities_gdf.to_file(os.path.join(UTILITIES_DIR, "utilities_bengaluru_final.geojson"), driver='GeoJSON')
logging.info("Utility data saved.")
print(f" ✅ Utility infrastructure points (N={len(utilities_gdf)}) saved to ../data/utilities/")


print("\n--- Phase 1, Step 1.5 Complete: All Multimodal Feature Files Generated ---")