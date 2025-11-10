import os
import pandas as pd
import geopandas as gpd

# --- 1. CONFIGURATION AND SETUP ---
print("--- Starting Remote Sensing Data Extraction Framework ---")

# Define target data directory
output_dir_remote = "../data/remote_sensing/"
os.makedirs(output_dir_remote, exist_ok=True)
print(f"   -> Saving files to: {output_dir_remote}")


# --- 2. VIIRS NIGHTTIME LIGHTS (NTL) EXTRACTION ---
# NTL data is a strong proxy for economic activity, energy use, and population density.
# This data is usually downloaded as a GeoTIFF (raster file) from NASA/NOAA archives
# (e.g., VIIRS DNB monthly composites). It is then clipped and processed to the study area.
print("\n[STEP 2: VIIRS Nighttime Lights (NTL) Data]")

def process_ntl_data(output_path):
    """
    Mocks the process of downloading, clipping, and rasterizing NTL data.

    In a full pipeline, this would use libraries like rasterio and geopandas
    to clip the global NTL raster file to the Bengaluru boundary and convert
    it to a GeoTIFF or aggregated vector data.
    """
    # Mocking the creation of a simplified NTL feature layer (for demonstration)
    data = {
        'grid_id': [1, 2, 3, 4],
        'avg_ntl_intensity': [1.2, 5.5, 0.8, 4.1],
        'geometry': [
            'POINT (77.5 13.0)', 'POINT (77.6 12.9)', 'POINT (77.4 12.8)', 'POINT (77.7 13.1)'
        ] # Placeholder geometries
    }
    # Using a simplified GeoDataFrame structure for file persistence
    ntl_gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry=gpd.points_from_xy([77.5, 77.6, 77.4, 77.7], [13.0, 12.9, 12.8, 13.1]), crs="EPSG:4326")

    # Save the mock file
    file_path = os.path.join(output_path, "ntl_bengaluru_mock.geojson")
    ntl_gdf.to_file(file_path, driver='GeoJSON')
    print(f"   ✅ NTL framework established. Mock data saved to: {file_path}")
    print("      -> ACTION REQUIRED: Replace mock data with real clipped NTL raster/vector data.")
    return ntl_gdf

ntl_data = process_ntl_data(output_dir_remote)


# --- 3. NORMALIZED DIFFERENCE VEGETATION INDEX (NDVI) ---
# NDVI measures greenness, crucial for assessing environmental quality and urban heat islands.
print("\n[STEP 3: NDVI (Vegetation Index) Data]")

def process_ndvi_data(output_path):
    """
    Mocks the process of fetching and processing Sentinel/Landsat derived NDVI data.

    In a full pipeline, this involves fetching satellite imagery, cloud masking,
    calculating the NDVI band, and saving the final raster or vector aggregation.
    """
    # Mocking the creation of a simplified NDVI feature layer
    data = {
        'grid_id': [1, 2, 3, 4],
        'avg_ndvi': [0.15, 0.65, 0.05, 0.40], # Values between -1 (water) and 1 (dense vegetation)
        'geometry': ntl_data['geometry'].iloc[:4].tolist()
    }
    ndvi_df = gpd.GeoDataFrame(data, geometry=ntl_data['geometry'].iloc[:4].tolist(), crs="EPSG:4326")

    # Save the mock file
    file_path = os.path.join(output_path, "ndvi_bengaluru_mock.geojson")
    ndvi_df.to_file(file_path, driver='GeoJSON')
    print(f"   ✅ NDVI framework established. Mock data saved to: {file_path}")
    print("      -> ACTION REQUIRED: Replace mock data with real clipped NDVI raster/vector data.")
    return ndvi_df

ndvi_data = process_ndvi_data(output_dir_remote)


print("\n--- Step 3 Complete: Remote Sensing Framework Established ---")
print("Next, we will focus on the two remaining layers: Socioeconomic/Equity (Census) and Policy.")
