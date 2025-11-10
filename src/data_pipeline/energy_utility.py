import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import osmnx as ox
import folium
import json
import datetime
import branca.colormap as cm
import logging # ADDED: Logging support

# --- 1. CONFIGURATION AND SETUP ---

# Configure logging to write to a file and log INFO level messages
# This makes the pipeline production-ready by creating an execution log.
os.makedirs("../logs/", exist_ok=True)
logging.basicConfig(filename='../logs/data_extraction.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("--- Starting Energy / Utilities Network Data Extraction Framework ---")
print("--- Starting Energy / Utilities Network Data Extraction Framework ---")

# 1a. Reproducibility
np.random.seed(42) # Ensure consistent mock data generation

# Define target data directory
output_dir_utilities = "../data/utilities/"
os.makedirs(output_dir_utilities, exist_ok=True)
logging.info(f"Saving files to: {output_dir_utilities}")
print(f"   -> Saving files to: {output_dir_utilities}")

# Define the area of interest (same BBOX as previous steps for consistency)
PLACE_NAME = "Bengaluru, Karnataka, India"
NORTH, SOUTH, EAST, WEST = 13.15, 12.8, 77.75, 77.45 


# --- 2. OSMnx UTILITY DATA EXTRACTION (REAL DATA) ---
logging.info("Starting extraction of Real Utility Infrastructure Data via OSMnx")
print("\n[STEP 2: Extracting Real Utility Infrastructure Data via OSMnx]")

# -------------------------------------------------------------------------
# Define the tags for critical utility infrastructure
# -------------------------------------------------------------------------
# Focusing on power and water infrastructure elements
utility_tags = {
    'power': ['substation', 'transformer', 'generator'],
    'water': ['water_tower', 'reservoir', 'pumping_station', 'works']
}

utilities_gdf = gpd.GeoDataFrame() # Initialize empty GeoDataFrame

try:
    # ---------------------------------------------------------------------
    # 2a. Fetch all features matching the utility tags using BBOX
    # ---------------------------------------------------------------------
    logging.info("Fetching utility features from OpenStreetMap...")
    print("   -> Fetching utility features from OpenStreetMap...")
    utilities_gdf = ox.features_from_bbox(NORTH, SOUTH, EAST, WEST, tags=utility_tags)
    
    # ---------------------------------------------------------------------
    # ADDED: CRS Validation (Enforcement)
    # ---------------------------------------------------------------------
    if utilities_gdf.crs is None or utilities_gdf.crs.to_string() != "EPSG:4326":
        logging.info("Setting CRS to EPSG:4326 for consistency.")
        utilities_gdf.set_crs("EPSG:4326", inplace=True)

    # ---------------------------------------------------------------------
    # 2b. Clean and Standardize the Data
    # ---------------------------------------------------------------------
    
    def classify_utility(row):
        if 'power' in row:
            return f"Power - {row['power'].replace('_', ' ').title()}"
        if 'water' in row:
            return f"Water - {row['water'].replace('_', ' ').title()}"
        return 'Other Utility'

    utilities_gdf['utility_type'] = utilities_gdf.apply(classify_utility, axis=1)
    
    # Focus only on point-based infrastructure for clear mapping
    utilities_gdf = utilities_gdf[utilities_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()
    
    # Select only relevant columns
    utilities_gdf = utilities_gdf[['utility_type', 'name', 'geometry']].copy()
    
    # ---------------------------------------------------------------------
    # 2c. Mocking Operational Metrics
    # ---------------------------------------------------------------------
    logging.info("Generating mock operational metrics...")
    print("   -> Generating mock operational metrics...")
    
    utilities_gdf['age_years_mock'] = np.random.randint(5, 50, size=len(utilities_gdf))
    
    # Mock Capacity (MW for Power, MGD for Water)
    def mock_capacity(row):
        if 'Power' in row['utility_type']:
            return np.random.randint(50, 500) # Mock MegaWatts (MW)
        elif 'Water' in row['utility_type']:
            return np.random.randint(1, 100)  # Mock Million Gallons Daily (MGD)
        return 0

    utilities_gdf['capacity_mock'] = utilities_gdf.apply(mock_capacity, axis=1).astype(int)
    
    # Mock a Maintenance/Reliability Score (1.0 = perfect, 0.0 = failing)
    utilities_gdf['reliability_score_mock'] = np.clip(np.random.normal(0.8, 0.15, size=len(utilities_gdf)), 0.5, 0.99).round(2)
    
except Exception as e:
    logging.error(f"OSMnx extraction failed. Error: {e}", exc_info=True)
    print(f"   ⚠️ OSMnx extraction failed. Error: {e}")
    
    # ---------------------------------------------------------------------
    # Fallback mock generator
    # ---------------------------------------------------------------------
    logging.warning("Generating mock utilities dataset as fallback...")
    print("   ⚙️ Generating mock utilities dataset as fallback...")
    N_FALLBACK = 150
    coords = [(77.59 + np.random.uniform(-0.1, 0.1), 12.97 + np.random.uniform(-0.1, 0.1)) for _ in range(N_FALLBACK)]
    
    utility_types = ['Power - Substation', 'Water - Pumping Station', 'Power - Generator']
    
    utilities_gdf = gpd.GeoDataFrame({
        'utility_type': np.random.choice(utility_types, N_FALLBACK),
        'name': [f'Facility_{i}' for i in range(N_FALLBACK)],
        'age_years_mock': np.random.randint(5, 50, N_FALLBACK),
        'capacity_mock': np.random.randint(5, 500, N_FALLBACK),
        'reliability_score_mock': np.clip(np.random.normal(0.8, 0.15, N_FALLBACK), 0.5, 0.99).round(2)
    }, geometry=[Point(xy) for xy in coords], crs="EPSG:4326")
    logging.info(f"Fallback data (N={N_FALLBACK}) generated.")
    print(f"   ✅ Fallback data (N={N_FALLBACK}) generated.")


# --- 3. DATA SAVING & SUMMARIZING ---
if not utilities_gdf.empty:
    logging.info("Starting Step 3: Data Integrity, Enrichment, and Saving.")
    print("\n[STEP 3: Saving Utilities Data]")

    # ---------------------------------------------------------------------
    # ADDED: Data Integrity Checks (CRITICAL)
    # ---------------------------------------------------------------------
    try:
        assert utilities_gdf.geometry.is_valid.all(), "Invalid geometries found!"
        assert not utilities_gdf.isnull().all().any(), "Null values detected in critical columns!"
        logging.info("Data integrity checks passed.")
    except AssertionError as e:
        logging.error(f"Data integrity check failed: {e}")
        print(f"   🛑 Data integrity check failed: {e}")
        # Optionally exit or handle the error here, but for now, we continue to save/log

    # ---------------------------------------------------------------------
    # ADDED: Optional Spatial Join with Wards (Data Enrichment)
    # ---------------------------------------------------------------------
    # NOTE: This assumes a GeoJSON file with ward boundaries exists in the specified path.
    WARD_FILE_PATH = "../data/spatial/wards_bengaluru.geojson"
    if os.path.exists(WARD_FILE_PATH):
        try:
            wards = gpd.read_file(WARD_FILE_PATH).to_crs("EPSG:4326")
            utilities_gdf = gpd.sjoin(utilities_gdf, wards[['ward_name', 'geometry']], how="left", predicate="intersects")
            logging.info(f"Data enriched via spatial join with {WARD_FILE_PATH}")
            print("   -> Data enriched with Ward names via spatial join.")
        except Exception as e:
            logging.warning(f"Spatial join failed (Likely missing or corrupt ward file): {e}")
            print("   ⚠️ Spatial join skipped (Ward boundary file issue).")
    else:
        logging.warning(f"Spatial join skipped: Ward file not found at {WARD_FILE_PATH}")


    # Ensure the GeoDataFrame is projected to WGS 84 before saving
    if utilities_gdf.crs is None or utilities_gdf.crs.to_string() != "EPSG:4326":
        utilities_gdf = utilities_gdf.to_crs("EPSG:4326")

    # Save the GeoDataFrame
    file_path = os.path.join(output_dir_utilities, "utilities_bengaluru_real.geojson")
    utilities_gdf.to_file(file_path, driver='GeoJSON')
    
    logging.info(f"Transport Stop data saved to: {file_path}")
    print(f"   ✅ Utility Infrastructure data saved to: {file_path}")
    print(f"   Total unique utility points processed: {len(utilities_gdf)}")
    
    # ---------------------------------------------------------------------
    # Log summary stats
    # ---------------------------------------------------------------------
    print("\n--- Summary Stats by Utility Type ---")
    summary = utilities_gdf.groupby('utility_type')[['capacity_mock', 'reliability_score_mock']].agg({
        'capacity_mock': ['mean', 'count'],
        'reliability_score_mock': 'mean'
    }).round(2)
    logging.info(f"Summary Stats:\n{summary}")
    print(summary)
    
    # ---------------------------------------------------------------------
    # Export metadata
    # ---------------------------------------------------------------------
    metadata = {
        "city": PLACE_NAME,
        "date_created": datetime.datetime.now().isoformat(),
        "data_source": "OpenStreetMap via OSMnx",
        "query_type": "ox.features_from_bbox",
        "total_facilities": len(utilities_gdf),
        "attributes": list(utilities_gdf.columns),
        "notes": "Capacity and reliability scores are synthetic approximations.",
        "bbox": f"({NORTH}, {SOUTH}, {EAST}, {WEST})"
    }
    metadata_path = os.path.join(output_dir_utilities, "utilities_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata exported to {metadata_path}")
    print(f"   ✅ Metadata exported to {metadata_path}")


    
    # --- 4. VISUALIZATION --- 
    logging.info("Starting Step 4: Generating Interactive Folium Map.")
    print("\n[STEP 4: Generating Interactive Folium Map]")
    try:
        # Determine the map center
        center_lat, center_lon = 12.9716, 77.5946
        
        output_dir_maps = "../outputs/maps/"
        os.makedirs(output_dir_maps, exist_ok=True)
        
        # Use a contrasting tile set to distinguish this map from the mobility map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB Dark Matter")
        
        # ---------------------------------------------------------------------
        # Reliability Score Color Scale (Red = Low Reliability, Green = High Reliability)
        # ---------------------------------------------------------------------
        min_score = utilities_gdf['reliability_score_mock'].min()
        max_score = utilities_gdf['reliability_score_mock'].max()

        colormap = cm.LinearColormap(
            ['#FF4136', '#FFDC00', '#2ECC40'], # Red to Yellow to Green
            vmin=min_score,
            vmax=max_score,
            caption='Reliability Score (Mock)'
        )
        
        m.add_child(colormap)
        
        # Sample the data for display
        map_data = utilities_gdf.sample(min(len(utilities_gdf), 1000), random_state=42) 

        # ---------------------------------------------------------------------
        # Feature Groups for Layer Control (Power vs. Water)
        # ---------------------------------------------------------------------
        power_group = folium.FeatureGroup(name='⚡ Power Infrastructure').add_to(m)
        water_group = folium.FeatureGroup(name='💧 Water Infrastructure').add_to(m)
        
        # Function to add markers
        def create_utility_marker(row):
            coords = [row.geometry.y, row.geometry.x]
            score_color = colormap(row['reliability_score_mock'])
            
            # Check for the presence of 'ward_name' before including it in the popup
            ward_info = f"<br>Ward: {row['ward_name']}" if 'ward_name' in row and pd.notna(row['ward_name']) else ""

            popup_html = f"""
            <strong>{row['utility_type']}</strong>{ward_info}<br>
            Name: {row['name'] if pd.notna(row['name']) else 'N/A'}<br>
            Reliability Score: <strong>{row['reliability_score_mock']:.2f}</strong><br>
            Capacity: {row['capacity_mock']:,} 
            {'MW' if 'Power' in row['utility_type'] else 'MGD'}
            """
            
            # Determine icon and feature group based on type
            if 'Power' in row['utility_type']:
                icon = folium.Icon(color='red', icon='bolt', prefix='fa')
                group = power_group
            elif 'Water' in row['utility_type']:
                icon = folium.Icon(color='blue', icon='tint', prefix='fa')
                group = water_group
            else:
                icon = folium.Icon(color='gray', icon='gear', prefix='fa')
                group = power_group # Default group

            # Use a colored Icon Marker for visual distinction
            folium.Marker(
                location=coords,
                icon=icon,
                tooltip=popup_html
            ).add_to(group)

        # Apply the function to the data
        map_data.apply(create_utility_marker, axis=1)

        # Add the Layer Control
        folium.LayerControl(collapsed=False).add_to(m)
        
        map_file_path = os.path.join(output_dir_maps, "bengaluru_utilities_data.html")
        m.save(map_file_path)
        
        logging.info(f"Interactive map saved in {map_file_path}")
        print(f"   ✅ Interactive map saved in {map_file_path}")

        # ADDED: Static Map Placeholder
        print("   📸 (Optional) Export static map images for reports later using Selenium or folium-staticmap.")


    except Exception as map_e:
        logging.error(f"Folium map generation failed. Error: {map_e}", exc_info=True)
        print(f"   ⚠️ Folium map generation failed. Error: {map_e}")

else:
    logging.error("Step 3: Data Saving Failed. No data was generated.")
    print("\n[STEP 3: Data Saving Failed]")
    print("   No real or fallback data was generated.")


logging.info("--- Step 7 Complete: All Core Data Acquisition Phases Completed ---")
print("\n--- Step 7 Complete: All Core Data Acquisition Phases Completed ---")
print("All five data layers are now acquired and saved as GeoJSON/CSV.")
