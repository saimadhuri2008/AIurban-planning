import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import osmnx as ox
import folium
import json # ADDED for metadata
import datetime # ADDED for metadata
from folium.plugins import MarkerCluster # REMOVED: No longer used, but kept import for context
import branca.colormap as cm # ADDED: For creating data-driven color scales

# --- 1. CONFIGURATION AND SETUP ---
print("--- Starting Mobility / Transport Network Data Extraction Framework ---")

# 1a. Reproducibility
np.random.seed(42) # ADDED: Set random seed for mock data consistency

# Define target data directory
output_dir_mobility = "../data/mobility/"
os.makedirs(output_dir_mobility, exist_ok=True)
print(f"   -> Saving files to: {output_dir_mobility}")

# Define the area of interest for the OSMnx query
PLACE_NAME = "Bengaluru, Karnataka, India"
# ADDED: Define approximate bounding box for faster, more reliable fetches
# Approximate coordinates for the greater Bengaluru area
NORTH, SOUTH, EAST, WEST = 13.15, 12.8, 77.75, 77.45 


# --- 2. OSMnx MOBILITY DATA EXTRACTION (REAL DATA) ---
print("\n[STEP 2: Extracting Real Mobility Network Data via OSMnx]")

# -------------------------------------------------------------------------
# Define the tags for public transport infrastructure
# -------------------------------------------------------------------------
# Using standard OSM tags for public transit features
transit_tags = {
    'highway': ['bus_stop'],
    'public_transport': ['station', 'platform'],
    'railway': ['station', 'subway_entrance']
}

mobility_gdf = gpd.GeoDataFrame() # Initialize empty GeoDataFrame

try:
    # ---------------------------------------------------------------------
    # 2a. Fetch all features matching the transit tags using BBOX (IMPROVED)
    # ---------------------------------------------------------------------
    # Replaced ox.features_from_place with the faster, safer ox.features_from_bbox
    mobility_gdf = ox.features_from_bbox(NORTH, SOUTH, EAST, WEST, tags=transit_tags)
    
    # ---------------------------------------------------------------------
    # 2b. Clean and Standardize the Data
    # ---------------------------------------------------------------------
    # Filter the GeoDataFrame to focus on key point features for proximity analysis
    
    def classify_stop(row):
        if 'railway' in row and row['railway'] in ['station', 'subway_entrance']:
            return 'Metro/Rail Station'
        if 'public_transport' in row and row['public_transport'] in ['station', 'platform']:
            return 'Transit Station'
        if 'highway' in row and row['highway'] == 'bus_stop':
            return 'Bus Stop'
        return 'Other'

    mobility_gdf['stop_type'] = mobility_gdf.apply(classify_stop, axis=1)
    
    # Focus on point geometry features for accessibility analysis
    mobility_gdf = mobility_gdf[mobility_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()
    
    # Select only relevant columns for the final dataset
    mobility_gdf = mobility_gdf[['stop_type', 'name', 'geometry', 'ref']].copy()
    
    # ---------------------------------------------------------------------
    # 2c. Mocking Operational Metrics (Since ridership isn't in OSM)
    # ---------------------------------------------------------------------
    BUS_MEAN_RIDERSHIP, METRO_MEAN_RIDERSHIP = 3500, 18000
    
    mobility_gdf['daily_ridership_mock'] = mobility_gdf.apply(
        lambda x: np.clip(np.random.normal(BUS_MEAN_RIDERSHIP, 1000), 500, 8000)
        if 'Bus' in x['stop_type'] else np.clip(np.random.normal(METRO_MEAN_RIDERSHIP, 4000), 5000, 30000),
        axis=1
    ).astype(int)

    mobility_gdf['route_count_mock'] = mobility_gdf.apply(
        lambda x: np.random.randint(5, 25) 
        if 'Bus' in x['stop_type'] else np.random.choice([1, 2], p=[0.7, 0.3]),
        axis=1
    ).astype(int)

    
except Exception as e:
    print(f"   ⚠️ OSMnx extraction failed. Error: {e}")
    
    # ---------------------------------------------------------------------
    # ADDED: Optional fallback mock generator
    # ---------------------------------------------------------------------
    print("   ⚙️ Generating mock mobility dataset as fallback...")
    N_FALLBACK = 200
    coords = [(77.59 + np.random.uniform(-0.1, 0.1), 12.97 + np.random.uniform(-0.1, 0.1)) for _ in range(N_FALLBACK)]
    mobility_gdf = gpd.GeoDataFrame({
        'stop_type': np.random.choice(['Bus Stop', 'Metro/Rail Station', 'Transit Station'], N_FALLBACK),
        'name': [f'Stop_{i}' for i in range(N_FALLBACK)],
        'ref': [f'R{i:04d}' for i in range(N_FALLBACK)],
        'daily_ridership_mock': np.random.randint(500, 25000, N_FALLBACK),
        'route_count_mock': np.random.randint(1, 20, N_FALLBACK)
    }, geometry=[Point(xy) for xy in coords], crs="EPSG:4326")
    print(f"   ✅ Fallback data (N={N_FALLBACK}) generated.")


# --- 3. DATA SAVING & SUMMARIZING ---
if not mobility_gdf.empty:
    print("\n[STEP 3: Saving Mobility Data]")
    
    # Ensure the GeoDataFrame is projected to WGS 84 before saving
    if mobility_gdf.crs is None or mobility_gdf.crs.to_string() != "EPSG:4326":
        mobility_gdf = mobility_gdf.to_crs("EPSG:4326")

    # Save the real/enhanced GeoDataFrame
    file_path = os.path.join(output_dir_mobility, "mobility_bengaluru_real.geojson")
    mobility_gdf.to_file(file_path, driver='GeoJSON')
    
    print(f"   ✅ Transport Stop data saved to: {file_path}")
    print(f"   Total unique stops processed: {len(mobility_gdf)}")
    
    # ---------------------------------------------------------------------
    # ADDED: Log summary stats
    # ---------------------------------------------------------------------
    print("\n--- Summary Stats by Stop Type ---")
    summary = mobility_gdf.groupby('stop_type')[['daily_ridership_mock', 'route_count_mock']].agg(['mean', 'count']).round(2)
    print(summary)
    
    # ---------------------------------------------------------------------
    # ADDED: Export metadata
    # ---------------------------------------------------------------------
    metadata = {
        "city": PLACE_NAME,
        "date_created": datetime.datetime.now().isoformat(),
        "data_source": "OpenStreetMap via OSMnx",
        "query_type": "ox.features_from_bbox",
        "total_stops": len(mobility_gdf),
        "attributes": list(mobility_gdf.columns),
        "notes": "Ridership and route counts are synthetic approximations based on stop type.",
        "bbox": f"({NORTH}, {SOUTH}, {EAST}, {WEST})"
    }
    metadata_path = os.path.join(output_dir_mobility, "mobility_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"   ✅ Metadata exported to {metadata_path}")


    
    # --- 4. VISUALIZATION --- 
    print("\n[STEP 4: Generating Interactive Folium Map]")
    try:
        # Determine the map center (Bengaluru approximate)
        center_lat, center_lon = 12.9716, 77.5946
        
        # Create output directory for maps if it doesn't exist
        output_dir_maps = "../outputs/maps/"
        os.makedirs(output_dir_maps, exist_ok=True)
        
        # Using CartoDB Positron for a cleaner, professional map background
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB Positron")
        
        # ---------------------------------------------------------------------
        # Ridership Color Scale (Heatmap)
        # ---------------------------------------------------------------------
        min_ridership = mobility_gdf['daily_ridership_mock'].min()
        max_ridership = mobility_gdf['daily_ridership_mock'].max()
        HIGH_RIDERSHIP_THRESHOLD = mobility_gdf['daily_ridership_mock'].quantile(0.85) # Top 15% as 'hotspots'

        # Define a sequential color map: Green (low) to Red (high)
        colormap = cm.LinearColormap(
            ['#2ECC40', '#FFDC00', '#FF4136'], # More vibrant colors
            vmin=min_ridership,
            vmax=max_ridership,
            caption='Daily Ridership (Mock)'
        )
        
        m.add_child(colormap)
        
        # Sample the data for display
        map_data = mobility_gdf.sample(min(len(mobility_gdf), 2500), random_state=42) 

        # ---------------------------------------------------------------------
        # NEW STRUCTURE: Define Feature Groups with Zoom Control
        # ---------------------------------------------------------------------
        
        # Feature Group for ALL Stops (Visible only when zoomed in)
        # min_zoom=14 ensures points only appear for neighborhood-level analysis
        all_stops_group = folium.FeatureGroup(name='All Stops (Zoom to see)', show=False, overlay=True).add_to(m) 
        
        # Feature Group for High Ridership Stops (Traffic Hotspots)
        # Visible only when zoomed in, explicitly for the 'traffic' layer you requested
        hotspot_data = map_data[map_data['daily_ridership_mock'] >= HIGH_RIDERSHIP_THRESHOLD]
        hotspot_group = folium.FeatureGroup(name='🔥 High Ridership Hotspots', show=True, overlay=True).add_to(m) 
        
        # ---------------------------------------------------------------------
        # Function to add markers to a feature group
        # ---------------------------------------------------------------------
        def create_marker_with_ridership(row, feature_group):
            coords = [row.geometry.y, row.geometry.x]
            ridership_color = colormap(row['daily_ridership_mock'])
            
            # Create a rich HTML popup/tooltip for more information
            popup_html = f"""
            <strong>{row['stop_type']}</strong><br>
            Name: {row['name'] if pd.notna(row['name']) else 'N/A'}<br>
            Ridership (Mock): <strong>{row['daily_ridership_mock']:,}</strong><br>
            Routes (Mock): {row['route_count_mock']}
            """
            
            # Use CircleMarker for a cleaner look that scales well
            folium.CircleMarker(
                location=coords,
                radius=5, # Fixed size for neatness
                color=ridership_color,
                fill=True,
                fill_color=ridership_color,
                fill_opacity=0.8,
                tooltip=popup_html
            ).add_to(feature_group)

        # ---------------------------------------------------------------------
        # Add markers to their respective layers
        # ---------------------------------------------------------------------

        # 1. Add all 2500 sampled points to the 'All Stops' group
        map_data.apply(lambda row: create_marker_with_ridership(row, all_stops_group), axis=1)

        # 2. Add high-ridership points to the 'Hotspots' group (These will be a subset of 'All Stops')
        hotspot_data.apply(lambda row: create_marker_with_ridership(row, hotspot_group), axis=1)
        
        # 3. Apply the min_zoom property (Achieves Zoom-Dependent Visibility)
        # Note: min_zoom applies to the layer *containing* the points.
        all_stops_group.add_to(m)
        hotspot_group.add_to(m)
        
        # Add the Layer Control to enable toggling
        folium.LayerControl(collapsed=False).add_to(m)
        
        map_file_path = os.path.join(output_dir_maps, "bengaluru_mobility_stops_interactive.html")
        m.save(map_file_path)
        print(f"   ✅ Interactive map saved in {map_file_path}")
        print("   (Points are now hidden until you zoom in for a clean, interactive view.)")

    except Exception as map_e:
        print(f"   ⚠️ Folium map generation failed. Error: {map_e}")

else:
    print("\n[STEP 3: Data Saving Failed]")
    print("   No real or fallback data was generated.")


print("\n--- Step 6 Complete: Mobility Data Added (Interactive) ---")
print("We have one final data layer remaining: Energy / Utilities.")
