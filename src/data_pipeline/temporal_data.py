"""
File: 04_temporal_data_extraction.py
Author: Gemini
Project: Urban Infrastructure & Spatial Planning - Bengaluru
Description:
    This script handles the temporal data layer (Weather and AQI). 
    It focuses on generating HALF-HOURLY data with simulated SPATIAL VARIATION
    (per H3 cell) for advanced GNN modeling, falling back to mock data if APIs fail.
"""
import os
import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND SETUP ---
os.makedirs("../logs/", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler('../logs/temporal_extraction.log')]
)
log = logging.getLogger(__name__)

log.info("--- Starting Temporal Data Extraction Framework (Weather/AQI Layer) ---")
print("--- Starting Temporal Data Extraction Framework (Weather/AQI Layer) ---")

# 1a. Reproducibility and Constants
np.random.seed(42)
output_dir_temporal = "../data/temporal/"
os.makedirs(output_dir_temporal, exist_ok=True)
log.info(f"Saving files to: {output_dir_temporal}")
print(f"  -> Saving files to: {output_dir_temporal}")

# API Keys and Locations
API_KEY_OWM = "9d33afe735de101aafcf0ca59982b014"
API_TOKEN_AQI = "fd107b04ec084847ba927a8cb472481ec355c1a5" # Replace with a real token
CITY_NAME = "Bengaluru,IN"
LAT, LON = 12.9716, 77.5946

# Time Period (ENHANCEMENT: High-Granularity Temporal Window)
N_TIMESTEPS = 48 # 24 hours * 2 (Half-hourly)
start_datetime = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=24)
timestamps = pd.date_range(start=start_datetime, periods=N_TIMESTEPS, freq='30min')

# Load H3 Indices from a dummy file or create a placeholder for spatial variation
# NOTE: In a complete pipeline, these should be loaded from the h3_gdf saved in Phase 1, Step 1.
# Here we use a small list to simulate the required spatial dimension (930 was your H3 count)
H3_INDICES = [f'8843a8e7a{i:x}' for i in range(930)] 
N_H3_CELLS = len(H3_INDICES)

# Generate temporal profile for diurnal variation
time_profile = np.sin(np.linspace(0, 2 * np.pi, N_TIMESTEPS))
temp_multiplier = (time_profile * 0.5) + 0.5 # Range 0 to 1
aqi_multiplier = (time_profile * 0.4) + 0.6 # Range 0.2 to 1.0


# --- 2. API FETCH / MOCK GENERATION FUNCTIONS (ENHANCED FOR SPATIAL/TEMPORAL GRIT) ---

def generate_mock_weather(timestamps, h3_indices, base_temp, base_wind):
    """Generates half-hourly mock weather data with spatial variation."""
    log.warning("Generating mock weather data with half-hourly and spatial variation.")
    
    # 1. Base temporal sequence
    base_temp_series = base_temp + (temp_multiplier * 5)
    
    all_data = []
    for t in range(len(timestamps)):
        ts = timestamps[t]
        
        # 2. Add spatial variation (simple linear relationship with H3 index proxy)
        spatial_noise = np.random.normal(0, 0.5, N_H3_CELLS) * (np.arange(N_H3_CELLS) / N_H3_CELLS)
        
        temp_c = (base_temp_series[t] + spatial_noise + np.random.normal(0, 0.5, N_H3_CELLS)).round(2)
        
        df_t = pd.DataFrame({
            'h3_index': h3_indices,
            'timestamp': ts,
            'temp_c': temp_c,
            'humidity': np.clip(np.random.normal(70, 5, N_H3_CELLS), 55, 85).astype(int),
            'wind_speed_mps': np.clip(np.random.normal(base_wind, 1, N_H3_CELLS), 0.5, 5.0).round(2),
            'pressure_hpa': np.random.randint(990, 1020, N_H3_CELLS),
            'is_raining': (np.random.rand(N_H3_CELLS) < 0.05).astype(int), # Low chance of rain per interval
            'weather_main': np.random.choice(['Clear', 'Clouds'], size=N_H3_CELLS, p=[0.7, 0.3]),
            'source': 'Mock_OWM_H3_Fallback'
        })
        all_data.append(df_t)
        
    df = pd.concat(all_data).reset_index(drop=True)
    return df

def fetch_weather_data(api_key, lat, lon, timestamps):
    """Attempts to fetch current weather data and uses it to seed mock H3 history."""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        log.info("Attempting to fetch current weather data from OpenWeatherMap API.")
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        
        current_temp = data['main']['temp']
        current_wind = data['wind']['speed']

        log.info(f"Weather data successfully fetched. Current Temp: {current_temp}°C. Generating synthetic half-hourly H3 history.")
        
        # Call mock generator with real current data as seeds
        return generate_mock_weather(timestamps, H3_INDICES, current_temp, current_wind)

    except Exception as e:
        log.error(f"Weather API fetch failed: {e}")
        print("  ⚠️ Weather API fetch failed. Generating pure mock data...")
        # Fallback to pure mock with standard seeds
        return generate_mock_weather(timestamps, H3_INDICES, base_temp=28, base_wind=2.5)


def generate_mock_aqi(timestamps, h3_indices, base_aqi):
    """Generates half-hourly mock AQI data with spatial variation."""
    log.warning("Generating mock AQI data with half-hourly and spatial variation.")
    
    # 1. Base temporal sequence (higher during day/commute times)
    base_aqi_series = base_aqi + (aqi_multiplier * 50)
    
    all_data = []
    for t in range(len(timestamps)):
        ts = timestamps[t]
        
        # 2. Add spatial variation (e.g., higher AQI near center/dense H3 codes)
        spatial_factor_aqi = (np.arange(N_H3_CELLS) / N_H3_CELLS) * 0.4 + 0.8 # Range 0.8 to 1.2
        
        aqi_level = base_aqi_series[t] * spatial_factor_aqi
        
        df_t = pd.DataFrame({
            'h3_index': h3_indices,
            'timestamp': ts,
            'aqi': np.clip(aqi_level + np.random.normal(0, 10, N_H3_CELLS), 50, 200).astype(int),
            'source': 'Mock_WAQI_H3_Fallback'
        })
        all_data.append(df_t)

    df = pd.concat(all_data).reset_index(drop=True)
    return df


def fetch_aqi_data(api_token, city_name, timestamps):
    """Attempts to fetch current AQI data and uses it to seed mock H3 history."""
    
    if api_token == "fd107b04ec084847ba927a8cb472481ec355c1a5": # Placeholder check
        log.warning("WAQI API token is a placeholder. Skipping API call.")
        print("  ⚠️ AQI API token is a placeholder. Generating mock data...")
        return generate_mock_aqi(timestamps, H3_INDICES, base_aqi=70) # Fallback to standard seeds
        
    url = f"https://api.waqi.info/feed/{city_name.split(',')[0].lower()}/?token={api_token}"
    
    try:
        log.info("Attempting to fetch current AQI data from WAQI API.")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        aqi_json = response.json()
        
        if aqi_json.get('status') == 'ok':
            current_aqi = aqi_json['data']['aqi']
            log.info(f"AQI data successfully fetched. Current AQI: {current_aqi}. Generating synthetic half-hourly H3 history.")
            return generate_mock_aqi(timestamps, H3_INDICES, current_aqi) # Seeded mock
        else:
            raise Exception(f"WAQI response status not OK: {aqi_json.get('status')}")

    except Exception as e:
        log.error(f"AQI API fetch failed: {e}")
        print("  ⚠️ AQI API fetch failed. Generating pure mock data...")
        return generate_mock_aqi(timestamps, H3_INDICES, base_aqi=70) # Fallback to standard seeds


# --- 3. EXECUTION ---
print("\n[STEP 3: Fetching Data or using Mock Fallback]")

# 3a. Execute fetches with fallback
weather_df = fetch_weather_data(API_KEY_OWM, LAT, LON, timestamps)
aqi_df = fetch_aqi_data(API_TOKEN_AQI, CITY_NAME, timestamps)

# 3b. Data Consolidation and Cleaning
log.info("Merging weather and AQI datasets...")

# Merge on spatial and temporal index
combined_df = pd.merge(aqi_df, weather_df, on=['h3_index', 'timestamp'], how='inner', suffixes=('_aqi', '_weather'))

# --- Final Cleanup ---
combined_df = combined_df.drop(columns=['source_weather'])
combined_df = combined_df.rename(columns={'source_aqi': 'source'})

combined_df['city'] = "Bengaluru"
combined_df['region_code'] = "BBMP"

# Set MultiIndex for GNN compatibility
combined_df = combined_df.set_index(['h3_index', 'timestamp'])


# --- 4. DATA SAVING AND METADATA ---
print("\n[STEP 4: Saving Data and Metadata]")

# Save the combined data file
filename = "temporal_data_h3_48h.csv"
filepath = os.path.join(output_dir_temporal, filename)
# Save with index for H3/Timestamp
combined_df.to_csv(filepath) 

log.info(f"Temporal data saved to: {filepath}")
print(f"  ✅ Data saved to: {filepath}")
print(f"  Shape: {combined_df.shape}")

# Export metadata
source_type = 'API Seeded (Synthetic H3)' if 'Synthetic' in combined_df['source'].iloc[0] else 'Pure Mock H3'
metadata = {
    "city": "Bengaluru",
    "n_timesteps": N_TIMESTEPS,
    "spatial_unit": "H3 Cell (N_H3_CELLS)",
    "time_granularity": "30-min interval",
    "generated_on": datetime.now().isoformat(),
    "source_type": source_type,
    "variables": list(combined_df.columns),
    "notes": "High-granularity temporal data with simulated spatial (H3) and temporal (Diurnal) variation, suitable for GNN input."
}
metadata_path = os.path.join(output_dir_temporal, "temporal_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
log.info(f"Metadata exported to {metadata_path}")
print(f"  ✅ Metadata exported to {metadata_path}")

log.info("--- Temporal Dynamics Data Acquisition Complete ---")
print("\n--- Temporal Dynamics Data Acquisition Complete ---")