"""
temporal_data.py
Final corrected script for Phase-0 temporal data generation (weather + AQI)
- Uses H3 to create spatial cells around a lat/lon
- Generates half-hourly mock (or API-seeded) weather & AQI per H3 cell
- Saves data to Parquet with metadata
Requirements:
    pip install pandas numpy requests pyarrow h3 tqdm
Usage:
    Set env vars OWM_API_KEY and WAQI_TOKEN if you want real seeding; otherwise script uses mock seeds.
    python src/data_pipeline/temporal_data.py
"""

import os
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# --- Configuration & Logging ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "temporal")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "temporal_extraction.log"))
    ]
)
log = logging.getLogger("temporal_data")

# --- Environment / User config (do NOT hardcode secrets) ---
API_KEY_OWM = os.environ.get("OWM_API_KEY")    # OpenWeatherMap (current weather) - optional
API_TOKEN_AQI = os.environ.get("WAQI_TOKEN")   # WAQI token - optional
CITY_NAME = os.environ.get("CITY_NAME", "Bengaluru,IN")
LAT = float(os.environ.get("CITY_LAT", 12.9716))
LON = float(os.environ.get("CITY_LON", 77.5946))
TIMEZONE = os.environ.get("CITY_TZ", "Asia/Kolkata")

# Temporal config
N_TIMESTEPS = int(os.environ.get("N_TIMESTEPS", 48))  # half-hourly for 24h: 48
FREQ = os.environ.get("FREQ", "30min")

# H3 config
try:
    import h3
except Exception:
    h3 = None

H3_RESOLUTION = int(os.environ.get("H3_RESOLUTION", 8))
H3_K_RING = int(os.environ.get("H3_K_RING", 4))  # smaller by default; adjust as needed

# Output filenames
PARQUET_FILENAME = "temporal_data_h3_48h.parquet"
METADATA_FILENAME = "temporal_metadata.json"

# ---------------- Helper: H3 cell generation (robust) ----------------
def get_h3_cells_around_point(lat, lon, res=8, k_ring=4):
    """
    Return a list of H3 indices around lat/lon. Works with common h3 APIs:
    - latlng_to_cell + grid_disk / grid_ring
    - polygon_to_cells / polyfill fallback
    If h3 isn't installed, returns dummy indices (development mode).
    """
    if h3 is None:
        log.warning("h3 library not found. Returning dummy indices for development.")
        return [f"dummy_{i}" for i in range(max(100, k_ring * 10))]

    # Preferred: latlng_to_cell + grid_disk
    if hasattr(h3, "latlng_to_cell"):
        try:
            center = h3.latlng_to_cell(lat, lon, res)
            if hasattr(h3, "grid_disk"):
                cells = list(h3.grid_disk(center, k_ring))
            elif hasattr(h3, "grid_ring"):
                # include center + rings
                cells = []
                for r in range(k_ring + 1):
                    cells.extend(list(h3.grid_ring(center, r)))
                # deduplicate while preserving order
                seen = set()
                unique = []
                for c in cells:
                    if c not in seen:
                        seen.add(c)
                        unique.append(c)
                cells = unique
            else:
                cells = [center]
            return [str(c) for c in cells]
        except Exception as e:
            log.error("Error using latlng_to_cell/grid_disk: %s", e)

    # Fallback: polygon polyfill style using a small bbox
    if hasattr(h3, "polygon_to_cells") or hasattr(h3, "polyfill"):
        try:
            dlat, dlon = 0.03, 0.03  # ~3km box - adjust if necessary
            polygon = [[lon - dlon, lat - dlat],
                       [lon - dlon, lat + dlat],
                       [lon + dlon, lat + dlat],
                       [lon + dlon, lat - dlat],
                       [lon - dlon, lat - dlat]]
            if hasattr(h3, "polygon_to_cells"):
                cells = list(h3.polygon_to_cells(polygon, res))
            else:
                cells = list(h3.polyfill(polygon, res))
            return [str(c) for c in cells]
        except Exception as e:
            log.error("polygon_to_cells/polyfill failed: %s", e)

    # Final fallback: return a small dummy list
    log.warning("Could not generate H3 cells with installed API. Using dummy list.")
    return [f"dummy_{i}" for i in range(max(100, k_ring * 10))]

# ---------------- Diurnal / generator functions ----------------
def generate_mock_weather(timestamps, h3_indices, base_temp=28.0, base_wind=2.5):
    """
    Generates a DataFrame with columns:
      ['h3_index', 'timestamp', 'temp_c', 'humidity', 'wind_speed_mps',
       'pressure_hpa', 'is_raining', 'weather_main', 'source']
    timestamps: pandas.DatetimeIndex (timezone-aware recommended)
    """
    n_cells = len(h3_indices)
    # time multipliers local to the function (self-contained)
    time_profile = np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
    temp_multiplier = (time_profile * 0.5) + 0.5  # 0..1
    all_rows = []

    for t_idx, ts in enumerate(tqdm(timestamps, desc="Generating weather")):
        spatial_noise = np.random.normal(0, 0.5, n_cells) * (np.arange(n_cells) / max(1, n_cells - 1))
        temps = (base_temp + (temp_multiplier[t_idx] * 5) + spatial_noise + np.random.normal(0, 0.5, n_cells)).round(2)
        humidity = np.clip(np.random.normal(70, 5, n_cells), 55, 85).round(1)
        wind = np.clip(np.random.normal(base_wind, 1, n_cells), 0.5, 8.0).round(2)
        pressure = np.random.randint(990, 1020, n_cells)
        is_raining = (np.random.rand(n_cells) < 0.05).astype(int)
        weather_main = np.random.choice(['Clear', 'Clouds', 'Rain'], size=n_cells, p=[0.65, 0.28, 0.07])

        for i, h3i in enumerate(h3_indices):
            all_rows.append({
                "h3_index": str(h3i),
                "timestamp": pd.Timestamp(ts).isoformat(),
                "temp_c": float(temps[i]),
                "humidity": float(humidity[i]),
                "wind_speed_mps": float(wind[i]),
                "pressure_hpa": int(pressure[i]),
                "is_raining": int(is_raining[i]),
                "weather_main": weather_main[i],
                "source": "mock_weather_seeded"
            })

    df = pd.DataFrame.from_records(all_rows)
    # cast dtypes for efficiency
    df["weather_main"] = df["weather_main"].astype("category")
    return df

def generate_mock_aqi(timestamps, h3_indices, base_aqi=70):
    """
    Generates a DataFrame with columns:
      ['h3_index', 'timestamp', 'aqi', 'source']
    """
    n_cells = len(h3_indices)
    time_profile = np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
    aqi_multiplier = (time_profile * 0.4) + 0.6  # 0.2..1.0 approx
    all_rows = []

    for t_idx, ts in enumerate(tqdm(timestamps, desc="Generating AQI")):
        spatial_factor = ((np.arange(n_cells) / max(1, n_cells - 1)) * 0.4) + 0.8
        base_series = base_aqi + (aqi_multiplier[t_idx] * 50)
        aqi_vals = np.clip(base_series * spatial_factor + np.random.normal(0, 10, n_cells), 0, 500).astype(int)

        for i, h3i in enumerate(h3_indices):
            all_rows.append({
                "h3_index": str(h3i),
                "timestamp": pd.Timestamp(ts).isoformat(),
                "aqi": int(aqi_vals[i]),
                "source": "mock_aqi_seeded"
            })

    df = pd.DataFrame.from_records(all_rows)
    return df

# ---------------- API seed helpers ----------------
def fetch_weather_seed(api_key, lat, lon):
    if not api_key:
        log.warning("OWM API key not set. Will use mock seeds.")
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        temp = float(j["main"]["temp"])
        wind = float(j["wind"].get("speed", 0.0))
        log.info("Fetched current weather: temp=%s, wind=%s", temp, wind)
        return temp, wind
    except Exception as e:
        log.error("Failed to fetch OWM seed: %s", e)
        return None

def fetch_aqi_seed(api_token, city_name):
    if not api_token:
        log.warning("WAQI token not set. Will use mock seeds.")
        return None
    try:
        city = city_name.split(",")[0].lower()
        url = f"https://api.waqi.info/feed/{city}/?token={api_token}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        jj = r.json()
        if jj.get("status") == "ok":
            aqi_val = int(jj["data"].get("aqi", 70))
            log.info("Fetched current AQI: %s", aqi_val)
            return aqi_val
        else:
            log.warning("WAQI returned status: %s", jj.get("status"))
            return None
    except Exception as e:
        log.error("WAQI fetch failed: %s", e)
        return None

# ---------------- Main execution ----------------
def main():
    # timestamps (timezone-aware)
    start_dt = pd.Timestamp.now(tz=TIMEZONE).replace(minute=0, second=0, microsecond=0) - pd.Timedelta(hours=24)
    timestamps = pd.date_range(start=start_dt, periods=N_TIMESTEPS, freq=FREQ, tz=TIMEZONE)
    log.info("Timestamps generated: %s -> %s (%d steps)", timestamps[0], timestamps[-1], len(timestamps))

    # generate H3 indices
    h3_indices = get_h3_cells_around_point(LAT, LON, res=H3_RESOLUTION, k_ring=H3_K_RING)
    n_h3 = len(h3_indices)
    log.info("Generated %d H3 cells at res=%d", n_h3, H3_RESOLUTION)

    # fetch seeds if available
    seed_weather = fetch_weather_seed(API_KEY_OWM, LAT, LON)
    seed_aqi = fetch_aqi_seed(API_TOKEN_AQI, CITY_NAME)

    # generate dataframes
    if seed_weather:
        weather_df = generate_mock_weather(timestamps, h3_indices, base_temp=seed_weather[0], base_wind=seed_weather[1])
    else:
        weather_df = generate_mock_weather(timestamps, h3_indices, base_temp=28.0, base_wind=2.5)

    if seed_aqi:
        aqi_df = generate_mock_aqi(timestamps, h3_indices, base_aqi=seed_aqi)
    else:
        aqi_df = generate_mock_aqi(timestamps, h3_indices, base_aqi=70)

    # Normalize timestamp dtype and merge
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"]).dt.tz_convert(TIMEZONE)
    aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"]).dt.tz_convert(TIMEZONE)

    log.info("Merging weather and AQI dataframes (inner join)...")
    combined = pd.merge(aqi_df, weather_df, on=["h3_index", "timestamp"], how="inner", suffixes=("_aqi", "_weather"))

    # unify source & metadata flags
    combined["source"] = combined["source_weather"].fillna(combined["source_aqi"]).fillna("mock")
    combined["api_seeded"] = combined["source"].str.contains("seeded", case=False, na=False)

    # tidy columns (drop original source_*)
    drop_cols = [c for c in combined.columns if c.endswith("_aqi") or c.endswith("_weather")]
    # keep 'aqi' and the weather fields, drop redundant source columns only
    combined = combined.drop(columns=[c for c in drop_cols if c not in ("aqi", "source")], errors="ignore")

    # add city / region fields
    combined["city"] = CITY_NAME
    combined["region_code"] = os.environ.get("REGION_CODE", "BBMP")

    # set index and sort for convenience
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.set_index(["h3_index", "timestamp"]).sort_index()

    # small validation checks
    if combined.empty:
        log.error("Combined dataframe is empty after merge. Exiting.")
        raise SystemExit(1)

    # Save to Parquet for efficient IO
    out_path = os.path.join(OUTPUT_DIR, PARQUET_FILENAME)
    log.info("Saving combined temporal data to Parquet: %s", out_path)
    combined.to_parquet(out_path, compression="snappy")

    # metadata
    metadata = {
        "city": CITY_NAME,
        "lat": LAT,
        "lon": LON,
        "generated_on": datetime.now().astimezone().isoformat(),
        "n_timesteps": len(timestamps),
        "n_h3_cells": n_h3,
        "time_granularity": FREQ,
        "h3_resolution": H3_RESOLUTION,
        "api_seeded_weather": bool(seed_weather),
        "api_seeded_aqi": bool(seed_aqi),
        "parquet": PARQUET_FILENAME,
        "notes": "High-granularity temporal mock data (H3 x timestamp). Use read_parquet to load."
    }
    meta_path = os.path.join(OUTPUT_DIR, METADATA_FILENAME)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Temporal generation complete. Parquet: %s  Metadata: %s", out_path, meta_path)
    print(f"✅ Temporal data generated: {out_path}")
    print(f"✅ Metadata: {meta_path}")
    print(f"Rows: {combined.shape[0]}  Columns: {len(combined.columns)}")

if __name__ == "__main__":
    main()
