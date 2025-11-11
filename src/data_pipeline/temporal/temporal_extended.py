#!/usr/bin/env python3
"""
extend_temporal_data.py

Generate extended temporal spatio-temporal data (H3 x timestamp) for N days.
Outputs daily parquet files + combined parquet + metadata.

Usage:
    python src/data_pipeline/extend_temporal_data.py --days 30 --out data/raw/temporal --h3-res 8 --k-ring 6 --split-by-day

Environment:
    OWM_API_KEY   (optional) - OpenWeatherMap current weather API key (seed)
    WAQI_TOKEN    (optional) - WAQI token (seed)

Author: Assistant (for Urban-Planning project)
"""

'''python src\data_pipeline\temporal\temporal_extended.py --days 30'''


import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import requests

# Try to import h3 robustly
try:
    import h3
except Exception:
    h3 = None

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
log = logging.getLogger("extend_temporal_data")

# ----------------- Helpers: H3 compatibility -----------------
def get_h3_api_functions():
    """Return functions (latlng_to_cell, grid_disk, cell_to_boundary, cell_to_latlng, polygon_to_cells) as available."""
    if h3 is None:
        return {}
    api = {}
    # latlng_to_cell / cell_to_latlng
    if hasattr(h3, "latlng_to_cell"):
        api['latlng_to_cell'] = h3.latlng_to_cell
    elif hasattr(h3, "geo_to_h3"):
        api['latlng_to_cell'] = h3.geo_to_h3
    # grid_disk / grid_ring
    if hasattr(h3, "grid_disk"):
        api['grid_disk'] = h3.grid_disk
    elif hasattr(h3, "k_ring"):
        api['grid_disk'] = h3.k_ring
    # boundary / center
    if hasattr(h3, "cell_to_boundary"):
        api['cell_to_boundary'] = h3.cell_to_boundary
    if hasattr(h3, "cell_to_latlng"):
        api['cell_to_latlng'] = h3.cell_to_latlng
    # polyfill / polygon_to_cells
    if hasattr(h3, "polygon_to_cells"):
        api['polygon_to_cells'] = h3.polygon_to_cells
    elif hasattr(h3, "polyfill"):
        api['polygon_to_cells'] = h3.polyfill
    return api

def generate_h3_cells(lat, lon, res=8, k_ring=6):
    """Return a list of H3 indices around lat/lon using available h3 functions. Fallback to dummy list."""
    api = get_h3_api_functions()
    if not api or 'latlng_to_cell' not in api:
        log.warning("h3 not available or missing latlng_to_cell: returning dummy indices.")
        return [f"dummy_{i}" for i in range(max(100, k_ring*10))]
    center = api['latlng_to_cell'](lat, lon, res)
    if 'grid_disk' in api:
        cells = list(api['grid_disk'](center, k_ring))
    else:
        # k_ring fallback
        cells = list(h3.k_ring(center, k_ring))
    return [str(c) for c in cells]

# ----------------- Generators -----------------
def generate_time_index(days, freq="30min", tz="Asia/Kolkata"):
    """Return a timezone-aware pandas DatetimeIndex for last `days` days with frequency `freq`."""
    end = pd.Timestamp.now(tz=tz).replace(minute=0, second=0, microsecond=0)
    start = end - pd.Timedelta(days=days)
    idx = pd.date_range(start=start, end=end - pd.Timedelta(minutes=30), freq=freq, tz=tz)
    return idx

def time_multipliers(n_steps):
    """Return diurnal multipliers arrays for temperature and aqi (length n_steps)."""
    tp = np.sin(np.linspace(0, 2 * np.pi, n_steps))
    temp_mult = (tp * 0.5) + 0.5
    aqi_mult = (tp * 0.4) + 0.6
    return temp_mult, aqi_mult

def fetch_current_weather_seed(api_key, lat, lon):
    """Try to fetch current weather; return (temp_c, wind_mps) or None on failure."""
    if not api_key:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=8); r.raise_for_status()
        j = r.json()
        return float(j['main']['temp']), float(j['wind'].get('speed', 0.0))
    except Exception as e:
        log.warning("Weather seed fetch failed: %s", e)
        return None

def fetch_current_aqi_seed(token, city):
    """Try WAQI seed; return aqi int or None."""
    if not token:
        return None
    try:
        cityname = city.split(",")[0].lower()
        url = f"https://api.waqi.info/feed/{cityname}/?token={token}"
        r = requests.get(url, timeout=8); r.raise_for_status()
        j = r.json()
        if j.get("status") == "ok":
            return int(j['data'].get('aqi', 70))
        else:
            return None
    except Exception as e:
        log.warning("AQI seed fetch failed: %s", e)
        return None

def generate_mock_weather_for_step(base_temp_series, base_wind, n_cells, t_idx):
    """Produce arrays for a single timestep (temp_c, humidity, wind, pressure, is_raining, weather_main)."""
    spatial_noise = np.random.normal(0, 0.5, n_cells) * (np.arange(n_cells) / max(1, n_cells - 1))
    temp_c = (base_temp_series[t_idx] + spatial_noise + np.random.normal(0, 0.5, n_cells)).round(2)
    humidity = np.clip(np.random.normal(70, 5, n_cells), 55, 85).round(1)
    wind = np.clip(np.random.normal(base_wind, 1, n_cells), 0.5, 8.0).round(2)
    pressure = np.random.randint(990, 1020, n_cells)
    is_raining = (np.random.rand(n_cells) < 0.05).astype(int)
    weather_main = np.random.choice(['Clear', 'Clouds', 'Rain'], size=n_cells, p=[0.65, 0.28, 0.07])
    return temp_c, humidity, wind, pressure, is_raining, weather_main

def generate_mock_aqi_for_step(base_aqi_series, n_cells, t_idx):
    """Produce array for a single timestep AQI per cell."""
    spatial_factor = ((np.arange(n_cells) / max(1, n_cells - 1)) * 0.4) + 0.8
    aqi_level = base_aqi_series[t_idx] * spatial_factor
    aqi_vals = np.clip(aqi_level + np.random.normal(0, 10, n_cells), 0, 500).astype(int)
    return aqi_vals

def simulate_utilities(n_cells, t_idx, n_steps, base_elec=100, base_water=500):
    """Optional: simple diurnal utility profiles per H3 cell (kWh, liters)."""
    # sinusoidal daily pattern
    tp = np.sin(2 * np.pi * (t_idx / n_steps))
    # spatial variation
    spatial_factor = 0.6 + (np.arange(n_cells) / max(1, n_cells - 1)) * 0.8
    elec = np.clip(base_elec * (1 + 0.4 * tp) * spatial_factor + np.random.normal(0,5,n_cells), 0, None).round(2)
    water = np.clip(base_water * (1 + 0.3 * tp) * spatial_factor + np.random.normal(0,20,n_cells), 0, None).round(1)
    return elec, water

# ----------------- Main generation routine -----------------
def generate_extended_temporal(days=7,
                               lat=12.9716, lon=77.5946,
                               h3_res=8, k_ring=6,
                               out_dir="data/raw/temporal",
                               split_by_day=True,
                               simulate_util=True,
                               seed_with_api=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tz = "Asia/Kolkata"

    # H3 cells
    log.info("Generating H3 indices (res=%d, k_ring=%d)...", h3_res, k_ring)
    h3_indices = generate_h3_cells(lat, lon, res=h3_res, k_ring=k_ring)
    n_cells = len(h3_indices)
    log.info("H3 cells count: %d", n_cells)

    # time index
    timestamps = generate_time_index(days, freq="30min", tz=tz)
    n_steps = len(timestamps)
    log.info("Timestamps: %s -> %s (%d steps)", timestamps[0], timestamps[-1], n_steps)

    # multipliers/time series base
    temp_mult, aqi_mult = time_multipliers(n_steps)

    # seeds via API?
    owm_key = os.environ.get("OWM_API_KEY")
    waqi_token = os.environ.get("WAQI_TOKEN")
    if seed_with_api and (owm_key or waqi_token):
        weather_seed = fetch_current_weather_seed(owm_key, lat, lon) if owm_key else None
        aqi_seed = fetch_current_aqi_seed(waqi_token, city=f"{lat},{lon}") if waqi_token else None
    else:
        weather_seed = None
        aqi_seed = None

    base_temp_val = weather_seed[0] if weather_seed else 28.0
    base_wind_val = weather_seed[1] if weather_seed else 2.5
    base_aqi_val = aqi_seed if aqi_seed is not None else 70

    # prepare containers to save per-step rows then dump per-day (to limit memory)
    rows = []
    # For splitting per day, we'll accumulate per-day then write
    current_day = None
    daily_rows = []

    log.info("Starting generation (days=%d)...", days)
    for t_idx, ts in enumerate(tqdm(timestamps, desc="Generating temporal")):
        # base series values at step
        base_temp_series_val = base_temp_val + (temp_mult[t_idx] * 5)  # modulated value
        base_aqi_series_val = base_aqi_val + (aqi_mult[t_idx] * 50)

        # generate arrays for this step
        temps, hums, winds, pressures, rains, weather_mains = generate_mock_weather_for_step(
            base_temp_series=np.full(n_steps, base_temp_series_val), # pass vector but we only use index
            base_wind=base_wind_val,
            n_cells=n_cells,
            t_idx=0  # we used base_temp_series already; generate per-cell
        )
        # But simpler: use single-step generators using above functions directly
        # generate AQI
        aqi_vals = generate_mock_aqi_for_step(np.full(n_steps, base_aqi_series_val), n_cells, 0)

        # simulate utilities optionally
        if simulate_util:
            elec_vals, water_vals = simulate_utilities(n_cells, t_idx, n_steps, base_elec=100, base_water=500)
        else:
            elec_vals = np.zeros(n_cells)
            water_vals = np.zeros(n_cells)

        # build rows for all cells at this timestamp
        ts_iso = pd.Timestamp(ts).isoformat()
        for i, h3i in enumerate(h3_indices):
            row = {
                "h3_index": str(h3i),
                "timestamp": ts_iso,
                "aqi": int(aqi_vals[i]),
                "temp_c": float(temps[i]),
                "humidity": float(hums[i]),
                "wind_speed_mps": float(winds[i]),
                "pressure_hpa": int(pressures[i]),
                "is_raining": int(rains[i]),
                "weather_main": str(weather_mains[i]),
                "electricity_kwh": float(elec_vals[i]),
                "water_liters": float(water_vals[i]),
                "source": "api_seeded" if (weather_seed or aqi_seed) else "mock"
            }
            daily_rows.append(row)

        # if splitting by day, flush when day boundary passes
        day_str = pd.Timestamp(ts).date().isoformat()
        if current_day is None:
            current_day = day_str

        if split_by_day:
            # At the final step of a day or end of series, flush
            next_ts = timestamps[t_idx + 1] if t_idx + 1 < n_steps else None
            next_day = pd.Timestamp(next_ts).date().isoformat() if next_ts is not None else None
            if next_day != current_day:
                # write current_day parquet
                df_day = pd.DataFrame(daily_rows)
                df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])
                out_path = out_dir / f"temporal_{current_day}.parquet"
                df_day.set_index(['h3_index','timestamp']).to_parquet(out_path, compression='snappy')
                log.info("WROTE day parquet: %s (rows=%d)", out_path, df_day.shape[0])
                # reset
                daily_rows = []
                current_day = next_day

    # After loop, optionally write combined parquet
    log.info("Writing combined parquet (concatenating daily files)...")
    # find daily files and concat
    daily_files = sorted(out_dir.glob("temporal_*.parquet"))
    if not daily_files:
        # no splits (split_by_day False) -> write a combined parquet from daily_rows
        df_all = pd.DataFrame(daily_rows)
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        combined_path = out_dir / "temporal_combined.parquet"
        df_all.set_index(['h3_index','timestamp']).to_parquet(combined_path, compression='snappy')
        log.info("WROTE combined parquet: %s", combined_path)
    else:
        # concat daily
        dfs = [pd.read_parquet(p).reset_index() for p in daily_files]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        combined_path = out_dir / f"temporal_{days}d_combined.parquet"
        df_all.set_index(['h3_index','timestamp']).to_parquet(combined_path, compression='snappy')
        log.info("WROTE combined parquet: %s (rows=%d)", combined_path, df_all.shape[0])

    # metadata
    metadata = {
        "generated_on": datetime.now().astimezone().isoformat(),
        "days": days,
        "n_steps": n_steps,
        "freq": "30min",
        "h3_resolution": h3_res,
        "k_ring": k_ring,
        "n_h3_cells": n_cells,
        "combined_path": str(combined_path),
        "daily_files": [str(p) for p in sorted(out_dir.glob("temporal_*.parquet"))],
        "seeded": bool(weather_seed or aqi_seed),
        "simulate_utilities": bool(simulate_util)
    }
    meta_path = out_dir / "temporal_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("WROTE metadata: %s", meta_path)

    # quick validation
    log.info("Running quick validation on combined:")
    print(df_all.head(3))
    assert not df_all.empty, "Generated dataframe is empty!"
    dup = df_all.reset_index().duplicated(subset=['h3_index','timestamp']).sum()
    log.info("Duplicate (h3,timestamp) pairs: %d", int(dup))
    return combined_path, meta_path

# ----------------- CLI -----------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30, help="Number of days to generate")
    p.add_argument("--lat", type=float, default=12.9716, help="Center latitude")
    p.add_argument("--lon", type=float, default=77.5946, help="Center longitude")
    p.add_argument("--h3-res", type=int, default=8, help="H3 resolution")
    p.add_argument("--k-ring", type=int, default=6, help="k_ring around center")
    p.add_argument("--out", type=str, default="data/raw/temporal", help="Output directory (will be created)")
    p.add_argument("--no-sim-util", dest="sim_util", action="store_false", help="Disable utility simulation")
    p.add_argument("--no-split", dest="split_by_day", action="store_false", help="Disable per-day splitting")
    p.add_argument("--seed-api", dest="seed_with_api", action="store_true", help="Attempt to seed using APIs (set env vars)")
    p.set_defaults(sim_util=True, split_by_day=True, seed_with_api=False)
    args = p.parse_args()
    out = generate_extended_temporal(days=args.days,
                                     lat=args.lat, lon=args.lon,
                                     h3_res=args.h3_res, k_ring=args.k_ring,
                                     out_dir=args.out,
                                     split_by_day=args.split_by_day,
                                     simulate_util=args.sim_util,
                                     seed_with_api=args.seed_with_api)
    log.info("Done. Combined parquet: %s  metadata: %s", out[0], out[1])

if __name__ == "__main__":
    cli()
