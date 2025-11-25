#!/usr/bin/env python3
"""
synthesize_missing_data.py

Create synthetic but realistic missing datasets for the Bengaluru project:
- hourly traffic speeds per H3 cell
- hourly OD flows (aggregated)
- hourly electricity load per ward
- hourly water usage per ward
- hourly sewage load per ward
- climate projection deltas per H3 (2030, 2050 for RCP4.5 & RCP8.5)
- nightlight / economic proxy time series per ward

Usage example:
 python synthesize_missing_data.py \
    --wards /mnt/data/wards_master_enriched.geojson \
    --h3_grid /mnt/data/h3_to_wards_mapping_res8.csv \
    --pop_h3 /mnt/data/population_h3_res8.geojson \
    --roads /mnt/data/roads.parquet \
    --jobs_h3 /mnt/data/economy_h3_hotspot.geojson \
    --electricity_assets /mnt/data/electricity_assets.csv \
    --od_csv /mnt/data/od_flows.csv \
    --outdir ./outputs/synthesized \
    --days 30 \
    --seed 42

Notes:
 - The script is conservative about memory: default days=30 (30 days hourly). Increase if you want a full year.
 - If a required anchor file is missing, the script will still create synthetic outputs using population distributions.
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import datetime
import json
import random

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("synthesizer")

# -----------------------
# Helpers
# -----------------------
def safe_read_geo(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("File not found: %s", path)
        return None
    try:
        gdf = gpd.read_file(str(p))
        gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        log.warning("Geo read failed %s: %s", path, e)
        return None

def safe_read_csv(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("CSV not found: %s", path)
        return None
    try:
        df = pd.read_csv(str(p))
        return df
    except Exception as e:
        log.warning("CSV read failed %s: %s", path, e)
        return None

def ensure_outdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

# -----------------------
# Core synthesis functions
# -----------------------
def make_time_index(start, days, freq="H"):
    start = pd.to_datetime(start)
    periods = int(days * (24 if freq == "H" else 1))
    return pd.date_range(start=start, periods=periods, freq=freq)

def diurnal_profile(hour):
    # A generic diurnal demand factor (0..1)
    # morning peak ~8-10, evening peak ~18-20
    return (
        0.3
        + 0.6 * np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)  # morning
        + 0.6 * np.exp(-0.5 * ((hour - 19) / 2.5) ** 2)  # evening
    ) / 1.6

def weekday_multiplier(dt):
    # weekday  Mon-Fri higher, weekend lower
    wd = dt.weekday()
    if wd < 5:
        return 1.0
    else:
        return 0.75

def seasonal_multiplier(day_of_year):
    # mild seasonal effect for year; normalized around 1
    return 1 + 0.05 * np.sin(2 * np.pi * (day_of_year / 365.0))

def synthesize_electricity(wards_gdf, population_wards, jobs_h3, assets_df, time_index, seed=42):
    """
    Create hourly electricity load per ward.
    Base load proportional to population; industrial extra proportional to jobs/industry presence.
    Returns DataFrame: timestamp, ward_id, load_kw
    """
    rng = np.random.RandomState(seed)
    ward_ids = []
    populations = {}
    # If population_wards available: use its 'population' or 'population_2011' column
    if population_wards is not None:
        for idx, r in population_wards.iterrows():
            wid = r.get("ward_id") or r.get("ward") or r.get("ward_name") or idx
            populations[wid] = float(r.get("population_2011") or r.get("population") or 10000)
    elif wards_gdf is not None:
        for idx, r in wards_gdf.iterrows():
            wid = r.get("ward_id") or r.get("ward_name") or idx
            populations[wid] = float(r.get("population_2011") or 10000)
    else:
        # fallback single ward
        populations = {"ward_0": 100000}

    # jobs factor per ward via jobs_h3 aggregated by ward if possible
    job_factor = {w: 1.0 for w in populations.keys()}
    if jobs_h3 is not None:
        # jobs_h3 expected to have h3_index and it_job_density or job_count and ward mapping maybe
        # We'll try to map by ward if jobs_h3 includes ward_id
        if "ward_id" in jobs_h3.columns:
            jagg = jobs_h3.groupby("ward_id")["it_job_density"].sum() if "it_job_density" in jobs_h3.columns else jobs_h3.groupby("ward_id").size()
            for w, v in jagg.items():
                if w in job_factor:
                    job_factor[w] = float(v) / (jagg.mean() + 1e-9)  # normalize
    # assets_df could suggest industrial wards
    industrial_boost = {}
    if assets_df is not None and "ward_id" in assets_df.columns:
        idf = assets_df.groupby("ward_id").size().to_dict()
        for w in job_factor.keys():
            industrial_boost[w] = 1.0 + 0.2 * (idf.get(w, 0) / (max(idf.values()) if idf else 1))
    else:
        industrial_boost = {w: 1.0 for w in job_factor.keys()}

    rows = []
    # base per-capita hourly usage in kW (average) -> adapt
    per_capita_kw = 0.35  # average kW per person as baseline (varies)
    for ts in time_index:
        hour = ts.hour
        dow = weekday_multiplier(ts)
        season = seasonal_multiplier(ts.timetuple().tm_yday)
        diurnal = diurnal_profile(hour)
        for w, pop in populations.items():
            base = pop * per_capita_kw  # kW baseline
            load = base * diurnal * dow * season * job_factor.get(w, 1.0) * industrial_boost.get(w, 1.0)
            # add noise and small random spikes
            load = load * (1 + rng.normal(0, 0.05))
            # occasional event spikes (1% chance)
            if rng.rand() < 0.01:
                load *= (1 + rng.uniform(0.1, 0.4))
            rows.append({"timestamp": ts, "ward_id": w, "electricity_kw": max(0.1, float(load))})
    df = pd.DataFrame(rows)
    return df

def synthesize_water(wards_gdf, population_wards, time_index, seed=42):
    """
    Hourly water usage per ward (liters per hour)
    Base proportional to population; diurnal + weekday + season.
    """
    rng = np.random.RandomState(seed + 1)
    populations = {}
    if population_wards is not None:
        for idx, r in population_wards.iterrows():
            wid = r.get("ward_id") or r.get("ward_name") or idx
            populations[wid] = float(r.get("population_2011") or r.get("population") or 10000)
    elif wards_gdf is not None:
        for idx, r in wards_gdf.iterrows():
            wid = r.get("ward_id") or r.get("ward_name") or idx
            populations[wid] = float(r.get("population_2011") or 10000)
    else:
        populations = {"ward_0": 100000}

    per_capita_lph = 100.0 / 24.0  # liters per person per hour baseline ~100 L/day
    rows = []
    for ts in time_index:
        hour = ts.hour
        dow = weekday_multiplier(ts)
        season = 1.0 + 0.15 * np.cos(2 * np.pi * (ts.timetuple().tm_yday / 365.0))  # slightly seasonal
        diurnal = 0.5 + 0.8 * np.exp(-0.5 * ((hour - 7) / 2.0) ** 2) + 0.6 * np.exp(-0.5 * ((hour - 19) / 2.5) ** 2)
        for w, pop in populations.items():
            base = pop * per_capita_lph
            usage = base * diurnal * dow * season
            usage = usage * (1 + rng.normal(0, 0.08))
            if rng.rand() < 0.005:
                usage *= (1 + rng.uniform(0.2, 0.6))
            rows.append({"timestamp": ts, "ward_id": w, "water_lph": max(0.0, float(usage))})
    return pd.DataFrame(rows)

def synthesize_sewage(water_df):
    """
    Sewage approximated as a fraction of water usage per ward (e.g., 80% returns).
    """
    df = water_df.copy()
    df["sewage_lph"] = df["water_lph"] * 0.85  # assume 85% returns as sewage
    return df[["timestamp", "ward_id", "sewage_lph"]]

def synthesize_traffic(h3_grid_gdf, population_h3_gdf, roads_gdf, time_index, seed=42):
    """
    Create hourly avg_speed and max_speed per h3 cell.
    Base speed depends on road class & betweenness aggregated into H3.
    If no anchors available, use default distribution.
    Returns DataFrame: timestamp, h3_id, avg_speed_kmph, max_speed_kmph, congestion_index, vehicle_count
    """
    rng = np.random.RandomState(seed + 2)

    # create h3 list and base speed
    if h3_grid_gdf is not None and "h3_index" in h3_grid_gdf.columns:
        h3_list = h3_grid_gdf["h3_index"].tolist()
    elif population_h3_gdf is not None and "h3_index" in population_h3_gdf.columns:
        h3_list = population_h3_gdf["h3_index"].tolist()
    else:
        # fallback single cell
        h3_list = ["h3_0"]

    # estimate a base free-flow speed per h3 using roads betweenness mapped to h3 if possible
    base_speed_map = {}
    if roads_gdf is not None and "betweenness" in roads_gdf.columns and "h3_id" in roads_gdf.columns:
        # average betweenness per h3
        r = roads_gdf.copy()
        r["h3_id"] = r["h3_id"].fillna("none")
        agg = r.groupby("h3_id")["betweenness"].mean().to_dict()
        for h in h3_list:
            b = agg.get(h, 0.1)
            # base speed decreases with betweenness (busy corridors slower). But base freeflow higher for highways
            base_speed_map[h] = max(20, 60 - 30 * b)
    else:
        # fallback: assign base speeds with small variation
        for i, h in enumerate(h3_list):
            base_speed_map[h] = rng.uniform(30, 50)

    rows = []
    # vehicle_count baseline per h3 proportional to population or built density
    pop_map = {}
    if population_h3_gdf is not None and "h3_index" in population_h3_gdf.columns:
        pop_map = dict(zip(population_h3_gdf["h3_index"], population_h3_gdf.get("population_h3", pd.Series(1, index=population_h3_gdf.index))))
    else:
        pop_map = {h: 1000 for h in h3_list}

    for ts in time_index:
        hour = ts.hour
        dow = weekday_multiplier(ts)
        season = seasonal_multiplier(ts.timetuple().tm_yday)
        for h in h3_list:
            base = base_speed_map.get(h, 35.0)
            # congestion factor: inverse of diurnal demand
            demand_factor = (1.0 + 2.5 * (1.0 - diurnal_profile(hour))) * dow * season
            # vehicle count proportional to pop_map
            veh_base = max(5, pop_map.get(h, 1000) * 0.002)
            vehicle_count = int(rng.poisson(veh_base * (0.6 + diurnal_profile(hour))))
            # avg_speed decreases with demand
            avg_speed = base / (1.0 + 0.8 * (demand_factor - 1.0)) + rng.normal(0, 1.5)
            avg_speed = max(3.0, float(avg_speed))
            max_speed = min(90.0, float(avg_speed * (1.0 + rng.uniform(0.05, 0.25))))
            congestion_index = float(np.clip(1 - (avg_speed / 40.0), 0.0, 1.0))
            travel_time_index = float(1.0 / max(0.01, (1 - congestion_index)))
            rows.append({
                "timestamp": ts,
                "h3_id": h,
                "avg_speed_kmph": avg_speed,
                "max_speed_kmph": max_speed,
                "congestion_index": congestion_index,
                "travel_time_index": travel_time_index,
                "vehicle_count": vehicle_count
            })
    df = pd.DataFrame(rows)
    return df

def synthesize_od_flows(h3_grid_gdf, population_h3_gdf, days, seed=42, hourly=True, target_n_records=50000):
    """
    Create aggregated OD flows between H3 cells.
    If population_h3_gdf exists, sample origins/destinations proportional to population.
    Returns aggregated DataFrame with origin_h3, destination_h3, time_of_day, od_flow
    """
    rng = np.random.RandomState(seed + 3)
    if population_h3_gdf is not None and "h3_index" in population_h3_gdf.columns and "population_h3" in population_h3_gdf.columns:
        pop = population_h3_gdf[["h3_index", "population_h3"]].copy()
        pop["population_h3"] = pop["population_h3"].astype(float).clip(min=1)
        h3s = pop["h3_index"].tolist()
        weights = (pop["population_h3"] / pop["population_h3"].sum()).values
    elif h3_grid_gdf is not None and "h3_index" in h3_grid_gdf.columns:
        h3s = h3_grid_gdf["h3_index"].tolist()
        weights = np.ones(len(h3s)) / len(h3s)
    else:
        h3s = ["h3_0", "h3_1", "h3_2"]
        weights = np.ones(len(h3s)) / len(h3s)

    # time bins
    time_bins = list(range(24)) if hourly else [0]
    records = []
    total_samples = target_n_records
    for i in range(total_samples):
        o = rng.choice(h3s, p=weights)
        d = rng.choice(h3s, p=weights)
        tod = rng.choice(time_bins, p=np.ones(len(time_bins))/len(time_bins))
        records.append((o, d, int(tod)))
    df = pd.DataFrame(records, columns=["origin_h3", "destination_h3", "time_of_day"])
    agg = df.groupby(["origin_h3", "destination_h3", "time_of_day"]).size().reset_index(name="od_flow")
    return agg

def synthesize_climate_projections(h3_grid_gdf, seed=42):
    """
    Produce simple climate projection deltas per H3 for 2030 and 2050 under RCP4.5 and RCP8.5.
    We'll output temperature_delta_C and precip_percent_change.
    """
    rng = np.random.RandomState(seed + 4)
    rows = []
    if h3_grid_gdf is not None and "h3_index" in h3_grid_gdf.columns:
        for idx, r in h3_grid_gdf.iterrows():
            h = r["h3_index"]
            # base randomness by location
            local_noise = rng.normal(0, 0.05)
            # RCP4.5
            temp_2030_45 = 0.6 + local_noise + rng.uniform(-0.2, 0.3)
            temp_2050_45 = 1.0 + local_noise + rng.uniform(-0.2, 0.6)
            precip_2030_45 = rng.normal(0, 5)  # percent change
            precip_2050_45 = rng.normal(0, 7)
            # RCP8.5 more severe
            temp_2030_85 = 0.9 + local_noise + rng.uniform(0.0, 0.6)
            temp_2050_85 = 2.0 + local_noise + rng.uniform(0.5, 1.5)
            precip_2030_85 = rng.normal(1, 6)
            precip_2050_85 = rng.normal(2, 10)
            rows.append({
                "h3_id": h,
                "temp_delta_2030_rcp45_C": float(round(temp_2030_45, 3)),
                "temp_delta_2050_rcp45_C": float(round(temp_2050_45, 3)),
                "precip_pct_2030_rcp45": float(round(precip_2030_45, 3)),
                "precip_pct_2050_rcp45": float(round(precip_2050_45, 3)),
                "temp_delta_2030_rcp85_C": float(round(temp_2030_85, 3)),
                "temp_delta_2050_rcp85_C": float(round(temp_2050_85, 3)),
                "precip_pct_2030_rcp85": float(round(precip_2030_85, 3)),
                "precip_pct_2050_rcp85": float(round(precip_2050_85, 3)),
            })
    else:
        # fallback some generic rows
        for i in range(100):
            h = f"h3_{i}"
            rows.append({
                "h3_id": h,
                "temp_delta_2030_rcp45_C": float(round(0.6 + rng.normal(0, 0.2), 3)),
                "temp_delta_2050_rcp45_C": float(round(1.1 + rng.normal(0, 0.4), 3)),
                "precip_pct_2030_rcp45": float(round(rng.normal(0, 5), 2)),
                "precip_pct_2050_rcp45": float(round(rng.normal(0, 7), 2)),
                "temp_delta_2030_rcp85_C": float(round(1.0 + rng.normal(0, 0.3), 3)),
                "temp_delta_2050_rcp85_C": float(round(2.2 + rng.normal(0, 0.6), 3)),
                "precip_pct_2030_rcp85": float(round(rng.normal(1, 6), 2)),
                "precip_pct_2050_rcp85": float(round(rng.normal(2, 10), 2)),
            })
    return pd.DataFrame(rows)

def synthesize_nightlights(population_wards_df, jobs_wards_df, days, seed=42):
    """
    Produce daily nightlight index per ward as economic proxy.
    Nightlight grows slightly over time; correlate with job_density.
    """
    rng = np.random.RandomState(seed + 5)
    if population_wards_df is None:
        wards = ["ward_0"]
        pop = {wards[0]: 100000}
    else:
        wards = []
        pop = {}
        for idx, r in population_wards_df.iterrows():
            w = r.get("ward_id") or r.get("ward_name") or idx
            wards.append(w)
            pop[w] = float(r.get("population_2011") or r.get("population") or 10000)

    job_factor = {w: 1.0 for w in wards}
    if jobs_wards_df is not None and "ward_id" in jobs_wards_df.columns:
        jf = jobs_wards_df.groupby("ward_id")["it_job_density"].sum() if "it_job_density" in jobs_wards_df.columns else jobs_wards_df.groupby("ward_id").size()
        for w, v in jf.items():
            if w in job_factor:
                job_factor[w] = float(v) / (jf.mean() + 1e-9)

    dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=days, freq="D")
    rows = []
    for d_idx, d in enumerate(dates):
        trend = 1.0 + 0.0005 * d_idx  # gentle upward trend
        for w in wards:
            base = pop[w] * 0.0001 * job_factor.get(w, 1.0)
            noise = rng.normal(0, base * 0.1)
            value = max(0.0, base * trend + noise)
            rows.append({"date": d.date().isoformat(), "ward_id": w, "nightlight_index": float(value)})
    return pd.DataFrame(rows)

# -----------------------
# Main
# -----------------------
def main(args):
    outdir = ensure_outdir(args.outdir)
    log.info("Output directory: %s", outdir)

    # Read anchor files if available
    wards_gdf = safe_read_geo(args.wards)
    h3_grid_gdf = None
    # The user provided h3 mapping maybe as CSV (h3_to_wards_mapping_res8.csv) or as geojson
    if args.h3_grid and str(args.h3_grid).lower().endswith(".csv"):
        try:
            h3_grid_df = safe_read_csv(args.h3_grid)
            # if centroid lon/lat columns exist, convert to GeoDataFrame
            if h3_grid_df is not None and "centroid_lon" in h3_grid_df.columns and "centroid_lat" in h3_grid_df.columns and "h3_id" in h3_grid_df.columns:
                h3_grid_gdf = gpd.GeoDataFrame(h3_grid_df, geometry=gpd.points_from_xy(h3_grid_df["centroid_lon"], h3_grid_df["centroid_lat"]), crs="EPSG:4326")
                # normalize column name
                if "h3_id" in h3_grid_gdf.columns:
                    h3_grid_gdf = h3_grid_gdf.rename(columns={"h3_id":"h3_index"})
        except Exception as e:
            log.warning("Failed reading h3 CSV: %s", e)
    else:
        h3_grid_gdf = safe_read_geo(args.h3_grid)

    population_h3_gdf = safe_read_geo(args.pop_h3) if args.pop_h3 else None
    population_wards_df = safe_read_csv(args.population_wards) if args.population_wards else None
    jobs_h3_gdf = safe_read_geo(args.jobs_h3) if args.jobs_h3 else None
    jobs_wards_df = safe_read_csv(args.jobs_wards) if args.jobs_wards else None
    roads_gdf = safe_read_geo(args.roads) if args.roads else None
    assets_df = safe_read_csv(args.electricity_assets) if args.electricity_assets else None
    od_existing = safe_read_csv(args.od_csv) if args.od_csv else None

    # Time index
    start = args.start_date or datetime.datetime.now().strftime("%Y-%m-%d")
    time_index = make_time_index(start, args.days, freq="H")
    log.info("Time index from %s, days=%d, points=%d", start, args.days, len(time_index))

    # Synthesize electricity
    log.info("Synthesizing electricity loads...")
    elec_df = synthesize_electricity(wards_gdf, population_wards_df, jobs_h3_gdf, assets_df, time_index, seed=args.seed)
    elec_out = outdir / "electricity_hourly_{}.parquet".format(args.days)
    elec_df.to_parquet(elec_out, index=False)
    log.info("Saved electricity load: %s", elec_out)

    # Synthesize water
    log.info("Synthesizing water usage...")
    water_df = synthesize_water(wards_gdf, population_wards_df, time_index, seed=args.seed)
    water_out = outdir / "water_hourly_{}.parquet".format(args.days)
    water_df.to_parquet(water_out, index=False)
    log.info("Saved water usage: %s", water_out)

    # Synthesize sewage
    log.info("Synthesizing sewage loads...")
    sewage_df = synthesize_sewage(water_df)
    sewage_out = outdir / "sewage_hourly_{}.parquet".format(args.days)
    sewage_df.to_parquet(sewage_out, index=False)
    log.info("Saved sewage loads: %s", sewage_out)

    # Synthesize traffic (H3)
    log.info("Synthesizing traffic speeds per H3...")
    traffic_df = synthesize_traffic(h3_grid_gdf, population_h3_gdf, roads_gdf, time_index, seed=args.seed)
    traffic_out = outdir / "traffic_h3_hourly_{}.parquet".format(args.days)
    traffic_df.to_parquet(traffic_out, index=False)
    log.info("Saved traffic speeds: %s", traffic_out)

    # Synthesize OD flows aggregated
    log.info("Synthesizing OD flows...")
    od_agg = synthesize_od_flows(h3_grid_gdf, population_h3_gdf, args.days, seed=args.seed, hourly=True, target_n_records=args.od_samples)
    od_out = outdir / "od_flows_synthesized.csv"
    od_agg.to_csv(od_out, index=False)
    log.info("Saved OD flows: %s", od_out)

    # Climate projections
    log.info("Synthesizing climate projection deltas per H3...")
    climate_df = synthesize_climate_projections(h3_grid_gdf, seed=args.seed)
    climate_out = outdir / "climate_projections_h3.csv"
    climate_df.to_csv(climate_out, index=False)
    log.info("Saved climate projections: %s", climate_out)

    # Nightlights / economic proxy
    log.info("Synthesizing nightlight economic proxy (daily)...")
    days = args.days
    night_df = synthesize_nightlights(population_wards_df, jobs_wards_df, days=args.days, seed=args.seed)
    night_out = outdir / "nightlight_wards_daily.csv"
    night_df.to_csv(night_out, index=False)
    log.info("Saved nightlight proxy: %s", night_out)

    # Produce a small metadata JSON summarizing produced files
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat(),
        "seed": args.seed,
        "days": args.days,
        "time_step": "hourly",
        "files": {
            "electricity": str(elec_out),
            "water": str(water_out),
            "sewage": str(sewage_out),
            "traffic": str(traffic_out),
            "od": str(od_out),
            "climate": str(climate_out),
            "nightlight": str(night_out)
        },
        "notes": "Synthetic datasets created for modelling and pipeline development. Replace with real time-series where available for final analysis."
    }
    with open(outdir / "synthesized_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Synthesis complete. Metadata saved.")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Synthesize missing datasets for Bengaluru project")
    p.add_argument("--wards", type=str, default="C:\AIurban-planning\data\processed\wards\wards_master_enriched.geojson", help="Wards geojson")
    p.add_argument("--h3_grid", type=str, default="C:\AIurban-planning\data\processed\h3_wards\h3_grid_res8_mapped.geojson", help="H3 grid CSV or geojson (with h3_index/centroid cols)")
    p.add_argument("--pop_h3", type=str, default="C:\AIurban-planning\data\processed\population_h3_res8.csv", help="Population per H3")
    p.add_argument("--population_wards", type=str, default="C:\AIurban-planning\data\processed\population_wards.csv", help="Ward population CSV")
    p.add_argument("--jobs_h3", type=str, default="C:\AIurban-planning\data\processed\economy_h3_hotspot.geojson", help="Jobs per H3")
    p.add_argument("--jobs_wards", type=str, default="C:\AIurban-planning\data\processed\economy_wards_growth.csv", help="Jobs per ward CSV")
    p.add_argument("--roads", type=str, default="C:\AIurban-planning\data\processed\roads_edges.csv", help="Roads enriched (optional)")
    p.add_argument("--electricity_assets", type=str, default="C:\AIurban-planning\data\processed\electricty\electricity_assets.geojson", help="Electricity assets CSV (optional)")
    p.add_argument("--od_csv", type=str, default="C:\AIurban-planning\data\processed\od_flows.csv", help="Existing OD flows file (optional)")
    p.add_argument("--outdir", type=str, default="./outputs/synthesized", help="Output directory")
    p.add_argument("--days", type=int, default=30, help="Number of days to synthesize (hourly resolution). Default 30")
    p.add_argument("--od-samples", dest="od_samples", type=int, default=50000, help="Number of OD sample draws to aggregate")
    p.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD (default today)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()
    main(args)
