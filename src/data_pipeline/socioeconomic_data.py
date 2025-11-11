#!/usr/bin/env python3
"""
socioeconomic_data.py

Build enriched socioeconomic dataset from uploaded Census Excel (HH-14 style) and
produce CSV / GeoJSON / Parquet outputs. Robust header detection + dtype sanitization
for parquet export (pyarrow friendly).
"""
from __future__ import annotations
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys

# -------------------------
# CONFIG
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
log = logging.getLogger("socioeconomic")

# Input Excel (uploaded earlier)
INPUT_PATH = Path("D:\\hlpca-29572-2011_h14_census.xlsx")  # adjust if your file path differs

# Output directory (project-standard)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "socioeconomic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bengaluru center for synthetic geometry
CENTER_LAT, CENTER_LON = 12.9716, 77.5946

# -------------------------
# Helpers
# -------------------------
def find_header_row(excel_path: Path, sheet_name: str, max_rows: int = 40) -> int:
    """Heuristic to locate header row in a messy Excel sheet."""
    dfp = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, nrows=max_rows)
    keywords = ["ward", "area", "district", "household", "population", "toilet", "electric", "water", "structure", "tv"]
    for i, row in dfp.iterrows():
        row_text = " ".join([str(x).lower() for x in row.fillna("")])
        if any(k in row_text for k in keywords):
            return i
    return 0

def safe_colname(c: str) -> str:
    return str(c).strip().replace("\n", " ").replace("  ", " ").replace(" ", "_").lower()

def pct_from_col(df: pd.DataFrame, colname: str | None, default_mean: float, N: int) -> pd.Series:
    """Return a percent series either derived from a numeric column or a synthetic distribution."""
    if colname and colname in df.columns:
        s = pd.to_numeric(df[colname], errors="coerce")
        # if values look like counts (max > 100), convert to percent by dividing by households if plausible
        if s.max(skipna=True) is not None and s.max(skipna=True) > 100 and "households" in df.columns:
            denom = df["households"].replace({0: 1}).astype(float)
            pct = (s.astype(float) / denom * 100).clip(0, 100)
            return pct.fillna(default_mean)
        else:
            return s.fillna(default_mean)
    else:
        return pd.Series(np.random.normal(default_mean, 5, N)).clip(0, 100).round(2)

# -------------------------
# MAIN
# -------------------------
def build_socioeconomic_dataset(input_path: Path, output_dir: Path) -> None:
    log.info("Reading Excel: %s", input_path)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        raise FileNotFoundError(input_path)

    xls = pd.ExcelFile(input_path)
    sheets = xls.sheet_names
    log.info("Sheets found: %s", sheets)

    # pick a sheet with probable header row (heuristic)
    chosen_sheet = None
    header_row = 0
    for sh in sheets:
        hr = find_header_row(input_path, sh)
        if hr is not None:
            chosen_sheet = sh
            header_row = hr
            break
    if chosen_sheet is None:
        chosen_sheet = sheets[0]

    log.info("Chosen sheet: %s (header detection)", chosen_sheet)
    df_raw = pd.read_excel(input_path, sheet_name=chosen_sheet, header=header_row)
    df_raw = df_raw.dropna(how="all").reset_index(drop=True)
    log.info("Rows read: %d", len(df_raw))

    # Normalize column names
    rename_map = {c: safe_colname(c) for c in df_raw.columns}
    df_raw.rename(columns=rename_map, inplace=True)
    cols = df_raw.columns.tolist()

    # best-effort column finders
    def find_col_like(keywords):
        for k in keywords:
            for c in cols:
                if k in c:
                    return c
        return None

    # mappings (best-effort)
    pop_c = find_col_like(["population", "total_population", "persons", "total_persons"])
    hh_c = find_col_like(["household", "households"])
    area_c = find_col_like(["ward", "area_name", "area", "name"])
    lit_c = find_col_like(["literacy", "literate"])
    elec_c = find_col_like(["electric"])
    toilet_c = find_col_like(["toilet"])
    water_c = find_col_like(["water", "drinking"])
    tv_c = find_col_like(["tv", "television", "tele"])

    # Start building working DF
    df = df_raw.copy()
    N = len(df)
    np.random.seed(42)

    # population
    if pop_c:
        try:
            df["population"] = pd.to_numeric(df[pop_c], errors="coerce")
        except Exception:
            df["population"] = np.nan
    else:
        df["population"] = np.nan

    # households
    if hh_c:
        try:
            df["households"] = pd.to_numeric(df[hh_c], errors="coerce")
        except Exception:
            df["households"] = np.nan
    else:
        df["households"] = np.nan

    # area_name
    if area_c:
        df["area_name"] = df[area_c].astype(str)
    else:
        df["area_name"] = [f"Area_{i}" for i in range(N)]

    # literacy
    if lit_c:
        df["literacy_rate"] = pd.to_numeric(df[lit_c], errors="coerce")
    else:
        df["literacy_rate"] = np.nan

    # Fill missing population/households with plausible synthetic values
    if df["population"].isna().all():
        df["population"] = np.random.randint(20000, 90000, N)
    else:
        df["population"].fillna(int(df["population"].median()), inplace=True)

    if df["households"].isna().all():
        df["households"] = (df["population"] / np.random.uniform(3.5, 5.0, N)).astype(int)
    else:
        df["households"] = df["households"].fillna((df["population"] / 4).round().astype(int))

    # literacy fallback
    if df["literacy_rate"].isna().all():
        df["literacy_rate"] = np.round(np.random.uniform(70, 97, N), 2)
    else:
        df["literacy_rate"].fillna(df["literacy_rate"].median(), inplace=True)

    # infrastructure / amenities percents
    df["electricity_access_pct"] = pct_from_col(df, elec_c, 95, N)
    df["toilet_facility_pct"] = pct_from_col(df, toilet_c, 85, N)
    df["tap_water_access_pct"] = pct_from_col(df, water_c, 80, N)
    df["ownership_tv_pct"] = pct_from_col(df, tv_c, 65, N)

    # Derived/synthetic socioeconomic attributes
    df["avg_household_size"] = (df["population"] / df["households"]).replace([np.inf, -np.inf], 4).fillna(4).round(2)
    df["avg_household_income_rs"] = np.random.randint(60000, 450000, N)
    df["poverty_index"] = np.clip(np.random.normal(0.25, 0.10, N), 0.03, 0.6).round(3)
    df["employment_rate"] = np.clip((df["literacy_rate"] / 100) * (1 - df["poverty_index"]) * np.random.uniform(0.7, 1.0, N), 0.35, 0.96).round(2)
    df["pct_informal_housing"] = np.clip(df["poverty_index"] * 100 + np.random.normal(5, 3, N), 3, 70).round(2)
    df["pct_elderly"] = np.clip(8 + (1 - df["poverty_index"]) * 15 + np.random.normal(0, 3, N), 3, 30).round(2)
    df["green_space_access_pct"] = np.clip(np.random.normal(30, 10, N) - df["poverty_index"] * 10, 2, 70).round(2)
    df["vulnerability_index"] = ((df["poverty_index"] * 0.45) + (df["pct_informal_housing"] / 100 * 0.3) + ((30 - df["pct_elderly"]) / 30 * 0.25)).round(3)

    # placeholder for H3 (to be filled after spatial join externally)
    df["h3_index"] = None

    # Create simple point geometry for visualization (randomly jittered around Bengaluru center)
    df["lat"] = CENTER_LAT + np.random.uniform(-0.15, 0.15, N)
    df["lon"] = CENTER_LON + np.random.uniform(-0.15, 0.15, N)
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Save CSV (straightforward)
    csv_out = output_dir / "socioeconomic_bengaluru_final.csv"
    gdf.to_csv(csv_out, index=False)
    log.info("Saved CSV: %s", csv_out)

    # Save GeoJSON (sanitize dtypes to avoid driver issues)
    geojson_out = output_dir / "socioeconomic_bengaluru_final.geojson"
    gdf2 = gdf.copy()
    # convert pandas nullable integers/floats to numpy-backed types for GeoJSON compatibility
    for c in gdf2.columns:
        try:
            if pd.api.types.is_integer_dtype(gdf2[c].dtype) or str(gdf2[c].dtype).startswith("Int"):
                gdf2[c] = gdf2[c].fillna(-1).astype("int64")
            elif pd.api.types.is_float_dtype(gdf2[c].dtype) or str(gdf2[c].dtype).startswith("Float"):
                gdf2[c] = gdf2[c].astype("float64")
        except Exception:
            pass
    try:
        gdf2.to_file(geojson_out, driver="GeoJSON")
        log.info("Saved GeoJSON: %s", geojson_out)
    except Exception as e:
        log.warning("GeoJSON write warning/failure: %s", e)

    # Robust parquet save with dtype sanitization (pyarrow-friendly)
    parquet_dir = output_dir
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / "socioeconomic_bengaluru_final.parquet"
    try:
        import pyarrow  # ensure available
        df_parquet = gdf2.copy()
        if "geometry" in df_parquet.columns:
            df_parquet["geometry_wkt"] = df_parquet["geometry"].apply(lambda g: g.wkt if g is not None else None)
            df_parquet = df_parquet.drop(columns=["geometry"])

        # sanitize object columns (decode bytes, coerce numeric if majority numeric)
        obj_cols = df_parquet.select_dtypes(include=["object"]).columns.tolist()
        log.info("Object-typed columns before sanitization: %s", obj_cols)
        for c in obj_cols:
            def decode_if_bytes(x):
                if isinstance(x, (bytes, bytearray)):
                    try:
                        return x.decode("utf-8", errors="ignore")
                    except Exception:
                        return str(x)
                return x
            df_parquet[c] = df_parquet[c].apply(decode_if_bytes)

            parsed = pd.to_numeric(df_parquet[c], errors="coerce")
            num_count = parsed.notna().sum()
            frac_num = num_count / max(1, len(parsed))
            if frac_num >= 0.85:
                log.info("Column '%s' appears mostly numeric (%.2f). Casting to float64.", c, frac_num)
                df_parquet[c] = parsed.astype("float64")
            else:
                log.info("Column '%s' treated as text. Casting to str.", c)
                df_parquet[c] = df_parquet[c].astype(str).fillna("").replace("nan", "")

        # convert pandas nullable ints/floats to numpy floats to avoid pyarrow issues
        for c in df_parquet.columns:
            try:
                if str(df_parquet[c].dtype).startswith("Int") or str(df_parquet[c].dtype).startswith("Float"):
                    df_parquet[c] = df_parquet[c].astype("float64")
            except Exception:
                pass

        # finally write parquet
        df_parquet.to_parquet(parquet_path, index=False, engine="pyarrow")
        log.info("Saved Parquet: %s", parquet_path)
    except Exception as e:
        log.exception("Parquet write failed after sanitization. Error: %s", e)
        log.warning("Parquet export skipped or failed; check df_parquet.dtypes for problematic columns.")

    # Save small preview JSON (non-geometry) for quick inspection
    preview_df = gdf2.drop(columns=["geometry"], errors="ignore").head(10)
    preview_json_path = output_dir / "socio_head_preview.json"
    preview_df.to_json(str(preview_json_path), orient="records")
    log.info("Saved preview JSON: %s", preview_json_path)

    # metadata
    metadata = {
        "source_excel": str(input_path),
        "rows": int(N),
        "generated_on": pd.Timestamp.now().isoformat(),
        "outputs": {
            "csv": str(csv_out),
            "geojson": str(geojson_out),
            "parquet": str(parquet_path) if parquet_path.exists() else None,
            "preview_json": str(preview_json_path),
        },
        "notes": "Features include real-extracted fields where available and synthetic fields for missing attributes. 'h3_index' placeholder added for later H3 join."
    }
    meta_path = output_dir / "socioeconomic_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)
    log.info("Saved metadata: %s", meta_path)

    log.info("Done. Created %d socioeconomic records.", N)


if __name__ == "__main__":
    try:
        build_socioeconomic_dataset(INPUT_PATH, OUTPUT_DIR)
    except Exception as exc:
        log.exception("Failed to build socioeconomic dataset: %s", exc)
        sys.exit(1)
