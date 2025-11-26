#!/usr/bin/env python3
"""
analysis.py

Phase 2 — Step 2A: Descriptive + Spatial Statistical Baseline for Bengaluru.

Now updated so ALL outputs are written to: --outdir (required by DVC)
"""

import os
from pathlib import Path
import json
import warnings
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt
import matplotlib.pyplot as plt

# Stats / spatial libs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import libpysal
from esda import Moran, Moran_Local
from libpysal.weights import Queen, KNN, lag_spatial

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)


# ---------- ARGUMENTS ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wards", required=True, help="Path to wards_master_fixed.parquet")
    parser.add_argument("--outdir", required=True, help="Output directory (DVC stage output)")
    return parser.parse_args()


# ---------- UTILITIES ----------
def ensure_gdf(df):
    """Ensure object is GeoDataFrame with geometry column set and CRS is EPSG:4326."""
    if not isinstance(df, gpd.GeoDataFrame):
        if "geometry" in df.columns:
            geom_col = df["geometry"]
            if geom_col.dtype == object:
                sample = geom_col.dropna().iloc[0]
                if isinstance(sample, (bytes, bytearray)):
                    df["geometry"] = geom_col.apply(lambda b: wkb.loads(b) if pd.notnull(b) else None)
                elif isinstance(sample, str):
                    try:
                        df["geometry"] = geom_col.apply(lambda s: wkt.loads(s) if pd.notnull(s) else None)
                    except Exception:
                        pass
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
    else:
        gdf = df

    if gdf.geometry.name != "geometry":
        try:
            gdf = gdf.set_geometry("geometry", inplace=False)
        except Exception:
            geom_cols = [c for c in gdf.columns if gdf[c].dtype.name == "geometry"]
            if geom_cols:
                gdf = gdf.set_geometry(geom_cols[0], inplace=False)
            else:
                raise ValueError("No valid geometry column found in dataframe.")

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def save_json(obj, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------- LOAD WARDS ----------
def load_wards(path):
    print(f"Loading wards parquet: {path}")
    df = gpd.read_parquet(path)
    gdf = ensure_gdf(df)
    print(f"Loaded GeoDataFrame rows: {len(gdf)}; CRS: {gdf.crs}")
    return gdf


# ---------- DESCRIPTIVES ----------
def compute_basic_indicators(gdf, outdir):
    col_map = {
        "population": next((c for c in gdf.columns if c.startswith("population_est")), "population_est"),
        "population_2011": next((c for c in gdf.columns if c.startswith("population_2011")), "population_2011_x"),
        "it_jobs": next((c for c in gdf.columns if "it_job_density" in c), "it_job_density_mean"),
        "income": next((c for c in gdf.columns if "income_index" in c), "income_index_mean"),
        "congestion": next((c for c in gdf.columns if "congestion" in c), "congestion_index_mean"),
        "aqi": next((c for c in gdf.columns if "aqi" in c), "aqi_mean"),
        "built_area": next((c for c in gdf.columns if "built_area_m2_sum" in c), "built_area_m2_sum"),
        "outflow": next((c for c in gdf.columns if "total_outflow" in c), "total_outflow_sum"),
        "inflow": next((c for c in gdf.columns if "total_inflow" in c), "total_inflow_sum"),
        "schools": next((c for c in gdf.columns if "schools" in c), "schools_count_sum"),
        "health": next((c for c in gdf.columns if "health" in c), "health_count_sum"),
        "elec": next((c for c in gdf.columns if "electricity" in c), "electricity_asset_count")
    }

    for k, v in col_map.items():
        if v not in gdf.columns:
            gdf[v] = np.nan

    gdf["pop_k"] = gdf[col_map["population"]].astype(float) / 1000

    if "area_sqkm_x" in gdf.columns:
        gdf["pop_density_km2"] = gdf[col_map["population"]] / gdf["area_sqkm_x"]
    else:
        gdf["area_sqkm"] = gdf.geometry.to_crs(epsg=3857).area / 1e6
        gdf["pop_density_km2"] = gdf[col_map["population"]] / gdf["area_sqkm"]

    gdf["net_flow"] = gdf[col_map["inflow"]].fillna(0) - gdf[col_map["outflow"]].fillna(0)

    cols_for_summary = [
        col_map["population"], "pop_k", "pop_density_km2",
        col_map["it_jobs"], col_map["income"], col_map["congestion"],
        col_map["aqi"], col_map["built_area"], "net_flow",
        col_map["schools"], col_map["health"], col_map["elec"]
    ]

    summary = gdf[cols_for_summary].describe().transpose()
    summary.to_csv(os.path.join(outdir, "descriptive_summary.csv"))
    print("Wrote descriptive summary:", os.path.join(outdir, "descriptive_summary.csv"))
    return gdf, col_map


# ---------- SPATIAL ----------
def build_weights(gdf, k_fallback=5):
    try:
        w = Queen.from_dataframe(gdf)
        w.transform = "R"
        print("Built Queen contiguity weights.")
    except Exception as e:
        print("Queen contiguity failed (fallback to KNN):", e)
        coords = np.column_stack((gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values))
        w = KNN.from_array(coords, k=k_fallback)
        w.transform = "R"
        print(f"Built KNN (k={k_fallback}) weights.")
    return w


def compute_moran(gdf, w, var):
    x = gdf[var].fillna(0).astype(float).values
    mi = Moran(x, w)

    res = {
        "variable": var,
        "I": float(mi.I),
        "p_sim": float(mi.p_sim) if hasattr(mi, "p_sim") else None,
        "z_norm": float(mi.z_norm) if hasattr(mi, "z_norm") else None,
        "n": int(mi.n)
    }

    local = Moran_Local(x, w)
    gdf[f"{var}_lisa_I"] = local.Is
    gdf[f"{var}_lisa_p"] = local.p_sim
    gdf[f"{var}_lisa_q"] = local.q
    return res, gdf


# ---------- CORRELATION & VIF ----------
def correlation_and_vif(gdf, feature_cols, outdir):
    df = gdf[feature_cols].copy().astype(float).fillna(0)
    corr = df.corr()
    corr.to_csv(os.path.join(outdir, "correlation_matrix.csv"))
    print("Wrote correlation matrix:", os.path.join(outdir, "correlation_matrix.csv"))

    X = sm.add_constant(df)
    vif_data = []
    for i, col in enumerate(df.columns):
        try:
            vif = variance_inflation_factor(X.values, i+1)
        except Exception:
            vif = np.nan
        vif_data.append({"variable": col, "VIF": float(vif)})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    vif_df.to_csv(os.path.join(outdir, "vif.csv"), index=False)
    print("Wrote VIF:", os.path.join(outdir, "vif.csv"))

    return corr, vif_df


# ---------- MODELS ----------
def fit_baseline_models(gdf, col_map, outdir):
    target_cont = col_map["population"]
    predictors = [
        col_map["income"],
        col_map["it_jobs"],
        col_map["built_area"],
        col_map["congestion"],
        col_map["aqi"],
    ]

    for p in predictors:
        if p not in gdf.columns:
            gdf[p] = 0

    model_df = gdf[[target_cont] + predictors].copy().dropna()
    model_df = model_df[model_df[target_cont].notnull()]
    model_df["log_" + target_cont] = np.log(model_df[target_cont].astype(float) + 1)

    formula_ols = "log_" + target_cont + " ~ " + " + ".join(predictors)
    ols = smf.ols(formula=formula_ols, data=model_df).fit(cov_type="HC3")
    with open(os.path.join(outdir, "ols_population_summary.txt"), "w", encoding="utf8") as f:
        f.write(ols.summary().as_text())
    print("Wrote OLS summary:", os.path.join(outdir, "ols_population_summary.txt"))

    count_var = col_map["outflow"]
    if count_var not in gdf.columns:
        print("Count variable missing; skipping Poisson.")
        return ols, None

    poisson_df = gdf[[count_var] + predictors].copy().dropna()
    poisson_df[count_var] = poisson_df[count_var].clip(lower=0)

    try:
        formula_pois = f"{count_var} ~ " + " + ".join(predictors)
        poisson = smf.glm(formula=formula_pois, data=poisson_df,
                          family=sm.families.Poisson()).fit(cov_type="HC3")
        with open(os.path.join(outdir, "poisson_outflow_summary.txt"), "w", encoding="utf8") as f:
            f.write(poisson.summary().as_text())
        print("Wrote Poisson summary:", os.path.join(outdir, "poisson_outflow_summary.txt"))
    except Exception as e:
        print("Poisson model failed:", e)
        poisson = None

    return ols, poisson


# ---------- MAIN ----------
def main():
    args = parse_args()

    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)

    gdf = load_wards(args.wards)

    # Step B
    gdf, col_map = compute_basic_indicators(gdf, OUT_DIR)

    # Save enriched wards (gpkg + csv)
    gdf.to_file(os.path.join(OUT_DIR, "wards_phase2_enriched.gpkg"), layer="wards", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(os.path.join(OUT_DIR, "wards_phase2_table.csv"), index=False)
    print("Saved enriched wards in:", OUT_DIR)

    # Step C: spatial weights
    w = build_weights(gdf)
    moran_results = []

    vars_to_test = [
        col_map["congestion"],
        col_map["aqi"],
        col_map["population"],
        col_map["income"],
        col_map["it_jobs"]
    ]

    for var in vars_to_test:
        if var in gdf.columns:
            res, gdf = compute_moran(gdf, w, var)
            moran_results.append(res)
            print(f"Moran {var}: I={res['I']:.4f}, p={res['p_sim']}")

    save_json(moran_results, os.path.join(OUT_DIR, "moran_results.json"))

    # LISA CSV
    lisa_cols = [c for c in gdf.columns if c.endswith("_lisa_I") or c.endswith("_lisa_p") or c.endswith("_lisa_q")]
    gdf[lisa_cols + ["ward_id"]].to_csv(os.path.join(OUT_DIR, "wards_lisa.csv"), index=False)
    print("Wrote LISA CSV:", os.path.join(OUT_DIR, "wards_lisa.csv"))

    # Step D
    feature_cols = [
        col_map["population"], col_map["income"], col_map["it_jobs"],
        col_map["congestion"], col_map["aqi"], col_map["built_area"]
    ]
    corr, vif_df = correlation_and_vif(gdf, feature_cols, OUT_DIR)

    # Step E
    ols, poisson = fit_baseline_models(gdf, col_map, OUT_DIR)

    # Model summary JSON
    try:
        model_summaries = {
            "ols_AIC": float(ols.aic),
            "ols_BIC": float(ols.bic),
            "ols_params": {k: float(v) for k, v in ols.params.items()}
        }
    except Exception:
        model_summaries = {}

    save_json(model_summaries, os.path.join(OUT_DIR, "model_summaries.json"))

    # Correlation heatmap
    try:
        import seaborn as sns
        sns.set(style="white")
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=-1, vmax=1)
        plt.title("Feature correlation matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=200)
        plt.close()
        print("Wrote correlation heatmap:", os.path.join(OUT_DIR, "correlation_heatmap.png"))
    except Exception:
        pass

    # Final enriched GeoJSON
    gdf.to_file(os.path.join(OUT_DIR, "wards_phase2_enriched.geojson"), driver="GeoJSON")
    print("Saved final enriched GeoJSON:", os.path.join(OUT_DIR, "wards_phase2_enriched.geojson"))

    print("Phase 2A (descriptive + spatial baseline) complete. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
