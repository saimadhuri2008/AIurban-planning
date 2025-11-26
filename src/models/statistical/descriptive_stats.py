#!/usr/bin/env python3
"""
phase2b_descriptive.py

Phase 2B — Descriptive & Diagnostic Statistical Analysis (2B.1 - 2B.3)

Usage:
    python src/models/statistical/phase2b_descriptive.py \
        --wards data/processed/master/phase2/wards_phase2_enriched.geojson \
        --outdir results/phase2b_descriptive \
        --features population_est,it_job_density_mean,congestion_index_mean,aqi_mean,built_area_m2_sum

Outputs:
 - ward-level descriptive summary CSV
 - violin/boxplots, correlation heatmap
 - clustering outputs (kmeans, dbscan, hierarchical) and cluster maps
 - spatial stats: global Moran's I, LISA CSV & GeoJSON, Getis-Ord Gi* hotspots
 - saved GeoJSON with cluster + LISA + Gi* attributes

Dependencies:
 pip install pandas geopandas matplotlib seaborn scikit-learn libpysal esda mapclassify scipy seaborn
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import warnings
warnings.filterwarnings("ignore")

# spatial libs
import libpysal
from esda import Moran, Moran_Local, G_Local
import mapclassify

# clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import ListedColormap

sns.set(style="whitegrid", context="notebook")


def load_wards(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"wards file not found: {path}")
    # Accept parquet/geojson/gpkg/csv
    suffix = p.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
        # if geometry bytes present, convert
        if "geometry" in df.columns and not isinstance(df.loc[0, "geometry"], object):
            # geopandas can interpret WKB bytes via GeoSeries.from_wkb if necessary,
            # but GeoPandas read_parquet often preserves geometry; attempt direct
            try:
                gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            except Exception:
                gdf = gpd.GeoDataFrame(df)
        else:
            gdf = gpd.GeoDataFrame(df)
    elif suffix in [".geojson", ".json", ".gpkg", ".gpkg"]:
        gdf = gpd.read_file(p)
    else:
        df = pd.read_csv(p)
        gdf = gpd.GeoDataFrame(df)
    # Ensure geometry column is set
    if "geometry" in gdf.columns:
        try:
            gdf = gdf.set_geometry("geometry")
        except Exception:
            # sometimes geometry is WKB bytes: attempt to convert
            try:
                gdf["geometry"] = gpd.GeoSeries.from_wkb(gdf["geometry"])
                gdf = gdf.set_geometry("geometry")
            except Exception:
                pass
    # enforce CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def descriptive_stats(gdf, features, outdir):
    df = gdf.copy()
    stats = df[features].describe().transpose()
    stats["missing"] = df[features].isnull().sum()
    stats.to_csv(outdir / "descriptive_stats.csv")
    # correlation matrix
    corr = df[features].corr()
    corr.to_csv(outdir / "correlation_matrix.csv")
    # save correlation heatmap
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(outdir / "correlation_heatmap.png", dpi=200)
    plt.close()

    # violin + boxplots for key features
    fig, axs = plt.subplots(1, len(features), figsize=(4*len(features),4), squeeze=False)
    for i, f in enumerate(features):
        sns.violinplot(y=df[f], ax=axs[0,i], inner="quartile")
        axs[0,i].set_title(f)
    plt.tight_layout()
    plt.savefig(outdir / "violin_features.png", dpi=200)
    plt.close()

    # histograms
    df[features].hist(bins=20, figsize=(4*len(features),4))
    plt.tight_layout()
    plt.savefig(outdir / "hist_features.png", dpi=200)
    plt.close()

    return stats, corr


def clustering_analysis(gdf, features, outdir, k_list=(3,4,5), db_eps=0.5, db_min_samples=5):
    df = gdf.copy()
    X = df[features].fillna(df[features].median()).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cluster_results = {}

    # KMeans for multiple k
    kmeans_labels = {}
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(Xs)
        kmeans_labels[f"k{k}"] = labels
        # silhouette if >1 cluster
        sil = None
        if len(np.unique(labels)) > 1:
            try:
                sil = silhouette_score(Xs, labels)
            except Exception:
                sil = None
        cluster_results[f"k{k}"] = {"labels": labels, "inertia": float(km.inertia_), "silhouette": sil}
        df[f"kmeans_k{k}"] = labels

    # DBSCAN
    db = DBSCAN(eps=db_eps, min_samples=db_min_samples)
    db_labels = db.fit_predict(Xs)
    df["dbscan"] = db_labels
    cluster_results["dbscan"] = {"labels": db_labels}

    # Agglomerative clustering (hierarchical)
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(Xs)
    df["hier_4"] = agg_labels
    cluster_results["hier_4"] = {"labels": agg_labels}

    # Save clustering membership
    df_out = df.copy()
    cols_save = ["ward_id"] + [c for c in df_out.columns if c.startswith("kmeans_k") or c in ("dbscan","hier_4")]
    df_out[cols_save].to_csv(outdir / "cluster_memberships.csv", index=False)
    # Save kmeans centroids (scaled back)
    centroids = {}
    for k in k_list:
        key = f"k{k}"
        km = KMeans(n_clusters=int(k), random_state=0, n_init=10).fit(Xs)
        cent = scaler.inverse_transform(km.cluster_centers_)
        cent_df = pd.DataFrame(cent, columns=features)
        cent_df.to_csv(outdir / f"kmeans_centroids_k{k}.csv", index=False)
        centroids[key] = cent_df.to_dict(orient="list")

    # Map clusters on the geometry and save geojsons
    geo = gdf.copy()
    for k in k_list:
        geo[f"kmeans_k{k}"] = df[f"kmeans_k{k}"].values
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        geo.plot(column=f"kmeans_k{k}", categorical=True, legend=True, ax=ax)
        ax.set_title(f"KMeans k={k}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / f"kmeans_k{k}_map.png", dpi=200)
        plt.close()
    # DBSCAN map
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    geo["dbscan"] = df["dbscan"].values
    # cluster colors: noise (-1) separate color
    geo.plot(column="dbscan", categorical=True, legend=True, ax=ax)
    ax.set_title("DBSCAN clusters")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "dbscan_map.png", dpi=200)
    plt.close()

    # hierarchical dendrogram (for a small set; use a sample)
    try:
        Z = linkage(Xs, method="ward")
        plt.figure(figsize=(10,4))
        dendrogram(Z, no_labels=True, count_sort='ascending', truncate_mode='level', p=6)
        plt.title("Hierarchical clustering dendrogram (truncated)")
        plt.tight_layout()
        plt.savefig(outdir / "hier_dendrogram.png", dpi=200)
        plt.close()
    except Exception:
        pass

    return cluster_results, df_out


def spatial_analysis(gdf, target_vars, outdir):
    """
    Computes global Moran's I, local Moran (LISA), and Getis-Ord Gi* (G_Local).
    Saves CSVs and GeoJSON with LISA categories and Gi* z-scores.
    """
    g = gdf.copy()
    # project to geographic/equal-area? Moran works in geographic but recommended weights in planar:
    # We'll compute weights using queen contiguity on the polygon index (no reprojection needed for contiguity).
    # Build weights (queen)
    w = libpysal.weights.contiguity.Queen.from_dataframe(g)
    w.transform = "r"  # row-standardize
    out_summary = {"global_moran": {}, "lisa_files": [], "gi_files": []}

    for var in target_vars:
        vals = g[var].fillna(g[var].median()).values
        # global Moran
        try:
            mi = Moran(vals, w, two_tailed=True)
            out_summary["global_moran"][var] = {"I": float(mi.I), "p_norm": float(mi.p_norm), "z_norm": float(mi.z_norm)}
        except Exception as e:
            out_summary["global_moran"][var] = {"error": str(e)}
        # local Moran (LISA)
        try:
            local = Moran_Local(vals, w)
            # construct categories for LISA (HH, LL, HL, LH, not significant)
            sig = local.p_sim < 0.05
            quadrant = local.q
            lisa_cat = np.array(["NotSig"] * len(g), dtype=object)
            # q: 1 = HH, 2 = LH, 3 = LL, 4 = HL (esda doc)
            mask_hh = (quadrant == 1) & sig
            mask_ll = (quadrant == 3) & sig
            mask_hl = (quadrant == 4) & sig
            mask_lh = (quadrant == 2) & sig
            lisa_cat[mask_hh] = "HH"
            lisa_cat[mask_ll] = "LL"
            lisa_cat[mask_hl] = "HL"
            lisa_cat[mask_lh] = "LH"
            g[f"{var}_lisa_cat"] = lisa_cat
            g[f"{var}_lisa_z"] = local.z_sim
            # save LISA CSV
            g[["ward_id", var, f"{var}_lisa_cat", f"{var}_lisa_z"]].to_csv(outdir / f"lisa_{var}.csv", index=False)
            out_summary["lisa_files"].append(str(outdir / f"lisa_{var}.csv"))
        except Exception as e:
            out_summary["lisa_" + var] = {"error": str(e)}
        # Getis-Ord Gi* (G_Local)
        try:
            gl = G_Local(vals, w, transform="r")
            g[f"{var}_gi_z"] = gl.Zs
            g[f"{var}_gi_p"] = gl.p_sim
            g[["ward_id", var, f"{var}_gi_z", f"{var}_gi_p"]].to_csv(outdir / f"gi_{var}.csv", index=False)
            out_summary["gi_files"].append(str(outdir / f"gi_{var}.csv"))
        except Exception as e:
            out_summary["gi_" + var] = {"error": str(e)}

    # Save the enriched geojson
    g.to_file(outdir / "wards_phase2b_spatial_enriched.geojson", driver="GeoJSON")
    # Also save a CSV summary of all LISA / Gi values
    cols_keep = ["ward_id"] + [c for c in g.columns if any(s in c for s in ["_lisa_", "_gi_z", "_gi_p"])]
    g[cols_keep].to_csv(outdir / "spatial_stats_summary.csv", index=False)

    # Create choropleth maps for one variable examples
    for var in target_vars:
        # LISA map
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        catcol = f"{var}_lisa_cat"
        if catcol in g.columns:
            # color mapping
            cmap = {"NotSig":"lightgrey","HH":"red","LL":"blue","HL":"orange","LH":"cyan"}
            g.plot(column=catcol, categorical=True, legend=True, ax=ax, color=g[catcol].map(cmap))
            ax.set_title(f"LISA categories: {var}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(outdir / f"lisa_map_{var}.png", dpi=200)
            plt.close()
        # Gi* z-score map
        zcol = f"{var}_gi_z"
        if zcol in g.columns:
            fig, ax = plt.subplots(1,1,figsize=(8,6))
            g.plot(column=zcol, cmap="RdBu_r", legend=True, ax=ax)
            ax.set_title(f"Getis-Ord Gi* z-score: {var}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(outdir / f"gi_z_map_{var}.png", dpi=200)
            plt.close()

    # save summary JSON
    with open(outdir / "spatial_summary.json","w") as f:
        json.dump(out_summary, f, indent=2)

    return out_summary, g


def produce_basic_maps(gdf, var, outdir):
    g = gdf.copy()
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    g.plot(column=var, cmap="viridis", legend=True, ax=ax)
    ax.set_title(var)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / f"map_{var}.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wards", required=True, help="wards enriched geojson/parquet (phase2 output)")
    parser.add_argument("--outdir", default="results/phase2b_descriptive")
    parser.add_argument("--features", default="population_est,it_job_density_mean,congestion_index_mean,aqi_mean,built_area_m2_sum")
    parser.add_argument("--klist", default="3,4,5")
    parser.add_argument("--db_eps", default=0.5, type=float)
    parser.add_argument("--db_min_samples", default=5, type=int)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    klist = [int(x) for x in args.klist.split(",")]

    print("Loading wards:", args.wards)
    gdf = load_wards(args.wards)
    print("Rows:", len(gdf), "CRS:", gdf.crs)

    # ensure ward_id exists
    if "ward_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index":"ward_id"})
    # descriptive stats
    print("Computing descriptive stats for:", features)
    stats, corr = descriptive_stats(gdf, features, outdir)
    print("Descriptive stats saved.")

    # save enriched wards copy for Phase2B base
    gdf.to_file(outdir / "wards_phase2b_base.geojson", driver="GeoJSON")

    # produce per-variable maps
    for v in features:
        try:
            produce_basic_maps(gdf, v, outdir)
        except Exception:
            pass

    # clustering
    print("Running clustering analysis...")
    cluster_results, cluster_members = clustering_analysis(gdf, features, outdir, k_list=klist, db_eps=args.db_eps, db_min_samples=args.db_min_samples)
    with open(outdir / "clustering_summary.json","w") as f:
        json.dump(cluster_results, f, default=lambda o: str(o), indent=2)
    print("Clustering results saved.")

    # spatial stats (Moran, LISA, Gi*)
    print("Running spatial statistics (Moran, LISA, Gi*)...")
    spatial_vars = features  # run for all selected features
    spatial_summary, enriched = spatial_analysis(gdf, spatial_vars, outdir)
    print("Spatial stats saved.")

    # final: write a short human-readable summary
    summary_txt = outdir / "phase2b_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Phase 2B descriptive & diagnostic summary\n\n")
        f.write("Descriptive stats written to: descriptive_stats.csv\n")
        f.write("Correlation matrix: correlation_matrix.csv\n")
        f.write("Clustering memberships: cluster_memberships.csv\n")
        f.write("Spatial stats summary: spatial_stats_summary.csv\n")
        f.write("LISA & Gi* files: see spatial_summary.json\n")
    print("Phase 2B complete. Outputs in:", outdir)


if __name__ == "__main__":
    main()
