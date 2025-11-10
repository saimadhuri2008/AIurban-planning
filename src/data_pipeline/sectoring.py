# src/data_pipeline/sectoring.py
"""
Final 5-sector partitioning over the H3 grid produced by feature_aggregation.py.

Reads:
  data/processed/spatial/h3_gdf_final_static.geojson

Outputs:
  data/processed/spatial/h3_with_sectors.geojson          (geometry + fields)
  data/processed/spatial/h3_with_sectors.parquet          (no geometry; CSV fallback if Parquet fails)
  data/processed/sectors_kpis.csv                         (per-sector KPIs)

Sectors (fixed names):
  0: IT/Corporate Core
  1: Industrial Belt (powerplant)
  2: Residential-Prime + Entertainment
  3: Residential-General
  4: Residential-Affordable/Periphery
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- optional H3; used for fast adjacency if available
try:
    import h3
    HAS_H3 = True
except Exception:
    HAS_H3 = False

# --- H3 compatibility helpers (work across h3 v3/v4 APIs) ---
def h3_neighbors(hid: str, k: int = 1) -> set[str]:
    """
    Return k-ring neighbors of an H3 cell (excluding the cell itself)
    across h3 API variants.
    """
    try:
        if hasattr(h3, "k_ring"):  # old API
            return set(h3.k_ring(hid, k)) - {hid}
    except Exception:
        pass
    try:
        if hasattr(h3, "grid_disk"):  # new API (v4)
            return set(h3.grid_disk(hid, k)) - {hid}
    except Exception:
        pass
    try:
        if hasattr(h3, "hex_range"):  # very old alias
            return set(h3.hex_range(hid, k)) - {hid}
    except Exception:
        pass
    return set()

# --------- Paths ---------
ROOT = Path(__file__).resolve().parents[2]
IN_HEX  = ROOT / "data" / "processed" / "spatial" / "h3_gdf_final_static.geojson"
OUT_HEX_GEOJSON = ROOT / "data" / "processed" / "spatial" / "h3_with_sectors.geojson"
OUT_HEX_PARQUET = ROOT / "data" / "processed" / "spatial" / "h3_with_sectors.parquet"
OUT_KPI = ROOT / "data" / "processed" / "sectors_kpis.csv"

# --------- Settings ---------
K = 5
SEED = 42
CAP_MARGIN = 0.05  # ±5% slack
POP_SHARES  = np.array([0.25, 0.20, 0.20, 0.20, 0.15])  # sum to 1
JOBS_SHARES = np.array([0.45, 0.35, 0.10, 0.05, 0.05])  # sum to 1
MIN_SECTOR_SIZE = 30  # minimum cells per sector before reassignment (contiguity)

SECTOR_LABELS = {
    0: "IT/Corporate Core",
    1: "Industrial Belt (powerplant)",
    2: "Residential-Prime + Entertainment",
    3: "Residential-General",
    4: "Residential-Affordable/Periphery",
}

FEATURE_COLS: Tuple[str, ...] = ("pop", "jobs", "industrial_score", "amenity_density", "land_value")
IDX_COL = "h3_index"


def main(seed: int = SEED):
    np.random.seed(seed)
    print("[sectoring] reading hex grid:", IN_HEX)
    gdf = gpd.read_file(IN_HEX)

    if IDX_COL not in gdf.columns:
        raise KeyError(f"Expected '{IDX_COL}' in {IN_HEX.name}. Found: {list(gdf.columns)[:12]} ...")

    gdf = gdf.set_index(IDX_COL).sort_index()
    gdf = gdf.fillna(0)

    # ---------- Build required feature proxies from fused columns ----------
    # population proxy
    gdf["pop"] = np.maximum(gdf.get("sim_pop_density", 0.0).astype(float), 0.0)

    # jobs proxy: lights + transit ridership
    gdf["jobs"] = (
        1000.0 * gdf.get("static_ntl_intensity", 0.0).astype(float)
        + 0.5 * gdf.get("h3_avg_ridership", 0.0).astype(float)
    )

    # industrial intensity
    gdf["industrial_score"] = gdf.get("h3_avg_capacity", 0.0).astype(float)

    # amenities density: mobility + utilities POIs
    gdf["amenity_density"] = (
        gdf.get("poi_transit_count", 0.0).astype(float)
        + gdf.get("poi_utility_count", 0.0).astype(float)
    )

    # land value proxy: lights + ridership + green premium
    gdf["land_value"] = (
        1.5 * gdf.get("static_ntl_intensity", 0.0).astype(float)
        + 0.001 * gdf.get("h3_avg_ridership", 0.0).astype(float)
        + 0.3 * gdf.get("static_avg_ndvi", 0.0).astype(float)
    )

    missing = [c for c in FEATURE_COLS if c not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    # ---------- Feature matrix ----------
    X = gdf.loc[:, FEATURE_COLS].astype(float).values
    X = StandardScaler().fit_transform(X)

    # ---------- Capacitated k-means (approximate) ----------
    print("[sectoring] k-means init …")
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(X)
    centers = km.cluster_centers_
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    prefs = np.argsort(dists, axis=1)

    pop = gdf["pop"].values
    jobs = gdf["jobs"].values
    total_pop = float(pop.sum()  or 1.0)
    total_jobs = float(jobs.sum() or 1.0)

    pop_sh  = POP_SHARES / POP_SHARES.sum()
    jobs_sh = JOBS_SHARES / JOBS_SHARES.sum()

    pop_cap_lo  = pop_sh  * total_pop * (1 - CAP_MARGIN)
    pop_cap_hi  = pop_sh  * total_pop * (1 + CAP_MARGIN)
    jobs_cap_lo = jobs_sh * total_jobs * (1 - CAP_MARGIN)
    jobs_cap_hi = jobs_sh * total_jobs * (1 + CAP_MARGIN)

    pop_used  = np.zeros(K)
    jobs_used = np.zeros(K)
    assigned  = -np.ones(len(gdf), dtype=int)

    # Assign least decisive first (more flexible)
    decisiveness = dists.min(axis=1) / (dists.mean(axis=1) + 1e-6)
    order = np.argsort(decisiveness)

    print("[sectoring] capacity-aware assignment …")
    for i in order:
        for c in prefs[i]:
            if (pop_used[c] + pop[i]  <= pop_cap_hi[c]) and (jobs_used[c] + jobs[i] <= jobs_cap_hi[c]):
                assigned[i] = c
                pop_used[c]  += pop[i]
                jobs_used[c] += jobs[i]
                break
        if assigned[i] == -1:
            assigned[i] = prefs[i, 0]  # soft fallback

    # Light post-fix passes to reduce violations
    def _violation(pu, ju):
        v = 0.0
        for j in range(K):
            v += max(0.0, pu[j] - pop_cap_hi[j]) + max(0.0, pop_cap_lo[j] - pu[j])
            v += max(0.0, ju[j] - jobs_cap_hi[j]) + max(0.0, jobs_cap_lo[j] - ju[j])
        return v

    for _ in range(2):
        for i in range(len(gdf)):
            c0 = assigned[i]
            best = c0
            best_v = _violation(pop_used, jobs_used)
            for c in prefs[i]:
                if c == c0:
                    continue
                pu = pop_used.copy(); ju = jobs_used.copy()
                pu[c0] -= pop[i]; ju[c0] -= jobs[i]
                pu[c]  += pop[i]; ju[c]  += jobs[i]
                v = _violation(pu, ju)
                if v + 1e-9 < best_v and pu[c] <= pop_cap_hi[c] and ju[c] <= jobs_cap_hi[c]:
                    best, best_v = c, v
                    best_pu, best_ju = pu, ju
            if best != c0:
                assigned[i] = best
                pop_used, jobs_used = best_pu, best_ju

    if _violation(pop_used, jobs_used) > 0:
        print("⚠️  capacity constraints slightly violated; adjust shares or margin if needed.")

    gdf["sector_id"] = assigned

    # ---------- Contiguity enforcement ----------
    print("[sectoring] enforcing contiguity …")
    G = _build_adjacency(gdf)
    min_size = max(MIN_SECTOR_SIZE, len(gdf) // 300)  # scale with city size
    gdf["sector_id"] = _contiguity_fix(G, gdf["sector_id"].values, min_size=min_size, seed=seed)
    gdf["sector_name"] = gdf["sector_id"].map(SECTOR_LABELS)

    # ---------- KPIs ----------
    kpi = _sector_kpis(gdf)

    # ---------- Save ----------
    OUT_HEX_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    gdf.reset_index().to_file(OUT_HEX_GEOJSON, driver="GeoJSON")

    # save a light table without geometry for fast ML
    gdf_no_geom = gdf.drop(columns=["geometry"], errors="ignore").reset_index()
    try:
        gdf_no_geom.to_parquet(OUT_HEX_PARQUET, index=False)
    except Exception as e:
        print(f"Parquet save failed ({e}); writing CSV fallback.")
        gdf_no_geom.to_csv(OUT_HEX_PARQUET.with_suffix(".csv"), index=False)

    kpi.to_csv(OUT_KPI, index=False)

    print(f"[sectoring] sectors saved → {OUT_HEX_GEOJSON}")
    print(f"[sectoring] light table  → {OUT_HEX_PARQUET}")
    print(f"[sectoring] KPIs         → {OUT_KPI}")


# ---------- Helpers ----------

def _build_adjacency(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """Build hex-cell adjacency using H3 neighbors when available; else spatial index."""
    G = nx.Graph()
    ids = gdf.index.astype(str).tolist()
    G.add_nodes_from(ids)

    if HAS_H3:
        ids_set = set(ids)
        for hid in ids:
            for nb in h3_neighbors(hid, 1):
                if nb in ids_set:
                    G.add_edge(hid, nb)
        return G

    # Fallback: polygon touches/intersects (slower)
    sidx = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        for j in sidx.intersection(geom.bounds):
            if i == j:
                continue
            if geom.touches(gdf.geometry.iloc[j]) or geom.intersects(gdf.geometry.iloc[j]):
                G.add_edge(ids[i], ids[j])
    return G


def _contiguity_fix(G: nx.Graph, labels: np.ndarray, min_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    labs = labels.copy()

    for s in sorted(np.unique(labs)):
        part = [n for n in nodes if labs[idx[n]] == s]
        if not part:
            continue
        H = G.subgraph(part).copy()
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        if not comps:
            continue
        giant = comps[0]
        reassign = set().union(*comps[1:])
        if len(giant) < min_size:
            reassign = set(part)
        for n in reassign:
            nb = list(G.neighbors(n))
            if not nb:
                continue
            votes = {}
            for u in nb:
                lab = labs[idx[u]]
                votes[lab] = votes.get(lab, 0) + 1
            best = sorted(votes.items(), key=lambda kv: (-kv[1], rng.random()))[0][0]
            labs[idx[n]] = best
    return labs


def _sector_kpis(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    by = gdf.groupby(["sector_id", "sector_name"])
    out = by.agg(
        n_hex=("geometry", "count"),
        pop=("pop", "sum"),
        jobs=("jobs", "sum"),
        industrial_score=("industrial_score", "mean"),
        amenity_density=("amenity_density", "mean"),
        land_value=("land_value", "mean"),
        ntl=("static_ntl_intensity", "mean"),
        ndvi=("static_avg_ndvi", "mean"),
        ridership=("h3_avg_ridership", "mean"),
        utility_cap=("h3_avg_capacity", "mean"),
        utility_rel=("h3_avg_reliability", "mean"),
        income=("static_median_income_INR", "mean"),
        dependency=("static_dependency_ratio", "mean"),
    ).reset_index()
    out["pop_share"]  = out["pop"]  / (out["pop"].sum()  or 1.0)
    out["jobs_share"] = out["jobs"] / (out["jobs"].sum() or 1.0)
    return out


if __name__ == "__main__":
    main()
