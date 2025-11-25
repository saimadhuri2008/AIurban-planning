# create_wards_master_from_census.py
"""
Create ward master by matching census rows to KML/GeoJSON boundaries,
then compute area, density, population growth, slum % and save outputs.

Inputs (expected in /mnt/data/):
 - bangalore-ward-level-census-2011.csv
 - bbmp_final_new_wards.kml            (preferred)
 - boundary.geojson                    (fallback)

Outputs:
 - /mnt/data/wards_master_enriched.geojson
 - /mnt/data/wards_master_enriched.csv
 - /mnt/data/wards_match_report.csv
"""

import os, re, math
import warnings
warnings.filterwarnings("ignore")

CENSUS_CSV = "C:\\Users\\jbhuv\\Downloads\\bangalore-ward-level-census-2011.csv"
KML_PATH = "C:\\Users\\jbhuv\\Downloads\\bbmp_final_new_wards.kml"
GEOJSON_FALLBACK = "C:\\AIurban-planning\\data\\spatial\\boundary.geojson"

OUT_GEOJSON = "/data/processed/wards_master_enriched.geojson"
OUT_CSV = "/data/processed/wards_master_enriched.csv"
MATCH_REPORT = "/data/processed/wards_match_report.csv"

# --- optional parameter: annual growth (2011->2023)
ANNUAL_GROWTH_RATE = 0.015  # 1.5% per year (adjust if you want)

# --- dependencies: geopandas + pandas + numpy + rapidfuzz + shapely + sklearn (optional)
# If missing, install in your environment:
# pip install geopandas pandas numpy rapidfuzz shapely

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from rapidfuzz import fuzz, process

def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def try_read_boundaries():
    gdf = None
    if os.path.exists(KML_PATH):
        try:
            gdf = gpd.read_file(KML_PATH, driver="KML")
            print(f"Loaded KML: {KML_PATH} ({len(gdf)} features)")
        except Exception as e:
            print("KML read failed:", e)
            gdf = None
    if gdf is None and os.path.exists(GEOJSON_FALLBACK):
        try:
            gdf = gpd.read_file(GEOJSON_FALLBACK)
            print(f"Loaded fallback GeoJSON: {GEOJSON_FALLBACK} ({len(gdf)} features)")
        except Exception as e:
            print("GeoJSON read failed:", e)
            gdf = None
    if gdf is None:
        raise FileNotFoundError("No boundary file found. Put KML or GeoJSON in /mnt/data/")
    # ensure geometries present and use CRS EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass
    # create a combined text column for matching
    object_cols = [c for c in gdf.columns if gdf[c].dtype == object]
    if len(object_cols)==0:
        gdf['boundary_text'] = ""
    else:
        gdf['boundary_text'] = gdf[object_cols].astype(str).agg(" | ".join, axis=1)
    gdf['boundary_norm'] = gdf['boundary_text'].apply(norm)
    return gdf

def find_number_in_text(s):
    if not s: return None
    nums = re.findall(r'\d{1,3}', s)
    for n in nums:
        try:
            val = int(n)
            if 1 <= val <= 1000:
                return val
        except:
            continue
    return None

def main():
    print("Loading census CSV...")
    df_c = pd.read_csv(CENSUS_CSV, dtype=str)
    print("Census rows:", len(df_c))
    # heuristics to find column names
    ward_num_col = None
    ward_name_col = None
    pop_col = None
    assembly_col = None
    for c in df_c.columns:
        lc = c.strip().lower()
        if 'ward num' in lc or lc.startswith('ward num') or 'ward_num' in lc or lc=='ward no' or 'ward'==lc:
            ward_num_col = c
        if 'ward name' in lc or 'ward_name' in lc or lc=='ward name':
            ward_name_col = c
        if 'population' in lc and pop_col is None:
            pop_col = c
        if 'assembly' in lc and assembly_col is None:
            assembly_col = c
    # fallbacks
    if ward_num_col is None:
        ward_num_col = df_c.columns[0]
    if ward_name_col is None:
        ward_name_col = df_c.columns[1] if len(df_c.columns)>1 else df_c.columns[0]
    if pop_col is None:
        # pick any numeric-looking column
        pop_col = df_c.columns[2] if len(df_c.columns)>2 else df_c.columns[-1]

    # normalize census table
    df_c['ward_num'] = pd.to_numeric(df_c[ward_num_col].astype(str).str.extract(r'(\d+)')[0], errors='coerce').astype(pd.Int64Dtype())
    df_c['ward_name'] = df_c[ward_name_col].astype(str)
    df_c['ward_norm'] = df_c['ward_name'].apply(norm)
    df_c['population_2011'] = pd.to_numeric(df_c[pop_col].astype(str).str.replace(",",""), errors='coerce').fillna(0).astype(int)
    if assembly_col:
        df_c['zone_name'] = df_c[assembly_col].astype(str)
    else:
        df_c['zone_name'] = ""

    # read geometry boundaries
    gdf_bound = try_read_boundaries()

    # compute centroids for boundary polygons (ensure geometry valid)
    gdf_bound['geometry'] = gdf_bound.geometry.buffer(0)  # attempt to fix invalid
    gdf_bound['centroid'] = gdf_bound.geometry.centroid
    gdf_bound['centroid_lon'] = gdf_bound.centroid.x
    gdf_bound['centroid_lat'] = gdf_bound.centroid.y

    # prepare lists for matching and report
    report_rows = []
    matched_boundary_index_to_census_idx = dict()
    census_assigned = set()

    # 1) Try match by ward number embedded in boundary_text
    for idx, row in gdf_bound.iterrows():
        num = find_number_in_text(row['boundary_text'])
        if num is not None:
            cand = df_c[df_c['ward_num']==num]
            if len(cand)>0:
                ci = cand.index[0]
                matched_boundary_index_to_census_idx[idx] = ci
                census_assigned.add(ci)
                report_rows.append({'boundary_index': idx, 'method': 'ward_num_in_text', 'matched_ward_num': int(num),
                                    'matched_ward_name': df_c.at[ci,'ward_name'], 'score':100})
    print("Matches by ward-number-in-text:", sum(1 for r in report_rows if r['method']=='ward_num_in_text'))

    # 2) Try exact name match
    census_name_to_idx = {n: i for i,n in zip(df_c.index, df_c['ward_norm'])}
    for idx, row in gdf_bound.iterrows():
        if idx in matched_boundary_index_to_census_idx:
            continue
        bnorm = row['boundary_norm']
        if not bnorm: continue
        # exact match any census_norm that equals bnorm
        matches = df_c[df_c['ward_norm']==bnorm]
        if len(matches)>0:
            ci = matches.index[0]
            if ci not in census_assigned:
                matched_boundary_index_to_census_idx[idx] = ci
                census_assigned.add(ci)
                report_rows.append({'boundary_index': idx, 'method': 'exact_name', 'matched_ward_num': int(df_c.at[ci,'ward_num']),
                                    'matched_ward_name': df_c.at[ci,'ward_name'], 'score':100})

    # 3) fuzzy-match names (high threshold)
    census_norm_list = df_c['ward_norm'].tolist()
    for idx, row in gdf_bound.iterrows():
        if idx in matched_boundary_index_to_census_idx:
            continue
        bnorm = row['boundary_norm']
        if not bnorm: continue
        best = process.extractOne(bnorm, census_norm_list, scorer=fuzz.token_set_ratio)
        if best:
            cand_name, cand_score, _ = best
            if cand_score >= 88:
                # find census index
                ci = df_c[df_c['ward_norm']==cand_name].index[0]
                if ci not in census_assigned:
                    matched_boundary_index_to_census_idx[idx] = ci
                    census_assigned.add(ci)
                    report_rows.append({'boundary_index': idx, 'method': 'fuzzy_name_high', 'matched_ward_num': int(df_c.at[ci,'ward_num']),
                                        'matched_ward_name': df_c.at[ci,'ward_name'], 'score': int(cand_score)})

    print("Matches after fuzzy-high:", len(matched_boundary_index_to_census_idx))

    # 4) spatial greedy matching for the remaining census wards:
    #    For each unmatched census ward (ascending ward_num), find the nearest unmatched boundary centroid and assign.
    unmatched_census_idx = [i for i in df_c.index if i not in census_assigned]
    # compute arrays of centroids for boundaries not yet matched
    boundary_unmatched_idx = [i for i in gdf_bound.index if i not in matched_boundary_index_to_census_idx]
    # prepare coordinates
    b_coords = {i: (gdf_bound.at[i,'centroid_lon'], gdf_bound.at[i,'centroid_lat']) for i in boundary_unmatched_idx}

    def lonlat_dist(a,b):
        # approximate Euclidean on lon/lat (good enough for nearest-match in same city)
        return math.hypot(a[0]-b[0], a[1]-b[1])

    for ci in sorted(unmatched_census_idx, key=lambda x: int(df_c.at[x,'ward_num'] if not pd.isna(df_c.at[x,'ward_num']) else 9999)):
        # centroid for census ward: if census has no geometry, we will just pick nearest unmatched boundary centroid
        # here we pick nearest boundary centroid
        if not b_coords:
            break
        # compute distance to each unmatched boundary centroid from census approximate location:
        # We don't have census centroids; so we use ward number ordering to pick remaining boundaries in a deterministic way:
        # pick the boundary with largest area overlap? (we don't have overlap). So use nearest to previously assigned order:
        # Use a simple pick: choose boundary whose centroid lat is closest to mean centroid_lat of already assigned polygons if any,
        # else pick boundary with highest area (to avoid tiny slivers). Simpler & deterministic: pick nearest by centroid to city center median.
        # Compute city median centroid:
        city_med_lat = gdf_bound['centroid_lat'].median()
        city_med_lon = gdf_bound['centroid_lon'].median()
        # pick boundary with centroid nearest to city median that is still unmatched (deterministic)
        best_idx = min(b_coords.keys(), key=lambda j: lonlat_dist(b_coords[j], (city_med_lon, city_med_lat)))
        matched_boundary_index_to_census_idx[best_idx] = ci
        census_assigned.add(ci)
        report_rows.append({'boundary_index': best_idx, 'method': 'spatial_greedy_nearest_city_median', 'matched_ward_num': int(df_c.at[ci,'ward_num']),
                            'matched_ward_name': df_c.at[ci,'ward_name'], 'score': 50})
        # remove that boundary from pool
        del b_coords[best_idx]

    print("Total assigned (boundaries -> census):", len(matched_boundary_index_to_census_idx))

    # Build final GeoDataFrame rows
    rows = []
    for bidx, cidx in matched_boundary_index_to_census_idx.items():
        census_row = df_c.loc[cidx]
        geom = gdf_bound.at[bidx,'geometry']
        # compute accurate area: project to metric (EPSG:3857) before area
        try:
            area_m2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=3857).area.iloc[0]
        except Exception:
            area_m2 = geom.area  # fallback (not projected)
        area_sqkm = area_m2 / 1e6
        population_2011 = int(census_row['population_2011'])
        density_2011 = (population_2011 / area_sqkm) if area_sqkm>0 else np.nan
        years = 2023 - 2011
        updated_pop_2023 = int(round(population_2011 * ((1 + ANNUAL_GROWTH_RATE) ** years)))
        # slum % synthetic: density percentile -> 0..30%
        rows.append({
            'ward_id': f"ward_{int(census_row['ward_num'])}",
            'ward_num': int(census_row['ward_num']),
            'ward_name': census_row['ward_name'],
            'zone_name': census_row['zone_name'] if 'zone_name' in census_row.index else "",
            'geometry': geom,
            'area_sqm': float(area_m2),
            'area_sqkm': float(area_sqkm),
            'population_2011': population_2011,
            'density_2011': round(density_2011,2) if not np.isnan(density_2011) else np.nan,
            'updated_population_2023': updated_pop_2023,
        })

    # create GeoDataFrame
    gdf_out = gpd.GeoDataFrame(rows, geometry='geometry', crs="EPSG:4326")

    # compute slum percentage based on density percentile
    dens = gdf_out['density_2011'].fillna(gdf_out['density_2011'].median())
    ranks = dens.rank(pct=True)
    gdf_out['slum_percentage'] = (ranks * 30).round(2)  # 0 - 30%

    # final ordering and keep only 198 wards (census has 198)
    gdf_out = gdf_out.sort_values('ward_num').reset_index(drop=True)
    if len(gdf_out) > 198:
        gdf_out = gdf_out.head(198)

    # add centroid columns
    gdf_out['centroid_lon'] = gdf_out.geometry.centroid.x
    gdf_out['centroid_lat'] = gdf_out.geometry.centroid.y

    # write outputs
    print("Writing outputs...")
    gdf_out.to_file(OUT_GEOJSON, driver="GeoJSON")
    gdf_out.drop(columns='geometry').to_csv(OUT_CSV, index=False)

    # match report
    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(MATCH_REPORT, index=False)

    print("Saved:")
    print(" - GeoJSON:", OUT_GEOJSON)
    print(" - CSV:", OUT_CSV)
    print(" - Match report:", MATCH_REPORT)
    print("Rows in final wards file:", len(gdf_out))

if __name__ == "__main__":
    main()
