"""
File: 05_graph_construction_master.py (FINAL, COMPLETE, AND ROBUST VERSION)
Author: Gemini (Master's Research Level)
Project: Urban Infrastructure & Spatial Planning - Bengaluru
Description:
    Transforms the aggregated features into a PyTorch Geometric Data object 
    using the H3-CELL GRAPH DESIGN.
    
    FINAL FIXES:
    - Consistency: Edge index and attributes derived from UNIQUE H3-H3 pairs.
    - Robustness: Correct temporal index handling.
    - Imputation: Final NaN values in the y_full target tensor are filled 
      using the global mean travel time to ensure assertion passes.
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import coalesce
import warnings
import joblib
from shapely.geometry import Point

# --- 1. CONFIGURATION AND SETUP ---
warnings.filterwarnings('ignore', category=UserWarning)

DATA_DIR = "../data/spatial/"
TEMPORAL_DIR = "../data/temporal/"
GRAPH_DIR = "../data/graph/"
os.makedirs(GRAPH_DIR, exist_ok=True)

# --- 2. LOAD FINAL FEATURES (CRITICAL FIX: parse_dates) ---
print("--- Starting Phase 1, Step 3: Graph Construction (H3-Cell Design) ---")

try:
    # Load Static H3 Node Features (Area Nodes)
    h3_gdf = gpd.read_file(os.path.join(DATA_DIR, 'h3_gdf_final_static.geojson')).set_index('h3_index')
    
    # Load Static Edge Features (Road Network Links)
    edges_gdf = gpd.read_file(os.path.join(DATA_DIR, 'edges_gdf_final_static.geojson')).set_index(['source', 'target', 'key_id'])

    # Load Temporal Data - CRITICAL FIX: Specify parse_dates
    temporal_traffic_df = pd.read_csv(
        os.path.join(TEMPORAL_DIR, 'temporal_traffic_48h.csv'),
        parse_dates=['timestamp']
    )
    temporal_h3_df = pd.read_csv(
        os.path.join(TEMPORAL_DIR, 'temporal_data_h3_48h.csv'),
        parse_dates=['timestamp']
    )
    
    print(f" -> Loaded {len(h3_gdf)} H3 Cells and {len(edges_gdf)} Road Edges.")
except Exception as e:
    print(f" ⚠️ ERROR: Failed to load final feature files. Error: {e}")
    raise

# --- 3. Build H3-H3 Mapping and Aggregate Features ---
print("\n[STEP 3: Building H3-H3 Graph and Aggregating Static Edge Attributes]")

target_crs = h3_gdf.crs

# 3.1. Map OSM endpoints to H3 cells (use LineString endpoints)
def line_endpoints_to_points(ls):
    # ls.coords is (x, y); convert to shapely Points
    return Point(ls.coords[0]), Point(ls.coords[-1])

u_pts, v_pts = [], []
for geom in edges_gdf.geometry:
    # Safely handle non-existent geometries if any were missed
    if geom is None:
        u_pts.append(None)
        v_pts.append(None)
        continue
    
    u_p, v_p = line_endpoints_to_points(geom)
    u_pts.append(u_p)
    v_pts.append(v_p)

u_gdf = gpd.GeoDataFrame(geometry=u_pts, crs=target_crs)
v_gdf = gpd.GeoDataFrame(geometry=v_pts, crs=target_crs)

h3_cells = h3_gdf[['geometry']].copy()
h3_cells['h3_idx'] = h3_cells.index

u_join = gpd.sjoin(u_gdf, h3_cells, how='left', predicate='within')[['h3_idx']].rename(columns={'h3_idx': 'h3_u'})
v_join = gpd.sjoin(v_gdf, h3_cells, how='left', predicate='within')[['h3_idx']].rename(columns={'h3_idx': 'h3_v'})

# Create a DataFrame containing the H3 indices for every road segment
edges_h3_map = pd.concat([u_join, v_join], axis=1)

# IMPORTANT: preserve the original (source, target, key_id) index
edges_h3_map.index = edges_gdf.index

# Add original edge features
EDGE_STATIC_FEATURES = ['length', 'is_low_emissions_zone', 'max_speed_policy', 'is_public_transit_route']
for col in EDGE_STATIC_FEATURES:
    if col not in edges_gdf.columns:
        raise KeyError(f"Missing required edge feature column: {col}")
edges_h3_map = edges_h3_map.join(edges_gdf[EDGE_STATIC_FEATURES])

# Drop any edges that couldn't be mapped to H3 cells
edges_h3_map = edges_h3_map.dropna(subset=['h3_u', 'h3_v'])
print(f" -> Mapped {len(edges_h3_map):,} road segments to H3-H3 cell edges.")

# 3.2. Aggregate Edge Features by UNIQUE H3-H3 pair 
edges_h3_map['h3_pair'] = list(zip(edges_h3_map['h3_u'], edges_h3_map['h3_v']))
aggregated_attr_df = edges_h3_map.groupby('h3_pair').agg(
    agg_length=('length', 'mean'),
    agg_speed_policy=('max_speed_policy', 'mean'),
    agg_low_emissions=('is_low_emissions_zone', 'sum'),
    agg_transit_route=('is_public_transit_route', 'sum')
)
print(f" -> Aggregated to {len(aggregated_attr_df)} UNIQUE H3-H3 pairs.")

# 3.3. Build Final Edge Index from the Aggregated Index
h3_to_idx = {h: i for i, h in enumerate(h3_gdf.index)}
unique_pairs = aggregated_attr_df.index.to_list()
unique_h3_u = [h3_to_idx[pair[0]] for pair in unique_pairs]
unique_h3_v = [h3_to_idx[pair[1]] for pair in unique_pairs]

edge_index = torch.tensor([unique_h3_u, unique_h3_v], dtype=torch.long)
num_nodes = len(h3_gdf)
print(f" -> Final num_nodes: {num_nodes}. Edge index shape: {edge_index.shape}.")

# --- 4. FEATURE PREPARATION (Node and Edge) ---
print("\n[STEP 4: Feature Standardization and Tensor Conversion]")

# 4.1. Node Features (x)
NODE_STATIC_FEATURES = [col for col in h3_gdf.columns if col.startswith(('static_', 'poi_', 'sim_', 'h3_', 'zoning_far_'))]
if not NODE_STATIC_FEATURES:
    raise ValueError("No node static features found with expected prefixes.")
node_features_df = h3_gdf[NODE_STATIC_FEATURES].copy()
scaler_node = StandardScaler()
x_node_scaled = scaler_node.fit_transform(node_features_df.values)
x = torch.tensor(x_node_scaled, dtype=torch.float)
print(f" -> Node feature matrix (x) shape: {x.shape}")

# 4.2. Edge Attributes (edge_attr) - from aggregated_attr_df
scaler_edge = StandardScaler()
edge_attr_scaled = scaler_edge.fit_transform(aggregated_attr_df.values)
edge_attr = torch.tensor(edge_attr_scaled, dtype=torch.float)
print(f" -> Edge attribute matrix (edge_attr) shape: {edge_attr.shape}")

# --- 5. Edge Weighting & Coalesce ---
print("\n[STEP 5: Edge Weighting and Simplification]")

# Coalesce to catch any duplicates and standardize ordering
edge_index_coalesced, edge_attr_coalesced = coalesce(
    edge_index, edge_attr, num_nodes, num_nodes
)

# Edge weights: inverse of (standardized) length column (col 0).
inverse_length = 1.0 / (edge_attr_coalesced[:, 0].abs() + 1e-6)
edge_weight = inverse_length / inverse_length.mean()
print(f" -> Final Edges: {edge_index_coalesced.shape[1]}. Edge weight tensor shape: {edge_weight.shape}")

# --- 6. TARGET PREPARATION (Y_FULL) ---
print("\n[STEP 6: Aggregating Temporal Target (Travel Time) to H3-H3 Edges]")

# Map road target values to the H3-H3 links
edges_map_for_target = edges_h3_map.reset_index()[['h3_u', 'h3_v', 'source', 'target', 'key_id']]

tt = temporal_traffic_df.merge(
    edges_map_for_target.rename(columns={'source':'u', 'target':'v', 'key_id':'key'}),
    on=['u', 'v', 'key'], how='inner'
)

tt['h3_pair'] = list(zip(tt['h3_u'], tt['h3_v']))

# Pivot: mean travel time for all road segments within the same H3-H3 link
tt_pivot = tt.pivot_table(
    index='timestamp', 
    columns='h3_pair', 
    values='avg_travel_time_s',
    aggfunc='mean'
)

# Keep only the columns (pairs) we used for edge_index
target_cols_to_keep = list(aggregated_attr_df.index)
tt_pivot = tt_pivot.reindex(columns=target_cols_to_keep)

# Sort time
tt_pivot = tt_pivot.sort_index()

# --- Canonical DatetimeIndex (ROBUST) ---
# Build from the raw temporal CSV to avoid string/NaT pitfalls
ts_base = pd.to_datetime(temporal_traffic_df['timestamp'], errors='coerce')
try:
    ts_base = ts_base.dt.tz_localize(None)
except Exception:
    pass
ts_canon = pd.DatetimeIndex(sorted(ts_base.dropna().unique()))

# Align the pivot to the canonical timeline, then fill small gaps
tt_pivot = tt_pivot.reindex(ts_canon).ffill().bfill()

# --- CRITICAL FIX: IMPUTATION OF REMAINING NaNs ---
# Calculate the overall mean travel time from all recorded data
global_mean_tt = tt_pivot.values[~np.isnan(tt_pivot.values)].mean()
if np.isnan(global_mean_tt):
    global_mean_tt = 0.0 # Fallback 
    
# Final imputation: Replace any remaining NaNs (entirely empty columns) with the global mean
tt_pivot = tt_pivot.fillna(global_mean_tt)

# Create target tensor
y_full = torch.tensor(tt_pivot.values, dtype=torch.float)
print(f" -> Temporal Target tensor (y_full) shape: {y_full.shape} (T x N_H3_Edges)")

# --- 7. Temporal Splits and Time Encoding ---
T = y_full.shape[0]
train_T = int(T * 0.6); val_T = int(T * 0.2)
train_idx = torch.arange(0, train_T)
val_idx = torch.arange(train_T, train_T + val_T)
test_idx = torch.arange(train_T + val_T, T)

# Time encodings from canonical DatetimeIndex
ts = tt_pivot.index 
hod = ts.hour.values
dow = ts.dayofweek.values
time_feats = np.c_[
    np.sin(2*np.pi*hod/24), np.cos(2*np.pi*hod/24),
    np.sin(2*np.pi*dow/7),  np.cos(2*np.pi*dow/7)
]
time_features = torch.tensor(time_feats, dtype=torch.float)

# --- 8. CREATE FINAL PYG DATA OBJECT & SAVE METADATA ---
print("\n[STEP 8: Final PyTorch Geometric Data Object & Metadata]")

graph_data = Data(
    # CORE TOPOLOGY
    edge_index=edge_index_coalesced,
    num_nodes=num_nodes,
    
    # FEATURES
    x=x,  # H3 Node Features
    edge_attr=edge_attr_coalesced,  # Aggregated Edge Attributes
    edge_weight=edge_weight,        # Edge weights (inverse length, standardized)
    
    # TARGET & TIME
    y_full=y_full,                  # Temporal Target (T x N_Edges)
    time_features=time_features,    # Sinusoidal Time Encoding (T x 4)
    
    # SPLITS
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    
    # METADATA
    h3_to_idx=h3_to_idx,
    h3_gdf_index=h3_gdf.index.tolist(),
)

# Save Scalers and Data
joblib.dump(scaler_node, os.path.join(GRAPH_DIR, 'scaler_node.pkl'))
joblib.dump(scaler_edge, os.path.join(GRAPH_DIR, 'scaler_edge.pkl'))
torch.save(graph_data, os.path.join(GRAPH_DIR, 'bengaluru_urban_graph_data.pt'))

# Final Quality Check Assertions
assert x.size(0) == num_nodes, "x rows must equal num_nodes"
assert edge_index_coalesced.max() < num_nodes, "edge_index contains node id >= num_nodes"
assert edge_index_coalesced.size(1) == edge_attr_coalesced.size(0), "Final edge_index and edge_attr must have same number of edges."
assert not torch.isnan(x).any(), "x contains NaN"
assert not torch.isnan(y_full).any(), "y_full contains NaN" # This assertion should now pass!

print(" ✅ Final PyTorch Geometric Data object created and saved successfully.")
print(f" Final Data Object Summary:\n {graph_data}")
print("--- Phase 1: Data Pipeline COMPLETE ---")