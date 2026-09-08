# Step 3 only: rebuild non-additive matrix from existing MOMA results
# (with frequency filter — matching build_feature_matrix_only.py standards)
import os, gc
import numpy as np
import pandas as pd

SINGLE_OUTPUT_DIR = r"../data/continuous_bounded_moma_single"
COMBO_OUTPUT_DIR  = r"../data/continuous_bounded_moma_all"
OUTPUT_MATRIX_DIR = r"../data/non_additive_matrix"
METADATA_FILE     = r"../data/metadata_merged.csv"
os.makedirs(OUTPUT_MATRIX_DIR, exist_ok=True)

# Filtering params (consistent with build_feature_matrix_only.py & flux_ml_pipeline.py)
CHANGE_THRESHOLD     = 1e-6
MIN_FREQUENCY_RATIO  = 0.1
MIN_FREQUENCY_ABSOLUTE = 5

# Load single-drug results
print("Loading single-drug MOMA results...")

# Name fix: metadata uses CLARITHROMYCIN, but actual file uses CLARYTHROMYCIN
NAME_FIX = {'CLARITHROMYCIN': 'CLARYTHROMYCIN'}

single_flux = {}
for f in os.listdir(SINGLE_OUTPUT_DIR):
    if not f.endswith("_FLUX.csv"):
        continue
    drug = f.replace("_FLUX.csv", "").upper()
    df = pd.read_csv(os.path.join(SINGLE_OUTPUT_DIR, f))
    single_flux[drug] = df.set_index("Reaction")["ΔFlux"]

print(f"  Loaded {len(single_flux)} single drugs")
print(f"  Drugs: {sorted(single_flux.keys())[:10]} ...")

# Load combo pair names
print("\nLoading combo pair names...")
if os.path.exists(METADATA_FILE):
    meta = pd.read_csv(METADATA_FILE)
    meta.columns = meta.columns.str.strip().str.lower()
    meta["drug_pair"] = meta["drug_pair"].astype(str).str.strip().str.upper().str.replace("-", "__")
    combo_names = meta["drug_pair"].unique()
    label_map = dict(zip(meta["drug_pair"], meta["label"]))
    print(f"  Combo pairs: {len(combo_names)} (from metadata)")
else:
    combo_names = sorted(set(f.replace("_FLUX.csv", "") for f in os.listdir(COMBO_OUTPUT_DIR) if f.endswith("_FLUX.csv")))
    label_map = None
    print(f"  Combo pairs: {len(combo_names)} (from flux dir)")

# Build non-additive matrix
print("\nBuilding non-additive matrix...")
non_additive_rows = []
skipped_no_singles = 0
skipped_no_combo = 0
matched = 0

for pair in combo_names:
    # Parse AMIKACIN__AMOXICILLIN -> (AMIKACIN, AMOXICILLIN)
    parts = pair.split("__")
    if len(parts) != 2:
        print(f"  !  Cannot parse: {pair}")
        continue
    drug_a, drug_b = parts[0], parts[1]
    # Apply name fixes
    drug_a = NAME_FIX.get(drug_a, drug_a)
    drug_b = NAME_FIX.get(drug_b, drug_b)

    flux_a = single_flux.get(drug_a)
    flux_b = single_flux.get(drug_b)

    if flux_a is None and flux_b is None:
        skipped_no_singles += 1
        continue

    combo_path = os.path.join(COMBO_OUTPUT_DIR, f"{pair}_FLUX.csv")
    if not os.path.exists(combo_path):
        skipped_no_combo += 1
        continue
    combo_df = pd.read_csv(combo_path)
    if "ΔFlux" not in combo_df.columns:
        continue
    combo_flux = combo_df.set_index("Reaction")["ΔFlux"]

    # Non-additive effect
    if flux_a is not None and flux_b is not None:
        delta = combo_flux - flux_a - flux_b
        status = "both"
    elif flux_a is not None:
        delta = combo_flux - flux_a
        status = "only_a"
    else:
        delta = combo_flux - flux_b
        status = "only_b"

    row = {"drug_pair": pair}
    for rxn_id, val in delta.items():
        row[str(rxn_id)] = val
    if label_map is not None and pair in label_map:
        row["label"] = label_map[pair]

    non_additive_rows.append(row)
    matched += 1

print(f"\n  Matched: {matched}")
print(f"  No singles found: {skipped_no_singles}")
print(f"  No combo file: {skipped_no_combo}")

# Build and save matrix
X = pd.DataFrame(non_additive_rows).fillna(0)
print(f"  Matrix shape: {X.shape}")

id_cols = [c for c in ["drug_pair", "label"] if c in X.columns]
feat_cols = [c for c in X.columns if c not in id_cols]

# ============================================================
# RAW
# ============================================================
raw_path = os.path.join(OUTPUT_MATRIX_DIR, "non_additive_raw.csv")
X.to_csv(raw_path, index=False)
print(f"  Saved raw: {raw_path}")

# ============================================================
# NORMALIZED — zero-var + frequency filter + L1 row norm
# ============================================================
n_samples = len(X)

# Step 1: zero-variance filter
stds = X[feat_cols].std()
feat_keep = stds[stds > 0].index.tolist()
print(f"\n  After zero-var   : {len(feat_keep)} features")

# Step 2: frequency filter (>=10% of samples, min 5)
freq_threshold = max(MIN_FREQUENCY_ABSOLUTE, round(MIN_FREQUENCY_RATIO * n_samples))
freq = (np.abs(X[feat_keep]) > CHANGE_THRESHOLD).sum(axis=0)
feat_keep = freq[freq >= freq_threshold].index.tolist()
print(f"  After freq filt  : {len(feat_keep)} features  (threshold >= {freq_threshold})")

# Step 3: L1 row normalization
X_norm_vals = X[feat_keep].div(X[feat_keep].abs().sum(axis=1), axis=0).fillna(0)
X_norm = pd.concat([X[id_cols].reset_index(drop=True), X_norm_vals], axis=1)
norm_path = os.path.join(OUTPUT_MATRIX_DIR, "non_additive_normalized.csv")
X_norm.to_csv(norm_path, index=False)
print(f"  Saved normalized: {norm_path}")
print(f"  Final shape      : {X_norm.shape}")

# Label distribution
if "label" in X.columns:
    print(f"\n  Label distribution:")
    print(f"    Synergy=1: {(X['label']==1).sum()}")
    print(f"    Synergy=0: {(X['label']==0).sum()}")
    print(f"    Unknown:   {X['label'].isna().sum()}")

print("\n** Done")
