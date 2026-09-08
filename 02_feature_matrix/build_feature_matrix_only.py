# %%
# Standalone Part B: Build feature matrices from existing _FLUX.xlsx
import os
import gc
import glob
import numpy as np
import pandas as pd

OUTPUT_DIR  = r"../data/continuous_bounded_moma_full-new"
METADATA_FILE = r"../data/metadata.csv"
FEATURE_COLUMN = "ΔFlux"
CHANGE_THRESHOLD = 1e-6
MIN_FREQUENCY_RATIO = 0.1
MIN_FREQUENCY_ABSOLUTE = 5

# Load labels
label_map = None
if os.path.exists(METADATA_FILE):
    meta_df = pd.read_csv(METADATA_FILE)
    meta_df.columns = meta_df.columns.str.strip().str.lower()
    if "drug_pair" in meta_df.columns and "label" in meta_df.columns:
        meta_df["drug_pair"] = meta_df["drug_pair"].astype(str).str.strip().str.upper()
        # Metadata: "AMIKACIN-AMOXICILLIN"
        # Flux files: "AMIKACIN__AMOXICILLIN_FLUX.xlsx"  (double underscore!)
        meta_df["drug_pair"] = meta_df["drug_pair"].str.replace("-", "__")
        label_map = dict(zip(meta_df["drug_pair"], meta_df["label"]))
        n_pos = (meta_df["label"] == 1).sum()
        n_neg = (meta_df["label"] == 0).sum()
        print(f"Labels loaded: {len(meta_df)} total (synergy=1: {n_pos}, non-synergy=0: {n_neg})")

# Scan flux files (CSV)
flux_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_FLUX.csv")])
print(f"Flux files found: {len(flux_files)}")

# Build feature matrix - process in chunks to manage memory
all_rows = []

# Check if partial save exists to resume
CHUNK_SIZE = 75
chunk_prefix = os.path.join(OUTPUT_DIR, "_temp_chunk_")

# Clean old temp files
for old in glob.glob(chunk_prefix + "*.pkl"):
    os.remove(old)

chunk_idx = 0
saved_chunks = []

for i, fname in enumerate(flux_files):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
    pair_name = fname.replace("_FLUX.csv", "").upper()
    row = {"drug_pair": pair_name}
    for _, r in df.iterrows():
        row[str(r["Reaction"])] = r[FEATURE_COLUMN]
    all_rows.append(row)
    del df

    if (i + 1) % CHUNK_SIZE == 0:
        chunk_path = f"{chunk_prefix}{chunk_idx}.pkl"
        pd.DataFrame(all_rows).to_pickle(chunk_path)
        saved_chunks.append(chunk_path)
        all_rows.clear()
        chunk_idx += 1
        print(f"  Saved chunk {chunk_idx} ({i+1}/{len(flux_files)})")
    gc.collect()

# Save last chunk
if all_rows:
    chunk_path = f"{chunk_prefix}{chunk_idx}.pkl"
    pd.DataFrame(all_rows).to_pickle(chunk_path)
    saved_chunks.append(chunk_path)
    all_rows.clear()
    print(f"  Saved final chunk ({len(flux_files)}/{len(flux_files)})")

# Merge chunks
print(f"  Merging {len(saved_chunks)} chunks...")
chunks = [pd.read_pickle(c) for c in saved_chunks]
X = pd.concat(chunks, ignore_index=True).fillna(0)
print(f"  Done, shape: {X.shape}")

# Attach label
if label_map is not None:
    X["label"] = X["drug_pair"].map(label_map)
    matched = X["label"].notna().sum()
    print(f"Labels matched: {matched} / {len(X)}")

feature_cols = [c for c in X.columns if c not in ("drug_pair", "label")]
print(f"Raw shape: {X.shape} ({len(feature_cols)} reactions x {len(X)} samples)")

# Zero-variance filter
print("Removing zero-variance reactions...")
stds = X[feature_cols].std()
kept_std = stds[stds > 0].index.tolist()
print(f"  Removed {len(feature_cols) - len(kept_std)}, kept {len(kept_std)}")
feature_cols = kept_std

# Frequency filter
n_samples = len(X)
freq_threshold = max(MIN_FREQUENCY_ABSOLUTE, round(MIN_FREQUENCY_RATIO * n_samples))
print(f"Frequency filter (threshold={freq_threshold}, {MIN_FREQUENCY_RATIO*100:.0f}% of {n_samples})...")
freq = (np.abs(X[feature_cols]) > CHANGE_THRESHOLD).sum(axis=0)
kept_freq = freq[freq >= freq_threshold].index.tolist()
print(f"  Removed {len(feature_cols) - len(kept_freq)}, kept {len(kept_freq)}")
feature_cols = kept_freq

id_cols = [c for c in ["drug_pair", "label"] if c in X.columns]

# Save raw
raw_path = os.path.join(OUTPUT_DIR, "flux_feature_matrix_raw.csv")
X[id_cols + feature_cols].to_csv(raw_path, index=False)
print(f"Saved raw: {raw_path}")

# Save normalized
norm_path = os.path.join(OUTPUT_DIR, "flux_feature_matrix_normalized.csv")
X_norm_vals = X[feature_cols].div(X[feature_cols].abs().sum(axis=1), axis=0).fillna(0)
pd.concat([X[id_cols].reset_index(drop=True), X_norm_vals], axis=1).to_csv(norm_path, index=False)
print(f"Saved normalized: {norm_path}")

# Reaction metadata
print("Building reaction metadata...")
# Read first flux CSV for Name/Subsystem
rxn_info = pd.read_csv(os.path.join(OUTPUT_DIR, flux_files[0]))[["Reaction", "Name", "Subsystem"]].drop_duplicates("Reaction")

X_all = pd.read_csv(raw_path)
full_features = [c for c in X_all.columns if c not in ("drug_pair", "label")]
X_sub = X_all[full_features]
full_freq = (np.abs(X_sub.values) > CHANGE_THRESHOLD).sum(axis=0)
full_std = X_sub.std().values
full_mean_abs = X_sub.abs().mean().values

meta = pd.DataFrame({
    "Reaction": full_features,
    "Frequency": full_freq,
    "Frequency_Ratio": full_freq / n_samples,
    "Mean_Abs_DeltaFlux": full_mean_abs,
    "Std_DeltaFlux": full_std,
    "Zero_Variance": (full_std == 0).astype(int),
})
meta = meta.merge(rxn_info, on="Reaction", how="left").sort_values("Frequency", ascending=False)

meta.to_csv(os.path.join(OUTPUT_DIR, "reaction_metadata.csv"), index=False)
meta[["Reaction", "Name", "Subsystem", "Frequency", "Frequency_Ratio"]].to_csv(
    os.path.join(OUTPUT_DIR, "reaction_frequency.csv"), index=False)

print("\nTop 20:")
print(f"{'Reaction':>12} {'Name':>40} {'Subsystem':>25} {'Freq':>6} {'Ratio':>6}")
print("-" * 90)
for _, r in meta.head(20).iterrows():
    name = str(r.get("Name", ""))[:38] if pd.notna(r.get("Name")) else ""
    sub = str(r.get("Subsystem", ""))[:23] if pd.notna(r.get("Subsystem")) else ""
    print(f"  {r['Reaction']:>12} {name:>40} {sub:>25} {r['Frequency']:>6} {r['Frequency_Ratio']:>5.0%}")

print("\n** Part B done")
print(f"Output dir: {OUTPUT_DIR}")
