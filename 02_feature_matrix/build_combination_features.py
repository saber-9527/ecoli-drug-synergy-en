# %%
import os
import glob
import pandas as pd
import numpy as np

# =========================
# Config
# =========================
INPUT_DIR = r"../data/matched_flux"
RXN2SUB_PATH = r"../data/reaction_to_subsystem.csv"
OUTPUT_FILE = r"../data/combination_feature_matrix.csv"

DELTA_COL = "ΔFlux"
STATUS_COL = "Solution_Status"
EPS = 1e-6

# =========================
# Load Reaction → Subsystem
# =========================
rxn2sub = pd.read_csv(RXN2SUB_PATH)
rxn2sub["Reaction"] = rxn2sub["Reaction"].astype(str)

# =========================
# Single-file feature extraction
# =========================
def extract_features_from_file(filepath):
    df = pd.read_excel(filepath)

    # Keep only optimal solutions
    if STATUS_COL in df.columns:
        df = df[df[STATUS_COL] == "optimal"]

    df["Reaction"] = df["Reaction"].astype(str)

    # Drop the original (empty) Subsystem column to avoid merge conflicts
    if "Subsystem" in df.columns:
        df = df.drop(columns=["Subsystem"])

    # Merge to fill in the subsystem
    df = df.merge(rxn2sub, on="Reaction", how="left")

    # Drop reactions without a subsystem (exchange / sink)
    df["Subsystem"] = df["Subsystem"].fillna("").astype(str)
    df = df[df["Subsystem"].str.strip() != ""]

    # ΔFlux cleaning
    df[DELTA_COL] = pd.to_numeric(df[DELTA_COL], errors="coerce").fillna(0.0)
    df.loc[df[DELTA_COL].abs() < EPS, DELTA_COL] = 0.0

    features = {}
    for subsystem, g in df.groupby("Subsystem"):
        delta = g[DELTA_COL].values

        features[f"{subsystem}__mean_delta"] = delta.mean()
        features[f"{subsystem}__sum_abs_delta"] = np.abs(delta).sum()
        features[f"{subsystem}__max_abs_delta"] = np.abs(delta).max()
        features[f"{subsystem}__fraction_active"] = np.mean(np.abs(delta) > EPS)
        features[f"{subsystem}__variance_delta"] = np.var(delta)

    return features

# =========================
# Scan folders
# =========================
rows = []

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")))
for fp in files:
    combo = os.path.basename(fp).replace("gimme_flux_", "").replace(".xlsx", "")
    print("Processing:", combo)

    feat = extract_features_from_file(fp)
    feat["Combination"] = combo
    rows.append(feat)

# =========================
# Merge & save
# =========================
feature_df = pd.DataFrame(rows).set_index("Combination")
feature_df = feature_df.fillna(0.0)
feature_df.to_csv(OUTPUT_FILE)

print("Done.")
print("Feature matrix shape:", feature_df.shape)
print("Saved to:", OUTPUT_FILE)




