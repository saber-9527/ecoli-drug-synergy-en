# =========================================================
# Generate all ~8000+ combinations, take the non-additive fluxes,
# score them with the two saved models, and output a ranking
# =========================================================
import os, warnings, itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ============================================================
# Directories & loading
# ============================================================
SINGLE_DIR = r"../data/continuous_bounded_moma_single"
COMBO_DIR  = r"../data/continuous_bounded_moma_all"
OUT_DIR    = r"../data/predictions_all_combos"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Two models ----
class ClipOutliers:
    def __init__(self, lower=0.001, upper=0.999):
        self.lower_percentile = lower; self.upper_percentile = upper
        self.lower_ = None; self.upper_ = None
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lo_ = np.percentile(X, self.lower_percentile * 100, axis=0)
        self.hi_ = np.percentile(X, self.upper_percentile * 100, axis=0)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, self.lo_, self.hi_)

import sys
sys.modules["__main__"].ClipOutliers = ClipOutliers

# Model 1: 199-sample L1 (968 features)
pipe1 = joblib.load(r"../data/best_model_analysis_enhanced\best_logreg_pipeline_enhanced.joblib")
feats1 = pd.read_csv(r"../data/best_model_analysis_enhanced\ranked_features.csv")["Feature"].tolist()

# Model 2: 411-sample Compact (120 features)
pipe2 = joblib.load(r"../data/final_model\pipeline_compact.joblib")
feats2 = pd.read_csv(r"../data/final_model\feature_list_compact.csv")["Feature"].tolist()

print("Model 1 (199-sample):", len(feats1), "features, C=0.01")
print("Model 2 (411-sample):", len(feats2), "features, C=0.5623")

# ---- Single-drug list ----
all_singles = sorted([
    f.replace("_FLUX.csv", "") for f in os.listdir(SINGLE_DIR) if f.endswith("_FLUX.csv")
])
print(f"Single drugs: {len(all_singles)}")

# ---- Existing combo files ----
existing_combos = set(
    f.replace("_FLUX.csv", "") for f in os.listdir(COMBO_DIR) if f.endswith("_FLUX.csv")
)

# Generate all ordered combinations
all_pairs = sorted("__".join(p) for p in itertools.combinations(sorted(all_singles), 2))
total_pairs = len(all_pairs)
exist_pairs = [p for p in all_pairs if p in existing_combos]
print(f"All possible combos: {total_pairs}")
print(f"With existing MOMA:  {len(exist_pairs)}")

# ============================================================
# Preload all single-drug fluxes (in memory)
# ============================================================
single_flux = {}
for drug in all_singles:
    df = pd.read_csv(os.path.join(SINGLE_DIR, f"{drug}_FLUX.csv"))
    single_flux[drug] = df.set_index("Reaction")["ΔFlux"]

# ============================================================
# Compute non-additive features for each existing combo and score it
# ============================================================
rows = []
skipped_missing = 0

for i, pair in enumerate(exist_pairs):
    if (i + 1) % 500 == 0:
        print(f"  Processing {i+1}/{len(exist_pairs)} ...")

    d1, d2 = pair.split("__")

    # Single-drug flux (no deviation check - single-drug files always exist)
    f1 = single_flux[d1]
    f2 = single_flux[d2]

    # combo flux
    cpath = os.path.join(COMBO_DIR, f"{pair}_FLUX.csv")
    if not os.path.exists(cpath):
        skipped_missing += 1
        continue
    cflux = pd.read_csv(cpath).set_index("Reaction")
    if "ΔFlux" not in cflux.columns:
        skipped_missing += 1
        continue

    # non-additive delta
    delta = cflux["ΔFlux"].sub(f1).sub(f2).dropna()
    delta_dict = delta.to_dict()

    # ---- Model 1 (199, 968 features) ----
    X1 = np.zeros(len(feats1))
    for j, f in enumerate(feats1):
        X1[j] = delta_dict.get(f, 0.0)
    X1 = X1.reshape(1, -1)
    proba1 = pipe1.predict_proba(X1)[0, 1]

    # ---- Model 2 (411, 120 features) ----
    X2 = np.zeros(len(feats2))
    for j, f in enumerate(feats2):
        X2[j] = delta_dict.get(f, 0.0)
    X2 = X2.reshape(1, -1)
    proba2 = pipe2.predict_proba(X2)[0, 1]

    rows.append({
        "drug_pair": pair,
        "drug_a": d1,
        "drug_b": d2,
        "proba_199_model": float(proba1),
        "proba_411_model": float(proba2),
        "pred_199_model": 1 if proba1 >= 0.5 else 0,
        "pred_411_model": 1 if proba2 >= 0.5 else 0,
    })

print(f"\nDone: {len(rows)} predictions written")
print(f"Skipped (no combo file): {skipped_missing}")

# ============================================================
# DataFrame & output
# ============================================================
df = pd.DataFrame(rows)

# Ranking: descending mean probability of the two models
df["avg_proba"] = (df["proba_199_model"] + df["proba_411_model"]) / 2
df_sorted = df.sort_values("avg_proba", ascending=False).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "all_combo_predictions.csv")
df_sorted.to_csv(out_path, index=False, float_format="%.6f")
print(f"Saved: {out_path}")

# ---- Summary ----
print(f"\n{'='*70}")
print("  TOP 30 两模型都打高分的组合")
print(f"{'='*70}")
print(f"{'Rank':<5} {'Drug_A':<22} {'Drug_B':<22} {'199_Prob':>8} {'411_Prob':>8} {'Avg':>8}")
print(f"{'─'*5} {'─'*22} {'─'*22} {'─'*8} {'─'*8} {'─'*8}")
top30 = df_sorted.head(30)
for rank, (_, r) in enumerate(top30.iterrows(), 1):
    print(f"{rank:<5} {r['drug_a']:<22} {r['drug_b']:<22} "
          f"{r['proba_199_model']:>8.4f} {r['proba_411_model']:>8.4f} {r['avg_proba']:>8.4f}")

print(f"\n{'='*70}")
print("  BOTTOM 10 (最不可能协同)")
print(f"{'='*70}")
for rank, (_, r) in enumerate(df_sorted.tail(10).iterrows(), len(df_sorted) - 9):
    print(f"{rank:<5} {r['drug_a']:<22} {r['drug_b']:<22} "
          f"{r['proba_199_model']:>8.4f} {r['proba_411_model']:>8.4f} {r['avg_proba']:>8.4f}")

print(f"\n{'='*70}")
print(f"  DONE. 输出: {OUT_DIR}")
print(f"{'='*70}")
