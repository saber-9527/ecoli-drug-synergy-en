# Merge 74 new pairs with existing 411 pairs, build combined non-additive matrix, then benchmark
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SINGLE_DIR = r"../data/continuous_bounded_moma_single"
COMBO_DIR  = r"../data/continuous_bounded_moma_all"
OUT_DIR    = r"../data/merged_matrix"
os.makedirs(OUT_DIR, exist_ok=True)

NAME_FIX = {'CLARITHROMYCIN': 'CLARYTHROMYCIN'}
CHANGE_THRESHOLD = 1e-6
MIN_FREQ_RATIO   = 0.1
MIN_FREQ_ABS     = 5

# ── Drug abbreviation -> full name mapping ──
# NOTE: external source (Mol Syst Biol 12:872, Dataset EV1) uses "CEF" for
# CEFOXITIN, not Cefsulodin. Confirmed by matching 19 shared pairs against the
# Nature Comms 2020 screen sign column: CEFOXITIN agrees 84%, CEFSULODIN 73%.
ABBR_MAP = {
    "AMK": "AMIKACIN", "CEF": "CEFOXITIN", "CHL": "CHLORAMPHENICOL",
    "CIP": "CIPROFLOXACIN", "CLA": "CLARYTHROMYCIN", "ERY": "ERYTHROMYCIN",
    "FUS": "FUSIDICACID", "GEN": "GENTAMICIN", "LEV": "LEVOFLOXACIN",
    "NAL": "NALIDIXICACID", "NIT": "NITROFURANTOIN", "OXA": "OXACILLIN",
    "RIF": "RIFAMPICIN", "SPE": "SPECTINOMYCIN", "TET": "TETRACYCLINE",
    "TOB": "TOBRAMYCIN", "TRI": "TRIMETHOPRIM", "VAN": "VANCOMYCIN",
}

# ── Load single-drug fluxes ──
single_flux = {}
for f in os.listdir(SINGLE_DIR):
    if f.endswith("_FLUX.csv"):
        drug = f.replace("_FLUX.csv", "").upper()
        single_flux[drug] = pd.read_csv(os.path.join(SINGLE_DIR, f)).set_index("Reaction")["ΔFlux"]
print(f"Loaded {len(single_flux)} singles")

# ── Load existing combo flux files ──
combo_files = set(f.replace("_FLUX.csv", "").upper() for f in os.listdir(COMBO_DIR) if f.endswith("_FLUX.csv"))
print(f"Existing combo files: {len(combo_files)}")

# ── Identify 74 new pairs that exist as combo files ──
new_df = pd.read_excel(r"../data/desktop_new_pairs.xlsx", engine="openpyxl")

new_pairs = []
for _, row in new_df.iterrows():
    d1 = ABBR_MAP.get(row["Drug 1"].strip().upper())
    d2 = ABBR_MAP.get(row["Drug 2"].strip().upper())
    if d1 is None or d2 is None:
        continue
    pair = "__".join(sorted([d1, d2]))
    if pair in combo_files:
        new_pairs.append((pair, row["Experimental Interaction Score"]))

print(f"\nNew pairs with existing MOMA: {len(new_pairs)}")
for p, s in sorted(new_pairs):
    print(f"  {p}: score={s}")

# ── Build non-additive rows for these new pairs ──
def build_na_row(pair):
    parts = pair.split("__")
    a, b = NAME_FIX.get(parts[0], parts[0]), NAME_FIX.get(parts[1], parts[1])
    fa, fb = single_flux.get(a), single_flux.get(b)
    cp = os.path.join(COMBO_DIR, f"{pair}_FLUX.csv")
    cflux = pd.read_csv(cp).set_index("Reaction")
    delta = cflux["ΔFlux"] - (fa if fa is not None else 0) - (fb if fb is not None else 0)
    row = {"drug_pair": pair}
    for rid, val in delta.items():
        row[str(rid)] = val
    return row

print("\nBuilding non-additive for new pairs...")
new_rows = []
for pair, score in new_pairs:
    r = build_na_row(pair)
    r["score"] = score  # continuous regression target
    # Sign convention: in this source, score < 0 = SYNERGY, score > 0 = ANTAGONISM.
    # Verified zero-exception against the source paper's own sign column.
    # Label must therefore be 1 (synergy) when score < 0, matching metadata*.csv.
    r["label"] = 1 if score < 0 else 0
    new_rows.append(r)

X_new = pd.DataFrame(new_rows).fillna(0)
print(f"New data: {X_new.shape}")

# ── Load existing matrix and merge ──
X_old = pd.read_csv(r"../data/non_additive_matrix\non_additive_raw.csv")
print(f"Old data: {X_old.shape}")

# Align columns
if "score" not in X_old.columns:
    X_old["score"] = np.nan
if "score" not in X_new.columns:
    X_new["score"] = np.nan

# Find common feature columns
old_cols = set(X_old.columns)
new_cols = set(X_new.columns)
common = sorted(old_cols & new_cols)
id_cols = ["drug_pair", "label", "score"]
common_feat = [c for c in common if c not in id_cols]

# Union of features (fill missing with 0)
all_cols = sorted(set(list(X_old.columns) + list(X_new.columns)))
all_feat = [c for c in all_cols if c not in ["drug_pair", "label", "score"]]

X_old_aligned = X_old.reindex(columns=["drug_pair", "label", "score"] + all_feat).fillna(0)
X_new_aligned = X_new.reindex(columns=["drug_pair", "label", "score"] + all_feat).fillna(0)

# Mark provenance before concat — note .fillna(0) above turns the old rows' NaN
# score into 0.0, so the score column cannot be used to tell the two sets apart.
X_old_aligned["_src"] = "old"
X_new_aligned["_src"] = "new"

# Merge
X_merged = pd.concat([X_old_aligned, X_new_aligned], ignore_index=True)
print(f"\nMerged (raw): {X_merged.shape}")
print(f"  Old: {len(X_old)}  New: {len(X_new)}  Total: {len(X_merged)}")

# ── Deduplicate ──
# The external set shares drug pairs with the training set. Those rows carry
# IDENTICAL MOMA features but labels from a different platform, so keeping both
# feeds the model contradictory examples of the same input. Keep the training
# row (the established label), drop the duplicate external row, and write the
# overlapping pairs out separately — they are a useful cross-platform comparison.
ov_pairs = set(X_old["drug_pair"]) & set(X_new["drug_pair"])
if ov_pairs:
    old_lab = dict(zip(X_old["drug_pair"], X_old["label"]))
    new_lab = dict(zip(X_new["drug_pair"], X_new["label"]))
    conflict = [p for p in ov_pairs if old_lab[p] != new_lab[p]]
    print(f"\n  Overlap with training set: {len(ov_pairs)} pairs ({len(conflict)} with conflicting labels)")
    agree_pct = (len(ov_pairs) - len(conflict)) / len(ov_pairs) * 100
    print(f"  Cross-platform label agreement: {agree_pct:.1f}%")

    cmp_rows = [{"drug_pair": p, "train_label": int(old_lab[p]),
                 "external_label": int(new_lab[p]),
                 "agree": int(old_lab[p] == new_lab[p])} for p in sorted(ov_pairs)]
    cmp_path = os.path.join(OUT_DIR, "external_vs_train_labels.csv")
    pd.DataFrame(cmp_rows).to_csv(cmp_path, index=False)
    print(f"  Saved cross-platform comparison: {cmp_path}")

    drop = X_merged["drug_pair"].isin(ov_pairs) & (X_merged["_src"] == "new")
    X_merged = X_merged[~drop].reset_index(drop=True)
    print(f"  After dedup: {X_merged.shape}  (dropped {int(drop.sum())} duplicate external rows)")
    assert len(X_merged) == len(X_old) + len(X_new) - len(ov_pairs), "dedup dropped the wrong rows"

X_merged = X_merged.drop(columns=["_src"])

# Label distribution
print(f"  Old labels: 1={X_old['label'].sum():.0f}, 0={(X_old['label']==0).sum():.0f}")
print(f"  New labels: 1={X_new['label'].sum():.0f}, 0={(X_new['label']==0).sum():.0f}")
print(f"  Total:      1={X_merged['label'].sum():.0f}, 0={(X_merged['label']==0).sum():.0f}")

# ── Save raw merged ──
raw_path = os.path.join(OUT_DIR, "merged_non_additive_raw.csv")
X_merged.to_csv(raw_path, index=False)
print(f"\nSaved raw: {raw_path}")

# ── Build normalized with frequency filter ──
feat_all = all_feat
n = len(X_merged)

stds = X_merged[feat_all].std()
keep = stds[stds > 0].index.tolist()

freq_threshold = max(MIN_FREQ_ABS, round(MIN_FREQ_RATIO * n))
freq = (np.abs(X_merged[keep]) > CHANGE_THRESHOLD).sum(axis=0)
keep = freq[freq >= freq_threshold].index.tolist()
print(f"\nAfter zero-var+freq filter: {len(keep)} features")

vals = X_merged[keep].div(X_merged[keep].abs().sum(axis=1), axis=0).fillna(0)
id_cols_final = [c for c in ["drug_pair", "label", "score"] if c in X_merged.columns]
X_norm = pd.concat([X_merged[id_cols_final].reset_index(drop=True), vals], axis=1)

norm_path = os.path.join(OUT_DIR, "merged_non_additive_normalized.csv")
X_norm.to_csv(norm_path, index=False)
print(f"Saved normalized: {norm_path}  shape={X_norm.shape}")

# ── BENCHMARK ──
print("\n" + "="*80)
print(f"  BENCHMARK: Old ({len(X_old)}) vs Merged ({len(X_merged)})")
print("="*80)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RF_GRID = {
    "n_estimators": [100, 200, 500], "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5], "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"], "class_weight": ["balanced", "balanced_subsample"],
}
XGB_GRID = {
    "n_estimators": [100, 200, 500], "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8], "gamma": [0, 0.1],
    "reg_alpha": [0, 0.1], "reg_lambda": [1.0], "scale_pos_weight": [1, 2],
}
LR_GRID = {
    "C": [0.01, 0.1, 1, 5, 50, 100], "penalty": ["l1", "l2"],
    "solver": ["saga"], "class_weight": ["balanced", None], "max_iter": [5000],
}

# Also load old normalized for side-by-side comparison
X_old_norm = pd.read_csv(r"../data/non_additive_matrix\non_additive_normalized.csv")

def run_one(name, X_df, model_class, grid, use_scaling):
    y = X_df["label"].dropna().values.astype(int)
    feats = [c for c in X_df.columns if c not in ("drug_pair", "label", "score")]
    X_arr = X_df[feats].values.astype(np.float64)
    X_arr = np.nan_to_num(X_arr, nan=0, posinf=0, neginf=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m = {"accuracy": [], "f1": [], "roc_auc": [], "pr_auc": [], "mcc": []}

    for fold, (tr, te) in enumerate(skf.split(X_arr, y), 1):
        Xtr, Xte = X_arr[tr], X_arr[te]
        ytr, yte = y[tr], y[te]

        # Fold-internal feature selection (top 200)
        if Xtr.shape[1] > 200:
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
            rf.fit(Xtr, ytr)
            top = np.argsort(rf.feature_importances_)[::-1][:200]
            Xtr, Xte = Xtr[:, top], Xte[:, top]

        if use_scaling:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42+fold)
        search = RandomizedSearchCV(model_class, grid, n_iter=25, scoring="roc_auc", cv=inner, n_jobs=-1, random_state=42)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        yp = best.predict(Xte)
        yprob = best.predict_proba(Xte)[:, 1]
        m["accuracy"].append(accuracy_score(yte, yp))
        m["f1"].append(f1_score(yte, yp))
        m["roc_auc"].append(roc_auc_score(yte, yprob))
        m["pr_auc"].append(average_precision_score(yte, yprob))
        m["mcc"].append(matthews_corrcoef(yte, yp))

    return m

models = [
    ("RF",  RandomForestClassifier(random_state=42, n_jobs=-1), RF_GRID, False),
    ("XGB", xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"), XGB_GRID, False),
    ("LR",  LogisticRegression(random_state=42, max_iter=5000), LR_GRID, True),
]

results = []
for name, mcls, grid, scaling in models:
    for ver, Xdf in [(f"Old_{len(X_old_norm)}", X_old_norm), (f"Merged_{len(X_norm)}", X_norm)]:
        metrics = run_one(f"{name}_{ver}", Xdf, mcls, grid, scaling)
        r = {
            "Model": f"{name}_{ver}",
            "Accuracy": f"{np.mean(metrics['accuracy']):.3f}+-{np.std(metrics['accuracy']):.3f}",
            "F1": f"{np.mean(metrics['f1']):.3f}+-{np.std(metrics['f1']):.3f}",
            "ROC_AUC": f"{np.mean(metrics['roc_auc']):.3f}+-{np.std(metrics['roc_auc']):.3f}",
            "PR_AUC": f"{np.mean(metrics['pr_auc']):.3f}+-{np.std(metrics['pr_auc']):.3f}",
            "MCC": f"{np.mean(metrics['mcc']):.3f}+-{np.std(metrics['mcc']):.3f}",
            "ROCm": np.mean(metrics['roc_auc']),
            "PRm": np.mean(metrics['pr_auc']),
            "F1m": np.mean(metrics['f1']),
            "MCCm": np.mean(metrics['mcc']),
            "Accm": np.mean(metrics['accuracy']),
            "N": len(Xdf),
        }
        results.append(r)

print(f"\n{'Model':<18} {'N':>5} {'Acc':>12} {'F1':>12} {'ROC':>12} {'PR':>12} {'MCC':>12}")
print("-"*88)
for r in results:
    print(f"{r['Model']:<18} {r['N']:>5} {r['Accuracy']:>12} {r['F1']:>12} {r['ROC_AUC']:>12} {r['PR_AUC']:>12} {r['MCC']:>12}")

print("\nRANKING by ROC-AUC:")
for i, r in enumerate(sorted(results, key=lambda x: x["ROCm"], reverse=True)):
    delta = ""
    if "Merged" in r["Model"]:
        # find old counterpart
        old_key = r["Model"].replace("Merged_", "Old_")
        old_r = next((x for x in results if x["Model"] == old_key), None)
        if old_r:
            d = r["ROCm"] - old_r["ROCm"]
            delta = f"  (vs old: {d:+.3f})"
    print(f"  {i+1}. {r['Model']:<16} ROC={r['ROC_AUC']}  F1={r['F1']}  MCC={r['MCC']}{delta}")

print(f"\nDone. Merged matrix: {OUT_DIR}")
