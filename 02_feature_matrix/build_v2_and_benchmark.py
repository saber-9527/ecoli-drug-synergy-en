# Build v2 non-additive matrix + benchmark vs v1
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SINGLE_DIR = r"../data/continuous_bounded_moma_single"
COMBO_V1   = r"../data/continuous_bounded_moma_all"
COMBO_V2   = r"../data/continuous_bounded_moma_all_v2"
META_FILE  = r"../data/metadata_merged.csv"
OUT_DIR    = r"../data/non_additive_matrix_v2"
os.makedirs(OUT_DIR, exist_ok=True)

CHANGE_THRESHOLD = 1e-6
MIN_FREQ_RATIO   = 0.1
MIN_FREQ_ABS     = 5
NAME_FIX = {'CLARITHROMYCIN': 'CLARYTHROMYCIN'}

# ── Load singles ──
single_flux = {}
for f in os.listdir(SINGLE_DIR):
    if f.endswith("_FLUX.csv"):
        drug = f.replace("_FLUX.csv", "").upper()
        single_flux[drug] = pd.read_csv(os.path.join(SINGLE_DIR, f)).set_index("Reaction")["ΔFlux"]
print(f"Singles: {len(single_flux)}")

# ── Load labels ──
meta = pd.read_csv(META_FILE)
meta.columns = meta.columns.str.strip().str.lower()
meta["drug_pair"] = meta["drug_pair"].astype(str).str.strip().str.upper().str.replace("-", "__")
label_map = dict(zip(meta["drug_pair"], meta["label"]))
combo_names = meta["drug_pair"].unique()

# ── Build non-additive for a given COMBO_DIR ──
def build_na(combo_dir, label):
    rows = []
    skipped = 0
    for pair in combo_names:
        parts = pair.split("__")
        if len(parts) != 2:
            continue
        a, b = NAME_FIX.get(parts[0], parts[0]), NAME_FIX.get(parts[1], parts[1])
        fa, fb = single_flux.get(a), single_flux.get(b)
        if fa is None and fb is None:
            continue
        cp = os.path.join(combo_dir, f"{pair}_FLUX.csv")
        if not os.path.exists(cp):
            skipped += 1
            continue
        cflux = pd.read_csv(cp).set_index("Reaction")
        if "ΔFlux" not in cflux.columns:
            continue
        delta = cflux["ΔFlux"] - (fa if fa is not None else 0) - (fb if fb is not None else 0)
        row = {"drug_pair": pair}
        for rid, val in delta.items():
            row[str(rid)] = val
        if pair in label_map:
            row["label"] = label_map[pair]
        rows.append(row)
    X = pd.DataFrame(rows).fillna(0)
    return X

print("\nBuilding v2 non-additive...")
Xv2 = build_na(COMBO_V2, "v2")
print(f"v2 raw: {Xv2.shape}")

# ── Normalize with filters ──
def normalize(X, name):
    id_cols = [c for c in ["drug_pair", "label"] if c in X.columns]
    feat = [c for c in X.columns if c not in id_cols]
    n = len(X)

    # zero-var
    stds = X[feat].std()
    keep = stds[stds > 0].index.tolist()
    # freq filter
    thresh = max(MIN_FREQ_ABS, round(MIN_FREQ_RATIO * n))
    freq = (np.abs(X[keep]) > CHANGE_THRESHOLD).sum(axis=0)
    keep = freq[freq >= thresh].index.tolist()
    # L1 norm
    vals = X[keep].div(X[keep].abs().sum(axis=1), axis=0).fillna(0)
    out = pd.concat([X[id_cols].reset_index(drop=True), vals], axis=1)

    path = os.path.join(OUT_DIR, f"non_additive_normalized_{name}.csv")
    out.to_csv(path, index=False)
    print(f"{name}: {out.shape}  ({len(keep)} features)")
    return out, keep

Xv2_norm, feat_v2 = normalize(Xv2, "v2")

# ── Load v1 for comparison ──
Xv1 = pd.read_csv(r"../data/non_additive_matrix\non_additive_normalized.csv")
print(f"\nv1 loaded: {Xv1.shape}")

# ── ML BENCHMARK ──
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

def run_one(name, X_df, model_class, grid, use_scaling, k=None):
    y = X_df["label"].dropna().values.astype(int)
    feats = [c for c in X_df.columns if c not in ("drug_pair", "label")]
    X_arr = X_df[feats].values.astype(np.float64)
    X_arr = np.nan_to_num(X_arr, nan=0, posinf=0, neginf=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m = {"accuracy": [], "f1": [], "roc_auc": [], "pr_auc": [], "mcc": []}

    for fold, (tr, te) in enumerate(skf.split(X_arr, y), 1):
        Xtr, Xte = X_arr[tr], X_arr[te]
        ytr, yte = y[tr], y[te]

        # Fold-internal feature selection if k is set
        if k and k < Xtr.shape[1]:
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
            rf.fit(Xtr, ytr)
            top = np.argsort(rf.feature_importances_)[::-1][:k]
            Xtr, Xte = Xtr[:, top], Xte[:, top]

        if use_scaling:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42+fold)
        search = RandomizedSearchCV(model_class, grid, n_iter=30, scoring="roc_auc", cv=inner, n_jobs=-1, random_state=42)
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

print("\n" + "=" * 90)
print("  BENCHMARK v1 vs v2")
print("=" * 90)

models = [
    ("RF",  RandomForestClassifier(random_state=42, n_jobs=-1), RF_GRID, False),
    ("XGB", xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"), XGB_GRID, False),
    ("LR",  LogisticRegression(random_state=42, max_iter=5000), LR_GRID, True),
]

results = []

for name, mcls, grid, scaling in models:
    for ver, Xdf in [("v1", Xv1), ("v2", Xv2_norm)]:
        nfeat = Xdf.shape[1] - 2
        metrics = run_one(f"{name}_{ver}", Xdf, mcls, grid, scaling, k=200)
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
            "Feat": nfeat,
        }
        results.append(r)

print(f"{'Model':<14} {'Acc':>12} {'F1':>12} {'ROC':>12} {'PR':>12} {'MCC':>12}")
print("-" * 86)
for r in results:
    print(f"{r['Model']:<14} {r['Accuracy']:>12} {r['F1']:>12} {r['ROC_AUC']:>12} {r['PR_AUC']:>12} {r['MCC']:>12}")

print("\nRANKING by ROC-AUC:")
for i, r in enumerate(sorted(results, key=lambda x: x["ROCm"], reverse=True)):
    print(f"  {i+1}. {r['Model']:<12} ROC={r['ROC_AUC']}  F1={r['F1']}  MCC={r['MCC']}")

# Best v2 feature summary
print("\n" + "=" * 90)
print("  V2 TOP FEATURES (from best model)")
print("=" * 90)
best_key = max(results, key=lambda r: r["ROCm"])
if "v2" in best_key["Model"]:
    # Quick RF feature importance on full v2
    feats_v2 = [c for c in Xv2_norm.columns if c not in ("drug_pair", "label")]
    Xtmp = Xv2_norm[feats_v2].values
    ytmp = Xv2_norm["label"].values.astype(int)
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtmp, ytmp)
    imp = rf.feature_importances_
    top = np.argsort(imp)[::-1][:20]
    for rank, idx in enumerate(top):
        print(f"  {rank+1:>2}. {feats_v2[idx]:<30} {imp[idx]:.6f}")

print(f"\nDone. v2 results: {OUT_DIR}")
