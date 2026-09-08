# =========================================================
# Compact vs Full comparison - feature selection inside each fold, no data leakage
# Within each fold: RF importance from the training set -> top-K features -> GridSearch on the training set -> evaluate on the test set
# =========================================================

import os, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# =========================================================
# CONFIG
# =========================================================
MATRIX_PATH = r"../data/non_additive_matrix\non_additive_normalized.csv"
OUTPUT_DIR  = r"../data/ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS     = 5
RANDOM_SEED  = 42

# =========================================================
# LOAD DATA
# =========================================================
print("=" * 70)
print("  Loading data ...")
df = pd.read_csv(MATRIX_PATH)
print(f"  Shape: {df.shape}")

y = df["label"].dropna().values.astype(int)
id_cols = ["drug_pair", "label"]
feat_cols = [c for c in df.columns if c not in id_cols]
X = df[feat_cols].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Pos=1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  Neg=0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  Features: {X.shape[1]}")

# =========================================================
# HYPERPARAMETER GRIDS (reduced for speed)
# =========================================================

LR_GRID = {
    "C":        [0.01, 0.1, 1, 5, 50, 100],
    "penalty":  ["l1", "l2"],
    "solver":   ["saga"],
    "class_weight": ["balanced", None],
    "max_iter":  [5000],
}

RF_GRID = {
    "n_estimators":     [100, 200, 500],
    "max_depth":        [5, 10, 15, None],
    "min_samples_split":[2, 5],
    "min_samples_leaf": [1, 2],
    "max_features":     ["sqrt", "log2"],
    "class_weight":     ["balanced", "balanced_subsample"],
}

XGB_GRID = {
    "n_estimators":     [100, 200, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "gamma":            [0, 0.1],
    "reg_alpha":        [0, 0.1],
    "reg_lambda":       [1.0],
    "scale_pos_weight": [1, 2],
}

# =========================================================
# EVALUATE ONE CONFIG
# =========================================================

def evaluate_with_fold_selection(name, model_class, param_grid, X, y, k, use_scaling=False):
    """
    Each fold: train RF on training set -> get top K features -> train target model only on those.
    Returns per-fold metrics.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    metrics = {"accuracy": [], "f1": [], "roc_auc": [], "pr_auc": [], "mcc": []}
    top_features_all = []  # record which features were selected each fold

    print(f"\n{'='*60}")
    print(f"  [{name}]  Top K = {k}")
    print(f"{'='*60}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_train_full, X_test_full = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        # --- Step 1: Feature ranking via RF on train only (NO test data used) ---
        rf_ranker = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1,
            class_weight="balanced"
        )
        rf_ranker.fit(X_train_full, y_train)
        imp = rf_ranker.feature_importances_
        top_idx = np.argsort(imp)[::-1][:k]

        top_features_all.append(top_idx)

        X_train = X_train_full[:, top_idx]
        X_test  = X_test_full[:, top_idx]

        # --- Step 2: Scaling if needed ---
        if use_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # --- Step 3: GridSearch on train only (top K features) ---
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + fold)
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        n_iter = min(40, n_combos)

        if n_combos <= 40:
            search = GridSearchCV(
                model_class, param_grid,
                scoring="roc_auc", cv=inner_cv, n_jobs=-1, verbose=0,
            )
        else:
            search = RandomizedSearchCV(
                model_class, param_grid,
                n_iter=n_iter, scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0, random_state=RANDOM_SEED,
            )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # --- Step 4: Evaluate on held-out test set ---
        y_pred  = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))
        metrics["pr_auc"].append(average_precision_score(y_test, y_proba))
        metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))

        print(f"  Fold {fold}: Acc={metrics['accuracy'][-1]:.3f}  "
              f"F1={metrics['f1'][-1]:.3f}  ROC={metrics['roc_auc'][-1]:.3f}  "
              f"PR={metrics['pr_auc'][-1]:.3f}  MCC={metrics['mcc'][-1]:+.3f}")

    return metrics, top_features_all


# =========================================================
# RUN: Full (973) vs Compact (200) vs Compact (120) vs Compact (80)
# =========================================================

MODELS = [
    ("LR",  LogisticRegression(random_state=RANDOM_SEED, max_iter=5000), LR_GRID, True),
    ("RF",  RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1), RF_GRID, False),
    ("XGB", xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss"), XGB_GRID, False),
]

K_VALUES = [None, 200, 120, 80]  # None = use all features

all_results = {}

for k in K_VALUES:
    k_label = f"K={k}" if k is not None else "Full"
    for name, model_class, grid, use_scaling in MODELS:
        key = f"{name}_{k_label}"
        if k is None:
            # Use all features (no fold-internal selection)
            n_features_used = X.shape[1]
            metrics, top_features_all = evaluate_with_fold_selection(
                f"{name} (Full)", model_class, grid, X, y, X.shape[1], use_scaling)
        else:
            n_features_used = k
            metrics, top_features_all = evaluate_with_fold_selection(
                f"{name} (K={k})", model_class, grid, X, y, k, use_scaling)

        all_results[key] = {
            "metrics": metrics,
            "n_features": n_features_used,
            "top_features": top_features_all,
        }

# =========================================================
# SUMMARY TABLE
# =========================================================

print("\n\n" + "=" * 105)
print("  SUMMARY: Compact vs Full (fold-internal feature selection, NO data leak)")
print("=" * 105)

print(f"{'Model':<18} {'Acc':>10} {'F1':>10} {'ROC':>10} {'PR':>10} {'MCC':>10}")
print("-" * 105)

summary_rows = []
for name, data in all_results.items():
    m = data["metrics"]
    row = {
        "key": name,
        "Accuracy": f"{np.mean(m['accuracy']):.3f}+-{np.std(m['accuracy']):.3f}",
        "F1":       f"{np.mean(m['f1']):.3f}+-{np.std(m['f1']):.3f}",
        "ROC_AUC":  f"{np.mean(m['roc_auc']):.3f}+-{np.std(m['roc_auc']):.3f}",
        "PR_AUC":   f"{np.mean(m['pr_auc']):.3f}+-{np.std(m['pr_auc']):.3f}",
        "MCC":      f"{np.mean(m['mcc']):.3f}+-{np.std(m['mcc']):.3f}",
        "ROC_mean": np.mean(m["roc_auc"]),
        "PR_mean":  np.mean(m["pr_auc"]),
        "F1_mean":  np.mean(m["f1"]),
        "MCC_mean": np.mean(m["mcc"]),
        "Acc_mean": np.mean(m["accuracy"]),
        "n_feat":   data["n_features"],
    }
    summary_rows.append(row)
    print(f"{row['key']:<18} {row['Accuracy']:>10} {row['F1']:>10} {row['ROC_AUC']:>10} {row['PR_AUC']:>10} {row['MCC']:>10}")

# =========================================================
# RANKING
# =========================================================
print("\n" + "=" * 105)
print("  RANKING by ROC-AUC")
print("=" * 105)

sorted_rows = sorted(summary_rows, key=lambda r: r["ROC_mean"], reverse=True)
for i, r in enumerate(sorted_rows):
    crown = "CROWN" if i == 0 else f"  {i+1}."
    print(f"  {crown} {r['key']:<14} [{r['n_feat']:>3} feat]  "
          f"ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  F1={r['F1']}  MCC={r['MCC']}")

# =========================================================
# FEATURE STABILITY: how many features repeat across folds for K=200?
# =========================================================
print("\n" + "=" * 105)
print("  FEATURE STABILITY (across 5 folds, K=200)")
print("=" * 105)

for name in ["RF_K=200", "XGB_K=200", "LR_K=200"]:
    if name not in all_results:
        continue
    feats = all_results[name]["top_features"]
    # Count how many features appear in all 5 folds
    from collections import Counter
    all_ids = np.concatenate(feats)
    counts = Counter(all_ids)
    in_all_5 = sum(1 for c in counts.values() if c == 5)
    in_4plus = sum(1 for c in counts.values() if c >= 4)
    in_3plus = sum(1 for c in counts.values() if c >= 3)
    print(f"  {name}:  in 5/5 folds={in_all_5},  >=4 folds={in_4plus},  >=3 folds={in_3plus}")

    # Print top 10 most stable
    most_stable = counts.most_common(10)
    print(f"    Most stable features:")
    for feat_idx, cnt in most_stable:
        print(f"      {feat_cols[feat_idx]:<25} appeared in {cnt}/5 folds")

# =========================================================
# SAVE CSV
# =========================================================
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "compact_vs_full_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n  Saved: {summary_path}")

print(f"\n{'='*105}")
print(f"  DONE.")
print(f"{'='*105}")
