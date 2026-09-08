# =========================================================
# Multi-model comparison: non-additive normalized matrix
# LR / RF / XGBoost — 5-fold Stratified CV with HP tuning
# =========================================================

import os, warnings
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

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
MATRIX_PATH = r"../data/non_additive_matrix\non_additive_normalized.csv"
OUTPUT_DIR  = r"../data/ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS     = 5
RANDOM_SEED  = 42

# =========================================================
# HYPERPARAMETER GRIDS
# =========================================================

LR_GRID = {
    "C":        [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100],
    "penalty":  ["l1", "l2"],
    "solver":   ["saga"],
    "class_weight": ["balanced", None],
    "max_iter":  [5000],
}

RF_GRID = {
    "n_estimators":     [100, 200, 500],
    "max_depth":        [5, 10, 15, None],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", None],
    "class_weight":     ["balanced", "balanced_subsample", None],
}

XGB_GRID = {
    "n_estimators":     [100, 200, 500],
    "max_depth":        [3, 5, 7, 9],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma":            [0, 0.1, 0.3],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0],
    "scale_pos_weight": [1, 2, 3],
}


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 70)
print("  Loading non-additive normalized matrix ...")
print(f"  {MATRIX_PATH}")

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
# EVALUATE ONE MODEL
# =========================================================

def evaluate(name, model, param_grid, X, y, use_scaling=False):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    metrics = {"accuracy": [], "f1": [], "roc_auc": [], "pr_auc": [], "mcc": []}
    best_params_all = []
    imp_accum = np.zeros(X.shape[1])

    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        # optional scaling (for LR)
        if use_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
            # GridSearchCV needs X as ndarray
            search_model = model
        else:
            search_model = model

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + fold)

        # Use RandomizedSearchCV for speed (cap at 50 combos per fold)
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        n_iter = min(50, n_combos)

        if n_combos <= 50:
            search = GridSearchCV(
                search_model, param_grid,
                scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0,
            )
        else:
            search = RandomizedSearchCV(
                search_model, param_grid,
                n_iter=n_iter, scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0, random_state=RANDOM_SEED,
            )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params_all.append(search.best_params_)

        y_pred  = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))
        metrics["pr_auc"].append(average_precision_score(y_test, y_proba))
        metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))

        if hasattr(best, "feature_importances_"):
            imp_accum += best.feature_importances_
        elif hasattr(best, "coef_"):
            imp_accum += np.abs(best.coef_).flatten()

        print(f"  Fold {fold}: Acc={metrics['accuracy'][-1]:.3f}  "
              f"F1={metrics['f1'][-1]:.3f}  ROC={metrics['roc_auc'][-1]:.3f}  "
              f"PR={metrics['pr_auc'][-1]:.3f}  MCC={metrics['mcc'][-1]:+.3f}")

    # Best params (most frequent across folds)
    param_strs = [str(p) for p in best_params_all]
    best_overall = max(set(param_strs), key=param_strs.count) if param_strs else "N/A"

    return metrics, best_overall, imp_accum / N_SPLITS


# =========================================================
# RUN ALL MODELS
# =========================================================

MODELS = [
    ("Logistic Regression", LogisticRegression(random_state=RANDOM_SEED, max_iter=5000, n_jobs=-1), LR_GRID, True),
    ("Random Forest",      RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1), RF_GRID, False),
    ("XGBoost",            xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss"), XGB_GRID, False),
]

all_results = {}

for name, model, grid, use_scaling in MODELS:
    metrics, best_param, importance = evaluate(name, model, grid, X, y, use_scaling)
    all_results[name] = {
        "metrics": metrics,
        "best_param": best_param,
        "importance": importance,
    }

# =========================================================
# SUMMARY TABLE
# =========================================================

print("\n\n" + "=" * 90)
print("  SUMMARY — Non-additive Normalized (973 features, 411 samples)")
print("=" * 90)

print(f"{'Model':<24} {'Accuracy':>12} {'F1':>12} {'ROC-AUC':>12} {'PR-AUC':>12} {'MCC':>12}")
print("-" * 90)

summary_rows = []
for name, data in all_results.items():
    m = data["metrics"]
    row = {
        "Model":    name,
        "Accuracy": f"{np.mean(m['accuracy']):.3f} +- {np.std(m['accuracy']):.3f}",
        "F1":       f"{np.mean(m['f1']):.3f} +- {np.std(m['f1']):.3f}",
        "ROC_AUC":  f"{np.mean(m['roc_auc']):.3f} +- {np.std(m['roc_auc']):.3f}",
        "PR_AUC":   f"{np.mean(m['pr_auc']):.3f} +- {np.std(m['pr_auc']):.3f}",
        "MCC":      f"{np.mean(m['mcc']):.3f} +- {np.std(m['mcc']):.3f}",
        "Acc_mean": np.mean(m["accuracy"]),
        "F1_mean":  np.mean(m["f1"]),
        "ROC_mean": np.mean(m["roc_auc"]),
        "PR_mean":  np.mean(m["pr_auc"]),
        "MCC_mean": np.mean(m["mcc"]),
    }
    summary_rows.append(row)
    print(f"{row['Model']:<24} {row['Accuracy']:>12} {row['F1']:>12} {row['ROC_AUC']:>12} {row['PR_AUC']:>12} {row['MCC']:>12}")

# =========================================================
# RANKING
# =========================================================
print("\n" + "=" * 90)
print("  RANKING (by ROC-AUC)")
print("=" * 90)

for metric_key in ["ROC_mean", "PR_mean", "F1_mean", "MCC_mean", "Acc_mean"]:
    sorted_rows = sorted(summary_rows, key=lambda r: r[metric_key], reverse=True)
    label = metric_key.replace("_mean", "")
    print(f"\n  Best by {label}:")
    for i, r in enumerate(sorted_rows):
        crown = "CROWN" if i == 0 else f"  {i+1}. "
        print(f"  {crown} {r['Model']:<22}  "
              f"ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  F1={r['F1']}  MCC={r['MCC']}")

# =========================================================
# BEST MODEL DETAILS
# =========================================================
best_name = sorted(all_results.keys(), key=lambda n: np.mean(all_results[n]["metrics"]["roc_auc"]), reverse=True)[0]
best_data = all_results[best_name]
best_m = best_data["metrics"]

print(f"\n{'='*90}")
print(f"  BEST MODEL: {best_name}")
print(f"  Best HP (most frequent): {best_data['best_param']}")
print(f"  ROC-AUC: {np.mean(best_m['roc_auc']):.4f} +- {np.std(best_m['roc_auc']):.4f}")
print(f"  PR-AUC:  {np.mean(best_m['pr_auc']):.4f} +- {np.std(best_m['pr_auc']):.4f}")
print(f"  F1:      {np.mean(best_m['f1']):.4f} +- {np.std(best_m['f1']):.4f}")
print(f"  MCC:     {np.mean(best_m['mcc']):.4f} +- {np.std(best_m['mcc']):.4f}")

# Top 20 features
imp = best_data["importance"]
top_idx = np.argsort(imp)[::-1][:20]
print(f"\n  Top 20 features:")
print(f"  {'Rank':<5} {'Reaction':<20} {'Importance':>12}")
print(f"  {'─'*5} {'─'*20} {'─'*12}")
for rank, idx in enumerate(top_idx):
    print(f"  {rank+1:<5} {feat_cols[idx]:<20} {imp[idx]:>12.6f}")

print(f"\n{'='*90}")
print(f"  DONE. Results dir: {OUTPUT_DIR}")
print(f"{'='*90}")
