# %%
# =========================================================
# ML BENCHMARK: RF vs XGBoost on 4 flux feature matrices
#
# Model A: Combo Raw
# Model B: Combo Normalized
# Model C: Non-additive Raw
# Model D: Non-additive Normalized
#
# Metrics: Accuracy, F1, ROC-AUC, PR-AUC
# CV: 5-fold stratified, hyperparameter grid search
# =========================================================

import os
import warnings
import numpy as np
import pandas as pd

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, make_scorer)
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

# Input matrices
MATRIX_DIR = r"../data/continuous_bounded_moma_all"
NA_DIR     = r"../data/non_additive_matrix"

EXPERIMENTS = {
    "A  Combo Raw":              os.path.join(MATRIX_DIR, "flux_feature_matrix_raw.csv"),
    "B  Combo Normalized":       os.path.join(MATRIX_DIR, "flux_feature_matrix_normalized.csv"),
}

OUTPUT_DIR = r"../data/ml_results_merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CV settings
N_SPLITS   = 5
N_REPEATS  = 3                # repeated stratified k-fold
RANDOM_SEED = 42

# =========================================================
# HYPERPARAMETER GRIDS
# =========================================================

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
    "scale_pos_weight": [1, 2, 3],   # handles class imbalance
}

# =========================================================
# LOAD & PREP
# =========================================================

def load_matrix(path, name):
    """Load a feature matrix and return X, y."""
    df = pd.read_csv(path)
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  Path   : {path}")
    print(f"  Shape  : {df.shape}")

    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {path}")

    y = df["label"].dropna().values.astype(int)
    # Only keep samples with labels
    valid_idx = df["label"].notna().values
    id_cols = ["drug_pair", "label"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df.loc[valid_idx, feat_cols].values.astype(np.float64)

    # Replace inf / extreme values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extreme outliers (beyond 99.9 percentile)
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = np.percentile(col, [0.1, 99.9])
        col = np.clip(col, lo, hi)
        X[:, j] = col

    print(f"  Pos=1  : {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"  Neg=0  : {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    return X, y, feat_cols


# =========================================================
# GRID SEARCH + EVAL
# =========================================================

def evaluate_model(name, X, y, model_type, param_grid):
    """Run repeated stratified CV with grid search. Returns per-fold metrics + best params."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    results = {
        "accuracy": [], "f1": [], "roc_auc": [], "pr_auc": [], "mcc": [],
        "best_params": [], "feature_importance": None,
    }

    n_features = X.shape[1]
    importance_accum = np.zeros(n_features)

    if model_type == "rf":
        base_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    else:
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_SEED, n_jobs=-1,
            eval_metric="logloss", use_label_encoder=False,
        )

    # Use randomized search for larger grids (faster)
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    n_iter = min(60, n_combos)  # cap at 60 random combos
    search_mode = "RandomizedSearchCV" if n_combos > 60 else "GridSearchCV"

    print(f"\n  [{model_type.upper()}] {search_mode}  (max {n_iter} / {n_combos} combos)")

    fold_idx = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for HP tuning (3-fold)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + fold_idx)

        if search_mode == "GridSearchCV":
            search = GridSearchCV(
                base_model, param_grid,
                scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0,
            )
        else:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0, random_state=RANDOM_SEED,
            )

        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        results["best_params"].append(search.best_params_)

        # Predict on held-out fold
        y_pred = best_est.predict(X_test)
        y_proba = best_est.predict_proba(X_test)[:, 1]

        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["f1"].append(f1_score(y_test, y_pred))
        results["roc_auc"].append(roc_auc_score(y_test, y_proba))
        results["pr_auc"].append(average_precision_score(y_test, y_proba))
        results["mcc"].append(matthews_corrcoef(y_test, y_pred))

        # Accumulate feature importance
        if hasattr(best_est, "feature_importances_"):
            importance_accum += best_est.feature_importances_

        print(f"    Fold {fold_idx}: "
              f"Acc={results['accuracy'][-1]:.3f}  "
              f"F1={results['f1'][-1]:.3f}  "
              f"ROC={results['roc_auc'][-1]:.3f}  "
              f"PR={results['pr_auc'][-1]:.3f}  "
              f"MCC={results['mcc'][-1]:+.3f}")

    results["feature_importance"] = importance_accum / N_SPLITS

    return results


# =========================================================
# MAIN
# =========================================================

print("=" * 70)
print("  ML BENCHMARK: 4 Matrices x 2 Models")
print("=" * 70)

all_results = {}   # key: "A_Combo_Raw__rf"

for exp_name, exp_path in EXPERIMENTS.items():
    if not os.path.exists(exp_path):
        print(f"\n  !  File not found, skipping: {exp_path}")
        continue

    X, y, feat_cols = load_matrix(exp_path, exp_name)

    for model_type, param_grid in [("rf", RF_GRID), ("xgb", XGB_GRID)]:
        label = f"{exp_name}__{model_type}"
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_name}  |  Model: {model_type.upper()}")
        print(f"{'='*60}")

        res = evaluate_model(exp_name, X, y, model_type, param_grid)
        all_results[label] = {
            "exp": exp_name,
            "model": model_type,
            "metrics": res,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_pos": int((y==1).sum()),
            "feat_cols": feat_cols,
        }


# =========================================================
# SUMMARY TABLE
# =========================================================

print("\n\n")
print("=" * 90)
print("  SUMMARY: Mean +/- Std across 5 folds")
print("=" * 90)

header = (f"{'Experiment':<28} {'Model':<6} "
          f"{'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10} {'MCC':>10}")
print(header)
print("-" * 100)

summary_rows = []

for label, data in all_results.items():
    m = data["metrics"]
    row = {
        "Experiment": data["exp"].replace("  ", " "),
        "Model":      data["model"].upper(),
        "Accuracy":   f"{np.mean(m['accuracy']):.3f} +- {np.std(m['accuracy']):.3f}",
        "F1":         f"{np.mean(m['f1']):.3f} +- {np.std(m['f1']):.3f}",
        "ROC_AUC":    f"{np.mean(m['roc_auc']):.3f} +- {np.std(m['roc_auc']):.3f}",
        "PR_AUC":     f"{np.mean(m['pr_auc']):.3f} +- {np.std(m['pr_auc']):.3f}",
        "MCC":        f"{np.mean(m['mcc']):.3f} +- {np.std(m['mcc']):.3f}",
        "Acc_mean":   np.mean(m["accuracy"]),
        "F1_mean":    np.mean(m["f1"]),
        "ROC_mean":   np.mean(m["roc_auc"]),
        "PR_mean":    np.mean(m["pr_auc"]),
        "MCC_mean":   np.mean(m["mcc"]),
        "n_features": data["n_features"],
    }
    summary_rows.append(row)

for r in summary_rows:
    print(f"{r['Experiment']:<28} {r['Model']:<6} "
          f"{r['Accuracy']:>10} {r['F1']:>10} {r['ROC_AUC']:>10} {r['PR_AUC']:>10} {r['MCC']:>10}")


# =========================================================
# RANKING
# =========================================================

print("\n" + "=" * 90)
print("  RANKING (by ROC-AUC)")
print("=" * 90)

for metric in ["ROC_mean", "PR_mean", "F1_mean", "MCC_mean", "Acc_mean"]:
    sorted_rows = sorted(summary_rows, key=lambda r: r[metric], reverse=True)
    print(f"\n  By {metric.replace('_mean','')}:")
    for i, r in enumerate(sorted_rows):
        bar = "#" * (3 - i) if i < 3 else " "
        print(f"    {i+1}. {r['Experiment']:<26} [{r['Model']}]  "
              f"ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  F1={r['F1']}  MCC={r['MCC']}")


# =========================================================
# SAVE CSV
# =========================================================

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n  Saved summary: {summary_path}")


# =========================================================
# TOP FEATURES per best model
# =========================================================

print("\n" + "=" * 90)
print("  TOP FEATURES (best model per experiment)")
print("=" * 90)

best_key = max(all_results.keys(), key=lambda k: np.mean(all_results[k]["metrics"]["roc_auc"]))
best_data = all_results[best_key]
best_imp = best_data["metrics"]["feature_importance"]
best_cols = best_data["feat_cols"]

top_idx = np.argsort(best_imp)[::-1][:20]
print(f"\n  Best model: {best_key}")
print(f"  {'Rank':<5} {'Feature':<20} {'Importance':>12}")
print(f"  {'─'*5} {'─'*20} {'─'*12}")
for rank, idx in enumerate(top_idx):
    print(f"  {rank+1:<5} {best_cols[idx]:<20} {best_imp[idx]:>12.6f}")

print(f"\n{'='*90}")
print(f"  DONE. Results in: {OUTPUT_DIR}")
print(f"{'='*90}")
