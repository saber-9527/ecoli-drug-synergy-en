# %%
# =========================================================
# ML BENCHMARK EXTENDED: RF / XGBoost / LogisticRegression / SVM / LightGBM
# on 4 flux feature matrices  +  ROC curve plots
#
# Model A: Combo Raw
# Model B: Combo Normalized
# Model C: Non-additive Raw
# Model D: Non-additive Normalized
#
# Metrics: Accuracy, F1, ROC-AUC, PR-AUC, MCC
# CV: 5-fold stratified, hyperparameter grid search
# =========================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, roc_curve)
import xgboost as xgb
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

# Input matrices
MATRIX_DIR = r"../data/continuous_bounded_moma_full-new"
NA_DIR     = r"../data/non_additive_matrix"

EXPERIMENTS = {
    "A  Combo Raw":              os.path.join(MATRIX_DIR, "flux_feature_matrix_raw.csv"),
    "B  Combo Normalized":       os.path.join(MATRIX_DIR, "flux_feature_matrix_normalized.csv"),
    "C  Non-additive Raw":       os.path.join(NA_DIR, "non_additive_raw.csv"),
    "D  Non-additive Normalized": os.path.join(NA_DIR, "non_additive_normalized.csv"),
}

OUTPUT_DIR = r"../data/ml_results_extended"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CV settings
N_SPLITS    = 5
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
    "scale_pos_weight": [1, 2, 3],
}

# FIX: LogReg requires a list of dicts because l1_ratio must be None for l1/l2 penalties
LOGREG_GRID = [
    {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["saga"],
        "max_iter": [5000],
        "class_weight": ["balanced", None],
        "l1_ratio": [None]
    },
    {
        "penalty": ["elasticnet"],
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["saga"],
        "max_iter": [5000],
        "class_weight": ["balanced", None],
        "l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9]
    }
]

SVM_GRID = {
    "C":            [0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma":        ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    "class_weight": ["balanced", None],
    "kernel":       ["rbf"],
    "probability":  [True],
    "max_iter":     [5000],
}

LGB_GRID = {
    "n_estimators":     [100, 200, 500],
    "max_depth":        [3, 5, 7, 9, -1],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 1.0],
    "bagging_freq":     [1, 5],  # FIX: Required for subsample to take effect in LightGBM
    "colsample_bytree": [0.7, 0.8, 1.0],
    "num_leaves":       [15, 31, 63, 127],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [0, 0.1, 1.0],
    "class_weight":     ["balanced", None],
    "min_child_samples":[5, 10, 20],
    "verbose":          [-1],
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

    y = df["label"].values.astype(int)
    id_cols = ["drug_pair", "label"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df[feat_cols].values.astype(np.float64)

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
    """
    Run stratified CV with grid/randomized search.
    Returns per-fold metrics, best params, feature importance, and ROC data.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    results = {
        "accuracy": [], "f1": [], "roc_auc": [], "pr_auc": [], "mcc": [],
        "best_params": [], "feature_importance": None,
        "roc_data": [],
    }

    n_features = X.shape[1]
    importance_accum = np.zeros(n_features)

    # --- Build base model ---
    if model_type == "rf":
        base_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    elif model_type == "xgb":
        # FIX: Removed deprecated use_label_encoder
        base_model = xgb.XGBClassifier(
            random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss"
        )
    elif model_type == "logreg":
        base_model = LogisticRegression(random_state=RANDOM_SEED, n_jobs=-1)
    elif model_type == "svm":
        base_model = SVC(random_state=RANDOM_SEED)
    elif model_type == "lgb":
        base_model = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Determine search strategy ---
    # FIX: Handle list of dictionaries for Logistic Regression
    if isinstance(param_grid, list):
        n_combos = int(sum(np.prod([len(v) for v in g.values()]) for g in param_grid))
    else:
        n_combos = int(np.prod([len(v) for v in param_grid.values()]))

    n_iter = min(60, n_combos)
    search_mode = "RandomizedSearchCV" if n_combos > 60 else "GridSearchCV"

    print(f"\n  [{model_type.upper()}] {search_mode}  (max {n_iter} / {n_combos} combos)")

    fold_idx = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Scaling for distance / regularisation-based models ---
        if model_type in ("logreg", "svm"):
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
        else:
            X_train_s, X_test_s = X_train, X_test

        # Inner CV for HP tuning (3-fold)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=RANDOM_SEED + fold_idx)

        if search_mode == "GridSearchCV":
            search = GridSearchCV(
                base_model, param_grid,
                scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0,
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, scoring="roc_auc", cv=inner_cv,
                n_jobs=-1, verbose=0, random_state=RANDOM_SEED,
            )

        search.fit(X_train_s, y_train)
        best_est = search.best_estimator_
        results["best_params"].append(search.best_params_)

        # Predict on held-out fold
        y_pred  = best_est.predict(X_test_s)
        y_proba = best_est.predict_proba(X_test_s)[:, 1]

        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["f1"].append(f1_score(y_test, y_pred))
        results["roc_auc"].append(roc_auc_score(y_test, y_proba))
        results["pr_auc"].append(average_precision_score(y_test, y_proba))
        results["mcc"].append(matthews_corrcoef(y_test, y_pred))

        # Store ROC curve data for this fold
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results["roc_data"].append({
            "fpr": fpr,
            "tpr": tpr,
            "auc": results["roc_auc"][-1],
        })

        # Accumulate feature importance (tree-based models only)
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
# ROC CURVE PLOTTING
# =========================================================

def plot_roc_curves(all_results, output_dir=OUTPUT_DIR):
    """
    Generate ROC curve comparison plots.
    """
    MODEL_COLORS = {
        "rf":     "#E41A1C",
        "xgb":    "#377EB8",
        "logreg": "#4DAF4A",
        "svm":    "#984EA3",
        "lgb":    "#FF7F00",
    }
    MODEL_LABELS = {
        "rf":     "Random Forest",
        "xgb":    "XGBoost",
        "logreg": "Logistic Regression",
        "svm":    "SVM (RBF)",
        "lgb":    "LightGBM",
    }

    # Group results by experiment
    exp_results = {}
    for label, data in all_results.items():
        exp_name = data["exp"]
        exp_results.setdefault(exp_name, []).append((label, data))

    # -------- (A) One figure per experiment --------
    for exp_name, entries in exp_results.items():
        fig, ax = plt.subplots(figsize=(8, 7))

        for label, data in entries:
            model_type = data["model"]
            roc_list  = data["metrics"]["roc_data"]
            color     = MODEL_COLORS.get(model_type, "#888888")
            disp_name = MODEL_LABELS.get(model_type, model_type.upper())

            # Interpolate all folds onto common FPR grid
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            aucs = [r["auc"] for r in roc_list]
            for r in roc_list:
                tpr_interp = np.interp(mean_fpr, r["fpr"], r["tpr"])
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
            tprs = np.array(tprs)
            mean_tpr = tprs.mean(axis=0)
            std_tpr  = tprs.std(axis=0)
            mean_auc = np.mean(aucs)
            std_auc  = np.std(aucs)

            ax.plot(mean_fpr, mean_tpr, color=color, lw=2.2,
                    label=f"{disp_name}  (AUC={mean_auc:.3f} +- {std_auc:.3f})")
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=color, alpha=0.12)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        title = exp_name.replace("  ", " - ")
        ax.set_title(f"ROC Curves  --  {title}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        fig.tight_layout()
        safe_name = exp_name.strip().replace("  ", "_").replace(" ", "_")
        fpath = os.path.join(output_dir, f"roc_{safe_name}.png")
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ROC plot   : {fpath}")

    # -------- (B) Combined 2x2 subplot figure --------
    n_exps = len(exp_results)
    if n_exps == 0:
        return

    n_cols = min(2, n_exps)
    n_rows = int(np.ceil(n_exps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6.5 * n_rows))
    if n_exps == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for ax, (exp_name, entries) in zip(axes, exp_results.items()):
        for label, data in entries:
            model_type = data["model"]
            roc_list  = data["metrics"]["roc_data"]
            color     = MODEL_COLORS.get(model_type, "#888888")
            disp_name = MODEL_LABELS.get(model_type, model_type.upper())

            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            aucs = [r["auc"] for r in roc_list]
            for r in roc_list:
                tpr_interp = np.interp(mean_fpr, r["fpr"], r["tpr"])
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr  = np.std(tprs, axis=0)
            mean_auc = np.mean(aucs)

            ax.plot(mean_fpr, mean_tpr, color=color, lw=2.0,
                    label=f"{disp_name}  ({mean_auc:.3f})")
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=color, alpha=0.10)

        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.35)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(exp_name.replace("  ", " - "), fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    for j in range(n_exps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("ROC Curve Comparison -- All Experiments", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    combined_path = os.path.join(output_dir, "roc_all_experiments.png")
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined    : {combined_path}")


# =========================================================
# MAIN
# =========================================================

print("=" * 70)
print("  ML BENCHMARK EXTENDED: 4 Matrices x 5 Models  (+ ROC curves)")
print("=" * 70)

MODEL_LIST = [
    ("rf",     RF_GRID),
    ("xgb",    XGB_GRID),
    ("logreg", LOGREG_GRID),
    ("svm",    SVM_GRID),
    ("lgb",    LGB_GRID),
]

all_results = {}

for exp_name, exp_path in EXPERIMENTS.items():
    if not os.path.exists(exp_path):
        print(f"\n  !  File not found, skipping: {exp_path}")
        continue

    X, y, feat_cols = load_matrix(exp_path, exp_name)

    for model_type, param_grid in MODEL_LIST:
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
print("=" * 120)
print("  SUMMARY: Mean +/- Std across 5 folds")
print("=" * 120)

header = (f"{'Experiment':<28} {'Model':<8} "
          f"{'Accuracy':>12} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10} {'MCC':>10}")
print(header)
print("-" * 120)

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
    print(f"{r['Experiment']:<28} {r['Model']:<8} "
          f"{r['Accuracy']:>12} {r['F1']:>10} {r['ROC_AUC']:>10} {r['PR_AUC']:>10} {r['MCC']:>10}")


# =========================================================
# RANKING
# =========================================================

print("\n" + "=" * 120)
print("  RANKING (by ROC-AUC)")
print("=" * 120)

for metric in ["ROC_mean", "PR_mean", "F1_mean", "MCC_mean", "Acc_mean"]:
    sorted_rows = sorted(summary_rows, key=lambda r: r[metric], reverse=True)
    print(f"\n  By {metric.replace('_mean','')}:")
    for i, r in enumerate(sorted_rows):
        bar = "#" * (3 - i) if i < 3 else " "
        print(f"    {i+1}. {r['Experiment']:<26} [{r['Model']:<6}]  "
              f"ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  F1={r['F1']}  MCC={r['MCC']}")


# =========================================================
# SAVE CSV
# =========================================================

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary_extended.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n  Saved summary: {summary_path}")


# =========================================================
# TOP FEATURES per best model
# =========================================================

print("\n" + "=" * 120)
print("  TOP FEATURES (best model per experiment)")
print("=" * 120)

best_key = max(all_results.keys(), key=lambda k: np.mean(all_results[k]["metrics"]["roc_auc"]))
best_data = all_results[best_key]
best_imp = best_data["metrics"]["feature_importance"]
best_cols = best_data["feat_cols"]

if best_imp is not None and best_imp.sum() > 0:
    top_idx = np.argsort(best_imp)[::-1][:20]
    print(f"\n  Best model: {best_key}")
    print(f"  {'Rank':<5} {'Feature':<20} {'Importance':>12}")
    print(f"  {'─'*5} {'─'*20} {'─'*12}")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:<5} {best_cols[idx]:<20} {best_imp[idx]:>12.6f}")
else:
    print(f"\n  Best model: {best_key}  (feature importance not available for this model type)")


# =========================================================
# ROC CURVE PLOTS
# =========================================================

print("\n" + "=" * 120)
print("  GENERATING ROC CURVE PLOTS")
print("=" * 120)

plot_roc_curves(all_results, output_dir=OUTPUT_DIR)


print(f"\n{'='*120}")
print(f"  **  Benchmark Extended finished successfully.")
print(f"  Results in: {OUTPUT_DIR}")
print(f"{'='*120}")
