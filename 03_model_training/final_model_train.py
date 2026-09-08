# %%
# =========================================================
# FINAL MODEL TRAINING — Two-Stage: Feature Selection -> HP Optimisation
#
# Methodology:
#   Stage 1 (feature_sweep.py) — already completed:
#     Feature ranking via nested CV L1.  The feature subsets are FINALISED:
#       Compact model: top 120 features  (minimal stable)
#       Full model:    top 800 features  (best performance)
#     These K values are HARD-CODED — they do not change.
#
#   Stage 2 (this script) — independent L2 hyperparameter optimisation:
#     After fixing the feature subset, each model finds its OWN optimal
#     L2 regularisation strength via GridSearchCV.  This avoids mixing
#     L1-derived hyperparameters with L2 models while allowing each
#     feature subset to achieve its best possible performance.
#
#     PART A — Honest Nested Evaluation (unbiased performance estimate):
#       For each outer fold:
#         Train -> Inner GridSearchCV (L2 C) on train ONLY
#              -> Select best C for this fold
#              -> Train model with that C on train
#              -> Predict on held-out test fold
#       GridSearchCV is nested INSIDE each fold -> zero information leakage.
#       Performance metrics come from this nested evaluation.
#
#     PART B — Final Deployed Model:
#       GridSearchCV on ALL data -> single best C for deployment.
#       Train final Pipeline on ALL data with that C -> save joblib.
#       This C is used ONLY for the deployable model, never for evaluation.
#
# Outputs:
#   pipeline_compact.joblib    — final trained pipeline (120 features)
#   pipeline_full.joblib       — final trained pipeline (800 features)
#   feature_list_compact.csv   — ordered feature names (120)
#   feature_list_full.csv      — ordered feature names (800)
#   model_comparison.csv       — side-by-side performance summary (nested CV)
# =========================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
)

warnings.filterwarnings("ignore")

# =========================================================
# CUSTOM TRANSFORMER  —  leakage-free percentile clipping
# =========================================================

class ClipOutliers(BaseEstimator, TransformerMixin):
    """Fit percentiles on training data; apply to any data.  Safe inside Pipeline."""

    def __init__(self, lower_pct: float = 0.1, upper_pct: float = 99.9):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lo_ = np.percentile(X, self.lower_pct, axis=0)
        self.hi_ = np.percentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, self.lo_, self.hi_)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = Path(
    r"../data/non_additive_matrix\non_additive_normalized.csv"
)
RANK_PATH = Path(
    r"../data/best_model_analysis_enhanced\ranked_features.csv"
)
OUTPUT_DIR = Path(r"../data/final_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED    = 42
N_OUTER_SPLITS = 5
N_REPEATS      = 10                      # 50 honest evaluations
N_INNER_SPLITS = 5                       # GridSearchCV inner folds

USE_BALANCED = False                     # True -> class_weight='balanced'
THRESHOLD    = 0.5                       # lower -> higher recall

# ---- FIXED feature subset sizes (from feature_sweep.py — DO NOT CHANGE) ----
# These are FINALISED after the nested-CV sweep.
# Compact = minimal stable K  (within 0.5σ of best AUC) -> parsimonious
# Full    = best performance K (global AUC maximum)       -> maximal
COMPACT_K = 120
FULL_K    = 800

# ---- L2 C search grid (independent of L1 sweep) ----
# Log-spaced: 0.001 -> 100, 21 points — wide enough to cover
# the optimal region for both small and large feature sets.
C_GRID = np.logspace(-3, 2, 21)

# =========================================================
# 1. LOAD RAW DATA
# =========================================================

print("=" * 70)
print("  FINAL MODEL TRAINING  —  Two-Stage Methodology")
print("  Stage 1 (done):   Feature selection via nested-CV L1 sweep")
print("  Stage 2 (now):    Independent L2 GridSearchCV per feature subset")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n[LOAD]  {DATA_PATH.name}  |  shape={df.shape}")

y = df["label"].values.astype(int)
feat_cols_all = [c for c in df.columns if c not in ("drug_pair", "label")]
X = df[feat_cols_all].values.astype(np.float64)

n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
print(f"       Pos=1: {n_pos} ({n_pos/len(y)*100:.1f}%)  "
      f"Neg=0: {n_neg} ({n_neg/len(y)*100:.1f}%)  "
      f"Features: {len(feat_cols_all)}")

# =========================================================
# 2. SELECT TOP FEATURES  (fixed subsets — no re-ranking)
# =========================================================

rank_df = pd.read_csv(RANK_PATH)

# ---- Validate feature names ----
rank_set = set(rank_df["Feature"].tolist())
data_set = set(feat_cols_all)
missing = rank_set - data_set
if missing:
    raise KeyError(
        f"{len(missing)} ranked features NOT found in data columns.\n"
        f"First 10 missing: {sorted(list(missing))[:10]}\n"
        f"This usually means the ranked_features.csv was produced from "
        f"a different dataset. Check DATA_PATH and RANK_PATH."
    )
print(f"[VALIDATE]  {len(rank_set)} ranked features all present in data  OK")

# ---- Stability columns (match sweep export) ----
stab_col_compact = f"Stability_K{COMPACT_K}"
stab_col_full    = f"Stability_K{FULL_K}"

for col_name, k_val, model_name in [
    (stab_col_full,    FULL_K,    "full"),
    (stab_col_compact, COMPACT_K, "compact"),
]:
    if col_name not in rank_df.columns:
        available = [c for c in rank_df.columns if c.startswith("Stability_K")]
        raise KeyError(
            f"Column '{col_name}' not found in ranked_features.csv.\n"
            f"Available Stability_K columns: {available}\n"
            f"Model '{model_name}' expects K={k_val}, but "
            f"ranked_features.csv was exported with a different K.\n"
            f"Re-run feature_sweep.py to regenerate ranked_features.csv "
            f"with the current K values."
        )

# ----
# Feature ranking is fixed — sort by stability@K then mean |coef|.
# This ranking NEVER changes; only the top-K cutoff differs per model.
# ----
def get_top_features(rank_df, k_value, stab_col):
    """Return ordered feature list using stability @ K + |coef|."""
    ordered = rank_df.sort_values(
        [stab_col, "Mean_Abs_Coef"], ascending=[False, False])
    return ordered["Feature"].tolist()[:k_value]

features = {
    "compact": get_top_features(rank_df, COMPACT_K, stab_col_compact),
    "full":    get_top_features(rank_df, FULL_K,    stab_col_full),
}
indices = {
    name: [feat_cols_all.index(f) for f in feats]
    for name, feats in features.items()
}
X_sub = {name: X[:, idx] for name, idx in indices.items()}

# ---- Verify Compact in Full (parsimonious model is nested in maximal) ----
overlap = len(set(features["compact"]) & set(features["full"]))
assert overlap == len(features["compact"]), \
    f"Compact in Full  FAIL  ({overlap} / {len(features['compact'])})"
print(f"[FEATURES]  Compact in Full  OK  ({overlap}/{len(features['compact'])})")

print(f"            Compact model : {COMPACT_K} features  (minimal stable)")
print(f"            Full model    : {FULL_K} features  (best performance)")

# ---- Export feature lists ----
for name, feats in features.items():
    feat_df = pd.DataFrame({"Rank": range(1, len(feats) + 1), "Feature": feats})
    feat_df.to_csv(OUTPUT_DIR / f"feature_list_{name}.csv", index=False)
    print(f"[SAVE]  {OUTPUT_DIR / f'feature_list_{name}.csv'}")

# =========================================================
# 3. PART A — HONEST NESTED EVALUATION
# =========================================================
# For each outer fold:
#   train -> Inner GridSearchCV (L2 C, 5-fold) on train only
#        -> best C for this fold
#        -> train model with that C on train
#        -> predict on held-out test fold
#
# GridSearchCV is nested INSIDE each outer fold — the C search
# NEVER sees the test data.  This is the honest estimate.
#
# A separate GridSearchCV on ALL data (Part B) is used ONLY
# for the final deployable model, not for performance reporting.
# =========================================================

class_weight = "balanced" if USE_BALANCED else None

outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED,
)
n_total = outer_cv.get_n_splits()

# Inner CV for per-fold GridSearchCV
inner_cv = StratifiedKFold(
    n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED,
)

print(f"\n{'-'*70}")
print(f"  PART A: Honest Nested Evaluation")
print(f"  Outer: {N_REPEATS}×{N_OUTER_SPLITS}-fold RepeatedStratifiedKFold ({n_total} evals)")
print(f"  Inner: GridSearchCV (L2 C, {N_INNER_SPLITS}-fold, scoring=roc_auc)")
print(f"  C grid: {len(C_GRID)} values  ({C_GRID[0]:.4f} -> {C_GRID[-1]:.1f})")
print(f"  GridSearchCV runs on train ONLY — zero leakage")
print(f"{'-'*70}")

# Store results
results = {}                      # per-model aggregated metrics
fold_best_c = {}                  # per-model list of best C per outer fold

for name in ("compact", "full"):
    k = COMPACT_K if name == "compact" else FULL_K
    X_m = X_sub[name]

    y_accum = np.zeros(len(y))        # sum of predicted probabilities
    n_accum = np.zeros(len(y), dtype=int)  # how many times each sample was test
    c_per_fold = []                   # track best C from each outer fold

    fold_num = 0
    for train_idx, test_idx in outer_cv.split(X_m, y):
        fold_num += 1
        X_tr, X_te = X_m[train_idx], X_m[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # ----- Step 1: Inner GridSearchCV on TRAIN ONLY -----
        pipe_search = Pipeline([
            ("clip",   ClipOutliers(lower_pct=0.1, upper_pct=99.9)),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l2", solver="lbfgs",
                class_weight=class_weight, max_iter=5000,
                random_state=RANDOM_SEED,
            )),
        ])

        grid = GridSearchCV(
            pipe_search,
            param_grid={"clf__C": C_GRID},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_tr, y_tr)
        best_c_fold = float(grid.best_params_["clf__C"])
        c_per_fold.append(best_c_fold)

        # ----- Step 2: Train model with best C on train, predict test -----
        fold_pipe = Pipeline([
            ("clip",   ClipOutliers(lower_pct=0.1, upper_pct=99.9)),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l2", solver="lbfgs", C=best_c_fold,
                class_weight=class_weight, max_iter=5000,
                random_state=RANDOM_SEED,
            )),
        ])
        fold_pipe.fit(X_tr, y_tr)
        y_accum[test_idx] += fold_pipe.predict_proba(X_te)[:, 1]
        n_accum[test_idx] += 1

        # Progress every 10 folds
        if fold_num % 10 == 0:
            print(f"  [{name:<7}]  fold {fold_num:>2d}/{n_total}  "
                  f"C={best_c_fold:.6f}")

    # ---- Aggregate predictions across repeated folds ----
    y_proba = y_accum / n_accum
    y_pred = (y_proba >= THRESHOLD).astype(int)

    results[name] = {
        "Accuracy": accuracy_score(y, y_pred),
        "F1":       f1_score(y, y_pred),
        "ROC_AUC":  roc_auc_score(y, y_proba),
        "PR_AUC":   average_precision_score(y, y_proba),
        "MCC":      matthews_corrcoef(y, y_pred),
    }
    fold_best_c[name] = np.array(c_per_fold)

    print(f"  [{name:<7}]  -- Honest Nested CV Results --")
    print(f"            "
          f"ROC={results[name]['ROC_AUC']:.4f}  "
          f"PR={results[name]['PR_AUC']:.4f}  "
          f"F1={results[name]['F1']:.4f}  "
          f"MCC={results[name]['MCC']:+.4f}")
    print(f"            Per-fold C: "
          f"median={np.median(c_per_fold):.4f}  "
          f"[{np.min(c_per_fold):.4f}, {np.max(c_per_fold):.4f}]  "
          f"±{np.std(c_per_fold, ddof=1):.4f}")

    # ---- Save per-fold C values ----
    c_fold_df = pd.DataFrame({
        "fold": np.arange(1, len(c_per_fold) + 1),
        "best_C": c_per_fold,
    })
    c_fold_path = OUTPUT_DIR / f"fold_best_c_{name}.csv"
    c_fold_df.to_csv(c_fold_path, index=False, float_format="%.6f")
    print(f"[SAVE]  {c_fold_path}")

    # ---- Save per-sample nested CV predictions ----
    pred_df = pd.DataFrame({
        "true_label":       y,
        "y_proba":          y_proba,
        "y_pred":           y_pred,
        "n_evaluations":    n_accum,
        "y_proba_raw_sum":  y_accum,       # sum before division
    })
    pred_path = OUTPUT_DIR / f"nested_cv_predictions_{name}.csv"
    pred_df.to_csv(pred_path, index=False, float_format="%.6f")
    print(f"[SAVE]  {pred_path}")

print(f"{'-'*70}")
print(f"  NOTE: Above metrics are UNBIASED — C was chosen per-fold on train only.")

# =========================================================
# 4. PART B — FINAL DEPLOYED MODEL
# =========================================================
# GridSearchCV on ALL data to select the single best C for deployment.
# This C is NOT used in any performance evaluation — it is purely
# for the final joblib pipeline that goes into production.
#
# The deployed C is reported alongside the per-fold C distribution
# from Part A so you can verify it falls within the expected range.

print(f"\n{'-'*70}")
print(f"  PART B: Final Deployed Model")
print(f"  GridSearchCV on ALL data  ({N_INNER_SPLITS}-fold, scoring=roc_auc)")
print(f"  These C values are for DEPLOYMENT only — not for evaluation.")
print(f"{'-'*70}")

deployed_c = {}
deployed_cv_auc = {}

for name in ("compact", "full"):
    k = COMPACT_K if name == "compact" else FULL_K
    X_m = X_sub[name]

    pipe = Pipeline([
        ("clip",   ClipOutliers(lower_pct=0.1, upper_pct=99.9)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="l2", solver="lbfgs",
            class_weight=class_weight, max_iter=5000,
            random_state=RANDOM_SEED,
        )),
    ])

    grid_final = GridSearchCV(
        pipe,
        param_grid={"clf__C": C_GRID},
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_final.fit(X_m, y)

    best_c_deploy = float(grid_final.best_params_["clf__C"])
    deployed_c[name] = best_c_deploy
    deployed_cv_auc[name] = grid_final.best_score_

    # Compare deployed C against per-fold distribution
    per_fold_c = fold_best_c[name]
    print(f"  [{name:<7}  K={k:>4d}]")
    print(f"            Deployed C           = {best_c_deploy:.6f}")
    print(f"            Deploy CV AUC        = {grid_final.best_score_:.4f}")
    print(f"            Per-fold C (Part A):   "
          f"median={np.median(per_fold_c):.4f}  "
          f"[{np.min(per_fold_c):.4f}, {np.max(per_fold_c):.4f}]")

    # ---- Train final pipeline on ALL data ----
    final_pipe = Pipeline([
        ("clip",   ClipOutliers(lower_pct=0.1, upper_pct=99.9)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="l2", solver="lbfgs", C=best_c_deploy,
            class_weight=class_weight, max_iter=5000,
            random_state=RANDOM_SEED,
        )),
    ])
    final_pipe.fit(X_m, y)

    path = OUTPUT_DIR / f"pipeline_{name}.joblib"
    joblib.dump(final_pipe, path)
    print(f"            Saved -> {path}")

print(f"{'-'*70}")
print(f"  NOTE: Deployed C may differ from per-fold median C.")
print(f"        This is expected — deployed C is optimised on ALL data,")
print(f"        while Part A C varies per fold (less training data).")
print(f"        Use Part A metrics for reporting, Part B pipeline for deployment.")

# =========================================================
# 5. COMPARISON TABLE
# =========================================================

print(f"\n{'='*80}")
print(f"  COMPARISON  (Nested CV: {n_total} evaluations, C optimised per fold)")
print(f"{'='*80}")

header = (f"  {'Metric':<14} "
          f"{'Compact (K=' + str(COMPACT_K) + ')':>28}   "
          f"{'Full (K=' + str(FULL_K) + ')':>28}")
print(header)
print(f"  {'-'*14} {'-'*28}   {'-'*28}")

for metric in ["ROC_AUC", "PR_AUC", "F1", "Accuracy", "MCC"]:
    c_val = results["compact"][metric]
    f_val = results["full"][metric]
    c_str = f"{c_val:.4f}"
    f_str = f"{f_val:.4f}"
    if c_val > f_val:
        c_str = f"* {c_str}"
    elif f_val > c_val:
        f_str = f"* {f_str}"
    print(f"  {metric:<14} {c_str:>28}   {f_str:>28}")

print(f"\n  * = better score  (metrics from nested CV — unbiased)")

# ---- Show C comparison ----
print(f"\n  {'C (deployed)':<14} "
      f"{deployed_c['compact']:>28.6f}   "
      f"{deployed_c['full']:>28.6f}")
print(f"  {'C (per-fold median)':<14} "
      f"{np.median(fold_best_c['compact']):>28.6f}   "
      f"{np.median(fold_best_c['full']):>28.6f}")

# =========================================================
# 6. SAVE MODEL COMPARISON CSV
# =========================================================

rows = []
for name in ("compact", "full"):
    r = results[name]
    k = COMPACT_K if name == "compact" else FULL_K
    per_fold_c = fold_best_c[name]
    rows.append({
        "Model":                f"{'Compact' if name == 'compact' else 'Full'} "
                                f"({'Minimal Stable' if name == 'compact' else 'Best Performance'})",
        "K":                    k,
        "C_deployed":           deployed_c[name],
        "C_per_fold_median":    np.median(per_fold_c),
        "C_per_fold_min":       np.min(per_fold_c),
        "C_per_fold_max":       np.max(per_fold_c),
        "C_per_fold_std":       np.std(per_fold_c, ddof=1),
        "C_source":             "PART A: per-fold inner GridSearchCV (train only). "
                                "PART B: GridSearchCV on all data (deployment only).",
        "C_search_range":       f"[{C_GRID[0]:.4f}, {C_GRID[-1]:.1f}]",
        "Penalty":              "l2",
        "Solver":               "lbfgs",
        "Class_Weight":         class_weight,
        "Threshold":            THRESHOLD,
        "N_features":           X_sub[name].shape[1],
        "Accuracy":             r["Accuracy"],
        "F1":                   r["F1"],
        "ROC_AUC":              r["ROC_AUC"],
        "PR_AUC":               r["PR_AUC"],
        "MCC":                  r["MCC"],
        "CV_repeats":           N_REPEATS,
        "CV_splits":            N_OUTER_SPLITS,
        "Inner_CV_splits":      N_INNER_SPLITS,
        "Total_outer_folds":    n_total,
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, float_format="%.6f")
print(f"\n[SAVE]  {OUTPUT_DIR / 'model_comparison.csv'}")

# =========================================================
# 7. DONE
# =========================================================

print(f"\n{'='*70}")
print(f"  **  Two-stage training complete.")
print(f"  Output: {OUTPUT_DIR}")
print(f"    pipeline_compact.joblib          — {COMPACT_K} features  "
      f"(C={deployed_c['compact']:.6f})")
print(f"    pipeline_full.joblib             — {FULL_K} features  "
      f"(C={deployed_c['full']:.6f})")
print(f"    feature_list_compact.csv         — top {COMPACT_K} feature names")
print(f"    feature_list_full.csv            — top {FULL_K} feature names")
print(f"    fold_best_c_compact.csv          — per-fold C (50 folds)")
print(f"    fold_best_c_full.csv             — per-fold C (50 folds)")
print(f"    nested_cv_predictions_compact.csv — per-sample proba + pred")
print(f"    nested_cv_predictions_full.csv    — per-sample proba + pred")
print(f"    model_comparison.csv             — side-by-side summary")
print(f"{'='*70}")
print(f"\n  Methodology summary:")
print(f"    Stage 1: Feature ranking via nested-CV L1 (feature_sweep.py)")
print(f"    Stage 2: Two-part independent L2 optimisation")
print(f"      PART A: Nested CV — GridSearchCV INSIDE each outer fold")
print(f"              -> unbiased performance estimates for reporting")
print(f"      PART B: GridSearchCV on ALL data")
print(f"              -> single best C for the deployed joblib pipeline")
print(f"    Each model (Compact/Full) uses its OWN optimal L2 C.")
print(f"{'='*70}")
