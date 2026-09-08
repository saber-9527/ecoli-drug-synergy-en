# %%
# =========================================================
# Feature Count Sweep — Publication-Quality Implementation  (v2)
#
# Methodology:
#   - Nested CV: feature ranking + C tuning inside each outer fold
#   - Outer: RepeatedStratifiedKFold (3 repeats × 5 folds)
#   - Inner: GridSearchCV for L1 C optimisation
#   - Fold-specific ranking → no information leakage
#   - Extended K range → observe plateau / saturation
#   - Automatic plateau detection → minimal parsimonious K
#   - Rank-variance tracking per feature
#   - Mean ± Std across 15 evaluations
#
# Outputs:
#   - feature_sweep.png            (AUC / PR / MCC vs K, with ±1σ band)
#   - feature_sweep_results.csv    (per-K summary)
#   - ranked_features.csv          (feature ranking + stability + rank variance)
# =========================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = Path(
    r"../data/non_additive_matrix\non_additive_normalized.csv"
)
OUTPUT_DIR = Path(
    r"../data/best_model_analysis_enhanced"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Extended sweep schedule ----
# Dense at low K (biology-driven), coarse at high K (saturation check)
K_VALUES_RAW = (
    list(range(2, 21, 2))     +     #   2, 4, ..., 20        (10 pts)
    list(range(25, 101, 5))   +     #  25, 30, ..., 100      (16 pts)
    list(range(120, 501, 20)) +     # 120, 140, ..., 500     (20 pts)
    list(range(600, 2501, 200))     # 600, 800, ..., 2400    (10 pts)
)
# Total: ~56 K-values  (capped at n_features at runtime)

# CV settings
N_REPEATS      = 3
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_SEED    = 42

# L1 C search space (log-spaced, wide range)
C_GRID = np.logspace(-3, 2, 21)   # 0.001 → 100, 21 pts

# Plateau detection: "within 0.5 std of best AUC"
#   "0.5std" → stricter, prevents overestimation when CV variance is high (small n, large p)
#   "1std"   → more permissive, may overestimate plateau region
#   "1pct"   → fixed 1 % of best AUC, independent of σ
PLATEAU_THRESHOLD = "0.5std"


# =========================================================
# 1. DATA LOADING
# =========================================================

def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and clean feature matrix.  Returns X, y, feature_names."""
    df = pd.read_csv(path)
    print(f"[LOAD]  {path.name}  |  shape={df.shape}")

    y = df["label"].values.astype(int)
    id_cols = ["drug_pair", "label"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df[feat_cols].values.astype(np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(X, 0.1, axis=0)
    hi = np.percentile(X, 99.9, axis=0)
    X = np.clip(X, lo, hi)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"       Pos=1: {n_pos} ({n_pos/len(y)*100:.1f}%)  "
          f"Neg=0: {n_neg} ({n_neg/len(y)*100:.1f}%)  "
          f"Features: {len(feat_cols)}")
    return X, y, feat_cols


# =========================================================
# 2. NESTED CROSS-VALIDATION SWEEP
# =========================================================

def nested_sweep(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    outer_cv: RepeatedStratifiedKFold,
    c_grid: np.ndarray,
    random_state: int = 42,
) -> dict:
    """
    Nested CV sweep — zero information leakage.

    Outer loop:
      1. Hold out a test fold.
      2. On the training fold:
         a. Scale.
         b. Inner GridSearchCV for L1 C.
         c. Fit best L1 → rank features by |coef|.
         d. For each K: select top-K, fit L2, evaluate on test fold.
      3. Record per-K metrics, per-feature rank, and per-feature |coef|.

    Returns
    -------
    dict with keys:
      "by_k"          : dict[int, list[dict]]
      "feature_freq"  : np.ndarray (n_features,)
      "feature_coef"  : np.ndarray (n_features,)
      "feature_ranks" : np.ndarray (n_features, n_folds)  — rank per fold
      "fold_best_c"   : list[float]
    """
    n_features  = X.shape[1]
    n_folds     = outer_cv.get_n_splits()

    feature_coef  = np.zeros(n_features)
    feature_ranks = np.full((n_features, n_folds), np.nan)

    by_k: dict[int, list[dict]] = {k: [] for k in k_values}
    fold_best_c: list[float] = []

    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=random_state
    )

    print(f"\n[NESTED CV]  {n_folds} outer folds  ×  "
          f"{N_INNER_SPLITS} inner folds  ×  {len(k_values)} K-values")
    print(f"             C grid: {len(c_grid)} values  "
          f"({c_grid[0]:.4f} → {c_grid[-1]:.1f})")
    print(f"             K range: {min(k_values)} → {max(k_values)}")
    print(f"{'─'*70}")

    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ----- a. Scale -----
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ----- b. Inner GridSearchCV for C -----
        base_l1 = LogisticRegression(
            penalty="l1", solver="saga", max_iter=10000,
            random_state=random_state, n_jobs=-1,
        )
        grid = GridSearchCV(
            base_l1, param_grid={"C": c_grid},
            scoring="roc_auc", cv=inner_cv, n_jobs=-1, verbose=0,
        )
        grid.fit(X_train_s, y_train)
        best_c = grid.best_params_["C"]
        fold_best_c.append(best_c)

        # ----- c. Fit L1 with best C → ranking -----
        l1_model = LogisticRegression(
            penalty="l1", solver="saga", C=best_c,
            max_iter=10000, random_state=random_state, n_jobs=-1,
        )
        l1_model.fit(X_train_s, y_train)
        coef = l1_model.coef_.flatten()
        rank_order = np.argsort(np.abs(coef))[::-1]

        # Track per-feature stats
        feature_coef += np.abs(coef)
        for rank_pos, feat_idx in enumerate(rank_order):
            feature_ranks[feat_idx, fold_idx - 1] = rank_pos + 1   # 1-based

        # ----- d. For each K: select top-K, fit, evaluate -----
        max_k_needed = min(max(k_values), X_train_s.shape[1])
        selected = rank_order[:max_k_needed]

        for k in k_values:
            if k > X_train_s.shape[1]:
                break   # beyond available features; skip larger K
            top_k_idx = selected[:k]
            X_tr_k = X_train_s[:, top_k_idx]
            X_te_k = X_test_s[:, top_k_idx]

            model_k = LogisticRegression(
                penalty="l2", solver="lbfgs", C=1.0,
                max_iter=5000, random_state=random_state,
            )
            model_k.fit(X_tr_k, y_train)
            y_proba = model_k.predict_proba(X_te_k)[:, 1]

            by_k[k].append({
                "auc": roc_auc_score(y_test, y_proba),
                "pr":  average_precision_score(y_test, y_proba),
                "mcc": matthews_corrcoef(y_test, (y_proba >= 0.5).astype(int)),
            })

        # Progress
        k_best_local = max(by_k, key=lambda kk: np.mean([v["auc"] for v in by_k[kk]]))
        auc_local = np.mean([v["auc"] for v in by_k[k_best_local]])
        print(f"  Fold {fold_idx:>2d}/{n_folds}  |  best_C={best_c:.6f}  "
              f"best_K={k_best_local:>4d}  AUC={auc_local:.4f}")

    feature_coef /= n_folds
    print(f"{'─'*70}")

    return {
        "by_k":          by_k,
        "feature_coef":  feature_coef,
        "feature_ranks": feature_ranks,
        "fold_best_c":   fold_best_c,
        "n_folds":       n_folds,
    }


# =========================================================
# 3. AGGREGATE RESULTS
# =========================================================

def aggregate_results(by_k: dict) -> pd.DataFrame:
    """Compute mean ± std for each K.  Returns sorted DataFrame."""
    rows = []
    for k in sorted(by_k.keys()):
        vals = by_k[k]
        aucs = [v["auc"] for v in vals]
        prs  = [v["pr"]  for v in vals]
        mccs = [v["mcc"] for v in vals]
        rows.append({
            "K":        k,
            "AUC_mean": np.mean(aucs),
            "AUC_std":  np.std(aucs,  ddof=1),
            "PR_mean":  np.mean(prs),
            "PR_std":   np.std(prs,   ddof=1),
            "MCC_mean": np.mean(mccs),
            "MCC_std":  np.std(mccs,  ddof=1),
            "n_evals":  len(vals),
        })
    return pd.DataFrame(rows).sort_values("K").reset_index(drop=True)


# =========================================================
# 4. PLATEAU & TREND ANALYSIS
# =========================================================

def find_minimal_stable_k(
    results_df: pd.DataFrame,
    best_k: int,
    threshold: str = "1std",
) -> int:
    """
    Smallest K whose AUC is within a tolerance of the global best.

    threshold="0.5std" → within 0.5 standard deviation  (recommended for small n)
    threshold="1std"   → within 1 standard deviation
    threshold="1pct"   → within 1% of best AUC
    """
    best_row = results_df[results_df["K"] == best_k].iloc[0]
    best_auc  = best_row["AUC_mean"]
    best_std  = best_row["AUC_std"]

    if threshold == "0.5std":
        target = best_auc - 0.5 * best_std
    elif threshold == "1std":
        target = best_auc - best_std
    else:
        target = best_auc * 0.99

    candidates = results_df[results_df["AUC_mean"] >= target]
    return int(candidates["K"].min())


def analyze_trend(results_df: pd.DataFrame, cutoff: int = 200) -> str:
    """
    Classify performance trend beyond `cutoff`.

    Returns one of:  "rising"  |  "plateau"  |  "declining"
    """
    below = results_df[results_df["K"] <= cutoff]
    above = results_df[results_df["K"] > cutoff]

    if above.empty:
        return "plateau"   # no data beyond cutoff

    auc_below_max = below["AUC_mean"].max()
    auc_above_max = above["AUC_mean"].max()
    delta = auc_above_max - auc_below_max

    # Use the typical std at high K as noise floor
    noise = above["AUC_std"].median() if len(above) > 1 else 0.01

    if delta > noise:
        return "rising"
    elif delta < -noise:
        return "declining"
    else:
        return "plateau"


# =========================================================
# 5. PUBLICATION-QUALITY PLOT
# =========================================================

def plot_sweep(
    results_df: pd.DataFrame,
    best_k: int,
    stable_k: int,
    trend: str,
    output_dir: Path,
):
    """Publication-quality figure: AUC / PR / MCC vs K, ±1σ, plateau annotations."""
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "DejaVu Sans"],
        "font.size":        11,
        "axes.titlesize":   14,
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ns = results_df["K"].values
    auc_mean = results_df["AUC_mean"].values
    auc_std  = results_df["AUC_std"].values
    pr_mean  = results_df["PR_mean"].values
    pr_std   = results_df["PR_std"].values
    mcc_mean = results_df["MCC_mean"].values
    mcc_std  = results_df["MCC_std"].values

    C_AUC = "#377EB8"
    C_PR  = "#4DAF4A"
    C_MCC = "#E41A1C"

    # ---- AUC ----
    ax1.plot(ns, auc_mean, "o-", color=C_AUC, lw=2.0, markersize=5.5, label="ROC-AUC")
    ax1.fill_between(ns,
                     np.clip(auc_mean - auc_std, 0, 1),
                     np.clip(auc_mean + auc_std, 0, 1),
                     color=C_AUC, alpha=0.15)
    # ---- PR ----
    ax1.plot(ns, pr_mean, "s-", color=C_PR, lw=2.0, markersize=5.5, label="PR-AUC")
    ax1.fill_between(ns,
                     np.clip(pr_mean - pr_std, 0, 1),
                     np.clip(pr_mean + pr_std, 0, 1),
                     color=C_PR, alpha=0.15)

    # ---- Best K (global maximum) ----
    best_row = results_df[results_df["K"] == best_k].iloc[0]
    ax1.axvline(best_k, color="grey", lw=1.2, ls="--", zorder=0)
    ax1.annotate(
        f"Best K={best_k}\nAUC={best_row['AUC_mean']:.3f}±{best_row['AUC_std']:.3f}",
        xy=(best_k, best_row["AUC_mean"]),
        xytext=(best_k + 35, best_row["AUC_mean"] - 0.12),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.1),
        fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.85),
    )

    # ---- Minimal Stable K (parsimonious) ----
    if stable_k != best_k:
        stable_row = results_df[results_df["K"] == stable_k].iloc[0]
        ax1.axvline(stable_k, color="#FF7F00", lw=1.2, ls="-.", zorder=0)
        ax1.annotate(
            f"Minimal Stable K={stable_k}\nAUC={stable_row['AUC_mean']:.3f}±{stable_row['AUC_std']:.3f}",
            xy=(stable_k, stable_row["AUC_mean"]),
            xytext=(stable_k - 90, stable_row["AUC_mean"] + 0.10),
            arrowprops=dict(arrowstyle="->", color="#FF7F00", lw=1.1),
            fontsize=9, color="#CC6600",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#FF7F00", alpha=0.85),
        )

    # ---- Plateau zone shading (stable_k → best_k) ----
    ax1.axvspan(stable_k, best_k, alpha=0.06, color="green", zorder=0)
    ax1.text((stable_k + best_k) / 2, 0.10, "plateau\nregion",
             ha="center", fontsize=8.5, color="green", style="italic")

    # ---- Trend annotation ----
    trend_label = {"rising": "↑  Rising", "plateau": "→  Plateau", "declining": "↓  Declining"}
    ax1.text(0.985, 0.04, f"Trend (K>200): {trend_label.get(trend, trend)}",
             transform=ax1.transAxes, ha="right", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    # ---- MCC (secondary axis) ----
    ax2 = ax1.twinx()
    ax2.plot(ns, mcc_mean, "D-", color=C_MCC, lw=1.6, markersize=4.5, label="MCC")
    ax2.fill_between(ns,
                     np.clip(mcc_mean - mcc_std, -1, 1),
                     np.clip(mcc_mean + mcc_std, -1, 1),
                     color=C_MCC, alpha=0.12)
    ax2.set_ylabel("MCC", fontsize=12, color=C_MCC)
    ax2.tick_params(axis="y", labelcolor=C_MCC)

    # ---- Axes ----
    ax1.set_xlabel("Number of Top Features (K)", fontsize=12)
    ax1.set_ylabel("AUC", fontsize=12)
    ax1.set_title(
        "Feature Count Sweep — Non-additive Normalized\n"
        f"(Nested CV: {N_REPEATS}×{N_OUTER_SPLITS}-fold outer, "
        f"{N_INNER_SPLITS}-fold inner L1 selection)",
        fontsize=13, fontweight="bold",
    )
    ax1.set_xlim(0, ns[-1] + 5)
    ax1.set_ylim(0.0, 1.0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, alpha=0.25, which="major")
    ax1.grid(True, alpha=0.12, which="minor")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower right", framealpha=0.9, fontsize=10)

    fig.tight_layout()
    path = output_dir / "feature_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVE]  {path}")


# =========================================================
# 6. EXPORT RANKED FEATURES
# =========================================================

def export_features(
    feature_names: list[str],
    feature_coef: np.ndarray,
    feature_ranks: np.ndarray,
    best_k: int,
    stable_k: int,
    output_dir: Path,
):
    """
    Export ranked feature CSV with:
      - Stability@K       (% folds where rank ≤ K)
      - Stability@BestK   (% folds where rank ≤ best_K)
      - Mean_Abs_Coef     (mean |coefficient| across folds)
      - Mean_Rank          (mean rank position across folds)
      - Rank_Variance      (variance of rank position across folds)
    """
    n_features = len(feature_names)

    # Mean rank & rank variance
    mean_rank = np.nanmean(feature_ranks, axis=1)
    rank_var  = np.zeros(n_features)
    for i in range(n_features):
        ranks_i = feature_ranks[i, :]
        valid = ranks_i[~np.isnan(ranks_i)]
        if len(valid) >= 2:
            rank_var[i] = np.var(valid, ddof=1)
        else:
            rank_var[i] = np.nan

    # Stability: fraction of folds where feature rank ≤ K
    stab_col_best   = f"Stability_K{best_k}"
    stab_col_stable = f"Stability_K{stable_k}"

    stability_at_best   = np.nanmean(feature_ranks <= best_k,   axis=1)
    stability_at_stable = np.nanmean(feature_ranks <= stable_k, axis=1)
    stability_at_50     = np.nanmean(feature_ranks <= 50,       axis=1).round(3)

    df = pd.DataFrame({
        "Feature":           feature_names,
        "Mean_Abs_Coef":     feature_coef.round(6),
        "Mean_Rank":         mean_rank.round(1),
        "Rank_Variance":     rank_var.round(1),
        stab_col_best:       stability_at_best.round(3),
        stab_col_stable:     stability_at_stable.round(3),
        "Stability_K50":     stability_at_50,
    })
    df = df.sort_values(
        [stab_col_best, "Mean_Abs_Coef"], ascending=[False, False]
    ).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    path = output_dir / "ranked_features.csv"
    df.to_csv(path, index=False)
    print(f"[SAVE]  {path}")

    n_at_best   = int((stability_at_best   == 1.0).sum())
    n_at_stable = int((stability_at_stable == 1.0).sum())
    n_at_50     = int((stability_at_50     == 1.0).sum())
    print(f"         features always in top-{best_k}   : {n_at_best} / {n_features}")
    print(f"         features always in top-{stable_k}  : {n_at_stable} / {n_features}")
    print(f"         features always in top-50    : {n_at_50} / {n_features}")

    # Print top 20
    print(f"\n  {'Rank':<5} {'Feature':<20} "
          f"{'Stab@'+str(best_k):>12} {'Stab@'+str(stable_k):>12} "
          f"{'|Coef|':>10} {'Rank_Var':>9}")
    print(f"  {'─'*5} {'─'*20} {'─'*12} {'─'*12} {'─'*10} {'─'*9}")
    for _, r in df.head(20).iterrows():
        rv = f"{r['Rank_Variance']:>9.1f}" if not np.isnan(r['Rank_Variance']) else "      N/A"
        print(f"  {r['Rank']:<5} {r['Feature']:<20} "
              f"{r[stab_col_best]:>11.0%} "
              f"{r[stab_col_stable]:>11.0%} "
              f"{r['Mean_Abs_Coef']:>10.6f} {rv}")


# =========================================================
# 7. MAIN
# =========================================================

def main():
    print("=" * 70)
    print("  FEATURE COUNT SWEEP v2  —  Publication-Quality Nested CV")
    print("=" * 70)

    # --- Load ---
    X, y, feat_names = load_data(DATA_PATH)
    n_features = X.shape[1]

    # Cap K-values at actual feature count
    k_values = sorted(set(k for k in K_VALUES_RAW if k <= n_features))
    skipped = len(K_VALUES_RAW) - len(k_values)
    if skipped:
        print(f"[INFO]  {skipped} K-values beyond n_features={n_features} were dropped")

    # --- CV setup ---
    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_OUTER_SPLITS, n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    n_total_folds = outer_cv.get_n_splits()

    # --- Nested sweep ---
    result = nested_sweep(X, y, k_values, outer_cv, C_GRID, random_state=RANDOM_SEED)

    by_k          = result["by_k"]
    feature_coef  = result["feature_coef"]
    feature_ranks = result["feature_ranks"]
    fold_best_c   = result["fold_best_c"]

    # --- Aggregate ---
    results_df  = aggregate_results(by_k)
    best_idx    = results_df["AUC_mean"].idxmax()
    best_row    = results_df.loc[best_idx]
    best_k      = int(best_row["K"])
    stable_k    = find_minimal_stable_k(results_df, best_k, threshold=PLATEAU_THRESHOLD)
    trend       = analyze_trend(results_df, cutoff=200)

    # --- Performance summary ---
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"  Best K                 : {best_k}")
    print(f"    ROC-AUC              : {best_row['AUC_mean']:.4f}  ± {best_row['AUC_std']:.4f}")
    print(f"    PR-AUC               : {best_row['PR_mean']:.4f}   ± {best_row['PR_std']:.4f}")
    print(f"    MCC                  : {best_row['MCC_mean']:.4f}  ± {best_row['MCC_std']:.4f}")
    print(f"  Minimal Stable K       : {stable_k}  "
          f"(within {'0.5σ' if PLATEAU_THRESHOLD == '0.5std' else '1σ' if PLATEAU_THRESHOLD == '1std' else '1%'} of best AUC)")
    if stable_k != best_k:
        stable_row = results_df[results_df["K"] == stable_k].iloc[0]
        print(f"    AUC at stable K      : {stable_row['AUC_mean']:.4f}  ± {stable_row['AUC_std']:.4f}")
        print(f"    Features saved       : {best_k - stable_k}  "
              f"({(best_k - stable_k) / best_k * 100:.1f}% reduction)")
    else:
        print(f"    (stable K = best K — no parsimonious alternative found)")
    print(f"  Trend beyond K=200     : {trend}")
    print(f"  L1 C (median)          : {np.median(fold_best_c):.4f}  "
          f"[{np.min(fold_best_c):.4f}, {np.max(fold_best_c):.4f}]")
    print(f"  CV evaluations         : {n_total_folds}")

    # --- Export CSVs ---
    csv_path = OUTPUT_DIR / "feature_sweep_results.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[SAVE]  {csv_path}")

    # Save per-fold best C for downstream model training
    c_df = pd.DataFrame({"fold": range(1, len(fold_best_c) + 1), "best_C": fold_best_c})
    c_path = OUTPUT_DIR / "sweep_fold_best_c.csv"
    c_df.to_csv(c_path, index=False, float_format="%.6f")
    print(f"[SAVE]  {c_path}")

    # --- Plot ---
    plot_sweep(results_df, best_k, stable_k, trend, OUTPUT_DIR)

    # --- Ranked features ---
    export_features(
        feat_names, feature_coef, feature_ranks,
        best_k=best_k, stable_k=stable_k, output_dir=OUTPUT_DIR,
    )

    print(f"\n{'='*70}")
    print(f"  **  Sweep complete.  Results in: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
