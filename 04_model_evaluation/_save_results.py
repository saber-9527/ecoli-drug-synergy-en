"""Re-run sweep and save CSVs — standalone to avoid conda inline issues."""
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(r"../data/best_model_analysis_enhanced")

from feature_sweep import (
    load_data, nested_sweep, aggregate_results, export_features,
    DATA_PATH, K_VALUES_RAW, N_REPEATS, N_OUTER_SPLITS, C_GRID, RANDOM_SEED,
)
from sklearn.model_selection import RepeatedStratifiedKFold

X, y, feat_names = load_data(DATA_PATH)
k_values = sorted(set(k for k in K_VALUES_RAW if k <= X.shape[1]))

outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
result = nested_sweep(X, y, k_values, outer_cv, C_GRID, random_state=RANDOM_SEED)
results_df = aggregate_results(result["by_k"])
results_df.to_csv(OUTPUT_DIR / "feature_sweep_results.csv", index=False, float_format="%.6f")
print(f"SAVED: feature_sweep_results.csv ({len(results_df)} rows)")

export_features(feat_names, result["feature_freq"], result["feature_coef"],
                result["feature_ranks"],
                top_k=int(results_df.loc[results_df["AUC_mean"].idxmax(), "K"]),
                n_total_folds=result["n_folds"], output_dir=OUTPUT_DIR)
print("DONE")
