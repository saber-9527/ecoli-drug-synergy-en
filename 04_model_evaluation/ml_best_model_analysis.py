# %%
# =========================================================
# ENHANCED BEST MODEL ANALYSIS: Logistic Regression on Non-additive Normalized
# - Pipeline + Cross-validated SHAP
# - Typical Sample Waterfall
# - Reaction -> Pathway Top Plot
# - SHAP Dependence & Top Features Correlation
# =========================================================

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             PrecisionRecallDisplay, RocCurveDisplay,
                             classification_report)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# ================== CONFIG ==================
DATA_PATH = Path(r"../data/non_additive_matrix\non_additive_normalized.csv")
OUTPUT_DIR = Path(r"../data/best_model_analysis_enhanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "best_logreg_pipeline_enhanced.joblib"

RANDOM_SEED = 42
N_SPLITS = 5

# Optional: Reaction -> Pathway mapping CSV
PATHWAY_MAP = Path(r"../data/reaction_to_pathway.csv")  # columns: reaction,pathway

# ================== LOAD DATA ==================
df = pd.read_csv(DATA_PATH)
y = df["label"].values.astype(int)
id_cols = ["drug_pair", "label"]
feat_cols = [c for c in df.columns if c not in id_cols]
X = df[feat_cols].values.astype(np.float64)

# Vectorized cleaning
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
lo_limits = np.percentile(X, 0.1, axis=0)
hi_limits = np.percentile(X, 99.9, axis=0)
X = np.clip(X, lo_limits, hi_limits)

print(f"Data shape: {X.shape}, Pos: {(y==1).sum()}, Neg: {(y==0).sum()}")

# ================== PIPELINE + GRID SEARCH ==================
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=RANDOM_SEED, n_jobs=-1, max_iter=10000)
)
lr_name = pipeline.steps[-1][0]

param_grid = [
    {
        f"{lr_name}__penalty": ["l1", "l2"],
        f"{lr_name}__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        f"{lr_name}__solver": ["saga"],
        f"{lr_name}__class_weight": ["balanced", None],
        f"{lr_name}__l1_ratio": [None],
    },
    {
        f"{lr_name}__penalty": ["elasticnet"],
        f"{lr_name}__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        f"{lr_name}__solver": ["saga"],
        f"{lr_name}__class_weight": ["balanced", None],
        f"{lr_name}__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
    },
]

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1, verbose=1)
search.fit(X, y)
best_pipeline = search.best_estimator_
joblib.dump(best_pipeline, MODEL_PATH)

# ================== CROSS-VALIDATED PREDICTIONS ==================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
y_preds = cross_val_predict(best_pipeline, X, y, cv=skf, method="predict")
y_probas = cross_val_predict(best_pipeline, X, y, cv=skf, method="predict_proba")[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y, y_preds))
print("F1:", f1_score(y, y_preds))
print("ROC-AUC:", roc_auc_score(y, y_probas))
print("PR-AUC:", average_precision_score(y, y_probas))
print("MCC:", matthews_corrcoef(y, y_preds))

# ================== CONFUSION MATRIX ==================
fig, ax = plt.subplots(figsize=(6,5))
cm = confusion_matrix(y, y_preds)
ConfusionMatrixDisplay(cm, display_labels=["Non-Synergy","Synergy"]).plot(ax=ax, cmap="YlOrRd", values_format="d")
ax.set_title("Confusion Matrix")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ================== SHAP EXPLAINER ON OUTER CV ==================
scaler = best_pipeline.named_steps['standardscaler']
lr_model = best_pipeline.named_steps[lr_name]
X_scaled = scaler.transform(X)

# Background sampling for speed
background = shap.utils.sample(X_scaled, 100, random_state=RANDOM_SEED)
explainer = shap.LinearExplainer(lr_model, background, feature_perturbation="interventional")

# Outer CV SHAP summary
shap_values_list = []
for train_idx, test_idx in skf.split(X_scaled, y):
    X_test_fold = X_scaled[test_idx]
    shap_fold = explainer(X_test_fold)
    shap_values_list.append(shap_fold.values)
shap_values_all = np.vstack(shap_values_list)

# SHAP summary plots
for plot_type, filename in [("bar","shap_bar_top20.png"), (None,"shap_beeswarm_top20.png")]:
    fig, ax = plt.subplots(figsize=(10,8))
    shap.summary_plot(shap_values_all, X_scaled, feature_names=feat_cols, plot_type=plot_type, max_display=20, show=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ================== TYPICAL SAMPLE WATERFALL ==================
def plot_typical_waterfall(y_class, y_probas, X_scaled, explainer, feat_cols, filename):
    idx_pool = np.where(y==y_class)[0]
    if len(idx_pool)==0: return
    # choose sample closest to probability 0.5
    prob_diff = np.abs(y_probas[idx_pool] - 0.5)
    sample_idx = idx_pool[np.argmin(prob_diff)]
    shap_single = explainer(X_scaled[sample_idx:sample_idx+1])
    fig, ax = plt.subplots(figsize=(10,6))
    shap.plots.waterfall(shap_single[0], max_display=15, show=False)
    label = "Synergy" if y_class==1 else "Non-Synergy"
    ax.set_title(f"SHAP Waterfall -- True {label} (pred={y_probas[sample_idx]:.3f})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

plot_typical_waterfall(1, y_probas, X_scaled, explainer, feat_cols, "shap_waterfall_typical_synergy.png")
plot_typical_waterfall(0, y_probas, X_scaled, explainer, feat_cols, "shap_waterfall_typical_nonsynergy.png")

# ================== REACTION -> PATHWAY TOP PLOT ==================
if PATHWAY_MAP.exists():
    map_df = pd.read_csv(PATHWAY_MAP)
    shap_df = pd.DataFrame(shap_values_all, columns=feat_cols)
    shap_df_mean = shap_df.abs().mean().reset_index()
    shap_df_mean.columns = ["reaction","mean_abs_shap"]
    merged = shap_df_mean.merge(map_df, left_on="reaction", right_on="reaction")
    top_pathways = merged.groupby("pathway")["mean_abs_shap"].sum().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10,8))
    top_pathways.plot.barh(ax=ax, color="#377EB8")
    ax.set_xlabel("Aggregated |SHAP|")
    ax.set_title("Top 20 Pathways by SHAP")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top20_pathways_shap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ================== SHAP DEPENDENCE PLOT + TOP 10 CORRELATION ==================
print("Plotting SHAP Dependence Plot and Top 10 Features Correlation Heatmap...")

# Top 1 feature dependence plot
top_feature_idx = np.argmax(np.abs(shap_values_all).mean(axis=0))
top_feature_name = feat_cols[top_feature_idx]
second_feature_idx = np.argsort(np.abs(shap_values_all).mean(axis=0))[::-1][1]
second_feature_name = feat_cols[second_feature_idx]

fig, ax = plt.subplots(figsize=(8,6))
shap.dependence_plot(
    top_feature_name,
    shap_values_all,
    X_scaled,
    feature_names=feat_cols,
    interaction_index=second_feature_name,
    ax=ax,
    show=False
)
ax.set_title(f"SHAP Dependence: {top_feature_name}\n(colored by {second_feature_name})", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "shap_dependence_top1.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Top 10 feature correlation heatmap
top_10_idx = np.argsort(np.abs(lr_model.coef_.flatten()))[::-1][:10]
top_10_names = [feat_cols[i] for i in top_10_idx]
corr_matrix = pd.DataFrame(X_scaled, columns=feat_cols)[top_10_names].corr()

fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(top_10_names)))
ax.set_yticks(range(len(top_10_names)))
ax.set_xticklabels(top_10_names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(top_10_names, fontsize=9)
ax.set_title("Correlation Heatmap of Top 10 Features", fontsize=12, fontweight="bold")
for i in range(len(top_10_names)):
    for j in range(len(top_10_names)):
        ax.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr_matrix.iloc[i,j])>0.5 else "black", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "top10_features_correlation.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ================== ROC + PR CURVES ==================
fig, axes = plt.subplots(1,2,figsize=(14,6))
RocCurveDisplay.from_predictions(y, y_probas, ax=axes[0], color="#377EB8", lw=2.2)
axes[0].plot([0,1],[0,1],"k--", lw=0.7, alpha=0.4)
axes[0].set_title(f"ROC Curve (AUC={roc_auc_score(y,y_probas):.4f})", fontsize=12, fontweight="bold")
PrecisionRecallDisplay.from_predictions(y,y_probas, ax=axes[1], color="#4DAF4A", lw=2.2)
axes[1].set_title(f"PR Curve (AP={average_precision_score(y,y_probas):.4f})", fontsize=12, fontweight="bold")
fig.suptitle("Best Model: Logistic Regression on Non-additive Normalized", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_pr_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ================== CALIBRATION CURVE ==================
fig, ax = plt.subplots(figsize=(7,6))
prob_true, prob_pred = calibration_curve(y, y_probas, n_bins=10, strategy="quantile")
ax.plot(prob_pred, prob_true, "s-", color="#FF7F00", lw=2.2, markersize=8, label="Logistic Regression")
ax.plot([0,1],[0,1],"k--", lw=0.8, alpha=0.5, label="Perfectly calibrated")
ax.set_xlim(-0.02,1.02)
ax.set_ylim(-0.02,1.02)
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Fraction of Positives", fontsize=11)
ax.set_title("Calibration Curve -- Logistic Regression", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\n{'='*70}\n  **  Enhanced Best Model Analysis Complete\n  Results saved in: {OUTPUT_DIR}\n{'='*70}")
