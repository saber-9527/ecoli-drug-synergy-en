# Use old 411 as training, new 74 (same drug pairs, different study) as held-out test
# 74 pairs have overlap with old data - different labels from different experimental platform
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb

# ── Load data ──
merged = pd.read_csv(r"../data/merged_matrix\merged_non_additive_normalized.csv")

# Split: old rows have score=0 or NaN; new rows have non-zero score
new_mask = merged["score"].notna() & (merged["score"].abs() > 0.001)
old_mask = ~new_mask

old_df = merged[old_mask].copy()
new_df = merged[new_mask].copy()

print(f"Old train: {len(old_df)}  (syn=1: {old_df['label'].sum():.0f}, non-syn=0: {(old_df['label']==0).sum():.0f})")
print(f"New test:  {len(new_df)}  (score>0 (syn): {new_df['label'].sum():.0f}, score<=0 (non): {(new_df['label']==0).sum():.0f})")
print(f"New score range: {new_df['score'].min():.3f} ~ {new_df['score'].max():.3f}")

# Inter-study agreement
agree = 0
disagree = 0
for p in new_df["drug_pair"].unique():
    ol = old_df[old_df["drug_pair"] == p]
    nl = new_df[new_df["drug_pair"] == p]
    if len(ol) > 0 and len(nl) > 0:
        if ol.iloc[0]["label"] == nl.iloc[0]["label"]:
            agree += 1
        else:
            disagree += 1
print(f"Inter-study label agreement: {agree}/{agree+disagree} = {agree/(agree+disagree)*100:.1f}%")
print(f"(Note: low agreement = different studies, different platforms, expected)")

# ── Prepare train/test ──
id_cols = ["drug_pair", "label", "score"]
old_feat = [c for c in old_df.columns if c not in id_cols]
new_feat = [c for c in new_df.columns if c not in id_cols]

common_feat = sorted(set(old_feat) & set(new_feat))
print(f"Common features: {len(common_feat)}")

X_train = old_df[common_feat].values.astype(np.float64)
y_train = old_df["label"].values.astype(int)
X_test  = new_df[common_feat].values.astype(np.float64)
y_test  = new_df["label"].values.astype(int)
scores  = new_df["score"].values.astype(np.float64)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ── Train on old, eval on new ──
print("\n" + "="*80)
print("  EXTERNAL VALIDATION: Train on Old 411, Test on New 74")
print("="*80)

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

results = []

for name, model_class, grid, use_scaling in [
    ("RF",  RandomForestClassifier(random_state=42, n_jobs=-1), RF_GRID, False),
    ("XGB", xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"), XGB_GRID, False),
    ("LR",  LogisticRegression(random_state=42, max_iter=5000), LR_GRID, True),
]:
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

    # HP tuning on train only (nested CV for fair HP selection)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    if use_scaling:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
    else:
        X_tr_s, X_te_s = X_train, X_test

    if n_combos <= 40:
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(model_class, grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(model_class, grid, n_iter=30, scoring="roc_auc", cv=inner_cv, n_jobs=-1, random_state=42)

    search.fit(X_tr_s, y_train)
    best = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # Predict on test
    y_pred  = best.predict(X_te_s)
    y_proba = best.predict_proba(X_te_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_proba)
    pr   = average_precision_score(y_test, y_proba)
    mcc  = matthews_corrcoef(y_test, y_pred)

    # Spearman correlation with continuous score
    sp_rho, sp_p = spearmanr(scores, y_proba)

    print(f"\n  Classification (binary label):")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    ROC-AUC:  {roc:.4f}")
    print(f"    PR-AUC:   {pr:.4f}")
    print(f"    MCC:      {mcc:+.4f}")

    print(f"\n  Regression (continuous score):")
    print(f"    Spearman rho = {sp_rho:.4f}  (p={sp_p:.4f})")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"           Pred=0  Pred=1")
    print(f"  True=0    {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"  True=1    {cm[1,0]:>5}    {cm[1,1]:>5}")

    results.append({
        "Model": name,
        "Accuracy": acc, "F1": f1, "ROC_AUC": roc, "PR_AUC": pr, "MCC": mcc,
        "Spearman_rho": sp_rho, "Spearman_p": sp_p,
        "Best_Params": str(search.best_params_),
    })

# ── Summary ──
print(f"\n{'='*80}")
print(f"  SUMMARY: External Validation on New 74 Pairs")
print(f"{'='*80}")
print(f"{'Model':<8} {'Acc':>8} {'F1':>8} {'ROC':>8} {'PR':>8} {'MCC':>8} {'Spearman ρ':>12}")
print("-"*68)
for r in results:
    print(f"{r['Model']:<8} {r['Accuracy']:>8.4f} {r['F1']:>8.4f} {r['ROC_AUC']:>8.4f} {r['PR_AUC']:>8.4f} {r['MCC']:>8.4f} {r['Spearman_rho']:>10.4f}")

print(f"\n{'='*80}")
print(f"  DONE.")
print(f"{'='*80}")
