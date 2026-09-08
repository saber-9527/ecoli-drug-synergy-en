# %%
# Single-feature diagnostic: top 1 feature vs full model
# =========================================================
import warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

warnings.filterwarnings('ignore')

PATH = r"../data/non_additive_matrix\non_additive_normalized.csv"
df = pd.read_csv(PATH)
y = df['label'].values.astype(int)
feat_cols = [c for c in df.columns if c not in ('drug_pair','label')]
X = df[feat_cols].values.astype(np.float64)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X = np.clip(X, np.percentile(X,0.1,axis=0), np.percentile(X,99.9,axis=0))

# ---- Step 1: find top 1 feature by abs coefficient ----
pipe = make_pipeline(StandardScaler(),
    LogisticRegression(penalty='l1',solver='saga',C=1.0,max_iter=10000,random_state=42,n_jobs=-1))
pipe.fit(X, y)
coef = pipe.named_steps['logisticregression'].coef_.flatten()
top_idx = np.argmax(np.abs(coef))
top_name = feat_cols[top_idx]
print(f'Top 1 feature: {top_name}')
print(f'  coef = {coef[top_idx]:.6f}')
print(f'  Non-zero coefs (L1): {(np.abs(coef)>1e-8).sum()} / {len(coef)}')

# ---- Step 2: single-feature 5-fold CV ----
X_single = X[:, top_idx:top_idx+1]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe_single = make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=5000,random_state=42))
y_proba_single = cross_val_predict(pipe_single, X_single, y, cv=skf, method='predict_proba')[:,1]
print(f'\nSingle-feature ({top_name}):')
print(f'  ROC-AUC = {roc_auc_score(y, y_proba_single):.4f}')
print(f'  PR-AUC  = {average_precision_score(y, y_proba_single):.4f}')
print(f'  MCC     = {matthews_corrcoef(y, (y_proba_single>=0.5).astype(int)):.4f}')

# ---- Step 3: top 5 / top 10 / top 20 / top 50 feature ROC ----
for n_top in [2, 5, 10, 20, 50, 100]:
    idx = np.argsort(np.abs(coef))[::-1][:n_top]
    X_sub = X[:, idx]
    pipe_sub = make_pipeline(StandardScaler(),
        LogisticRegression(penalty='l1',solver='saga',C=1.0,max_iter=10000,random_state=42,n_jobs=-1))
    y_proba_sub = cross_val_predict(pipe_sub, X_sub, y, cv=skf, method='predict_proba')[:,1]
    auc = roc_auc_score(y, y_proba_sub)
    print(f'  Top {n_top:>3d} features  ROC-AUC = {auc:.4f}')

# ---- Step 4: full model ----
pipe_full = make_pipeline(StandardScaler(),
    LogisticRegression(penalty='l1',solver='saga',C=1.0,max_iter=10000,random_state=42,n_jobs=-1))
y_proba_full = cross_val_predict(pipe_full, X, y, cv=skf, method='predict_proba')[:,1]
auc_full = roc_auc_score(y, y_proba_full)
print(f'  Full 970 features  ROC-AUC = {auc_full:.4f}')

# ---- Step 5: plot ----
fig, ax = plt.subplots(figsize=(9,8))
RocCurveDisplay.from_predictions(y, y_proba_single, ax=ax, color='#E41A1C', lw=2.2,
    label=f'Single feature ({top_name})  AUC={roc_auc_score(y,y_proba_single):.3f}')
RocCurveDisplay.from_predictions(y, y_proba_full,  ax=ax, color='#377EB8', lw=2.2,
    label=f'Full model (970 features)     AUC={auc_full:.3f}')
ax.plot([0,1],[0,1],'k--',lw=0.7,alpha=0.4)
ax.set_title(f'Single vs Full Feature ROC Comparison\nNon-additive Normalized', fontsize=13, fontweight='bold')
ax.legend(loc='lower right',fontsize=11)
ax.grid(True,alpha=0.3)
fig.tight_layout()
out = r"../data/best_model_analysis_enhanced\single_vs_full_roc.png"
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')
print(f'delta_AUC (full - single) = {auc_full - roc_auc_score(y,y_proba_single):+.4f}')
print('Done.')
