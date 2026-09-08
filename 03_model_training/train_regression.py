# ============================================================
# Train regression model on 262 scored combos
# Same preprocessing as classification: zero-var + freq filter + L1 row norm + StandardScaler
# ============================================================
import pandas as pd, numpy as np, json, os, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error

print("1. Load data...")
raw = pd.read_csv(r"../data/non_additive_matrix\non_additive_raw.csv")
raw["pair_norm"] = raw["drug_pair"].str.replace("-","__").str.upper()
feat_cols = [c for c in raw.columns if c not in ["drug_pair","label","pair_norm"]]

xl = pd.read_excel(r"../data/1-ecoli.xlsx","Supp Table 2",header=1)
xl = xl.iloc[2:].copy()
xl.columns = ["idx","D1","D2","Cons","Class","ConsStr","ST","PA","SBW","SiAi1","SST","SST14","SPAO1","SPA14","ScBW","SciAi1","ScST","ScST14","ScPAO1","ScPA14"]+[f"X{i}" for i in range(20,28)]
sc = xl[["D1","D2","ScBW"]].dropna()
sc["ScBW"] = pd.to_numeric(sc["ScBW"],errors="coerce")
sc = sc.dropna()
sc["pAB"] = sc["D1"].str.upper().str.strip()+"__"+sc["D2"].str.upper().str.strip()
sc["pBA"] = sc["D2"].str.upper().str.strip()+"__"+sc["D1"].str.upper().str.strip()
raw_set = set(raw["pair_norm"])
rows = []
for _,r in sc.iterrows():
    p = r["pAB"] if r["pAB"] in raw_set else (r["pBA"] if r["pBA"] in raw_set else None)
    if p: rows.append({"pair":p,"score":r["ScBW"]})
tdf = pd.DataFrame(rows)
print(f"   Matched: {len(tdf)} scored pairs")

print("2. Build train matrix...")
mask = raw["pair_norm"].isin(set(tdf["pair"]))
Xr = raw[mask].copy()
Xr["score"] = Xr["pair_norm"].map(dict(zip(tdf["pair"],tdf["score"])))
y = Xr["score"].values
n = len(y)

# Same preprocessing as classification
CHANGE_THRESHOLD = 1e-6
MIN_FREQ_RATIO = 0.1
MIN_FREQ_ABS = 5

stds = Xr[feat_cols].std()
keep = stds[stds > 0].index.tolist()
freq_th = max(MIN_FREQ_ABS, round(MIN_FREQ_RATIO * n))
freq = (np.abs(Xr[keep]) > CHANGE_THRESHOLD).sum(axis=0)
keep = freq[freq >= freq_th].index.tolist()
print(f"   Features: {len(feat_cols)} -> {len(keep)} (zero-var + freq>={freq_th})")

X_l1 = Xr[keep].div(Xr[keep].abs().sum(axis=1), axis=0).fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(X_l1)

print("3. Cross-validation...")
cv = KFold(5, shuffle=True, random_state=42)

# Ridge
rcv = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=3)
rcv.fit(X, y)
ypr = cross_val_predict(Ridge(alpha=rcv.alpha_), X, y, cv=cv)
r2r = r2_score(y, ypr)
maer = mean_absolute_error(y, ypr)
cr = np.corrcoef(y, ypr)[0,1]
print(f"   Ridge(a={rcv.alpha_:.3f}): R2={r2r:+.4f} MAE={maer:.4f} r={cr:+.4f}")

# GBR
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
ypg = cross_val_predict(gbr, X, y, cv=cv)
r2g = r2_score(y, ypg)
maeg = mean_absolute_error(y, ypg)
cg = np.corrcoef(y, ypg)[0,1]
print(f"   GBR:              R2={r2g:+.4f} MAE={maeg:.4f} r={cg:+.4f}")

# Pick best
if r2r >= r2g:
    best_name = "Ridge"
    final = Ridge(alpha=rcv.alpha_)
    best_r2, best_mae, best_r = r2r, maer, cr
else:
    best_name = "GBR"
    final = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    best_r2, best_mae, best_r = r2g, maeg, cg

final.fit(X, y)
yt = final.predict(X)
print(f"\n   Best: {best_name} | Train R2={r2_score(y,yt):.4f} r={np.corrcoef(y,yt)[0,1]:.4f}")

print("4. Save model...")
OUT = r"../data/regression_predictions"
os.makedirs(OUT, exist_ok=True)

joblib.dump(final, os.path.join(OUT, "regression_model.joblib"))
joblib.dump(scaler, os.path.join(OUT, "scaler.joblib"))
joblib.dump(list(keep), os.path.join(OUT, "feature_names.joblib"))

preproc = {
    "keep_features": list(keep),
    "all_features": feat_cols,
    "CHANGE_THRESHOLD": CHANGE_THRESHOLD,
    "freq_threshold": freq_th,
    "MIN_FREQ_RATIO": MIN_FREQ_RATIO,
    "MIN_FREQ_ABS": MIN_FREQ_ABS,
    "n_train": n,
}
joblib.dump(preproc, os.path.join(OUT, "preprocessing.joblib"))

pd.DataFrame({
    "drug_pair": Xr["drug_pair"].values,
    "score_actual": y,
    "score_predicted": yt,
}).to_csv(os.path.join(OUT, "training_predictions.csv"), index=False, float_format="%.6f")

summary = {
    "model": best_name,
    "n_samples": int(n),
    "n_features": len(keep),
    "cv_R2": round(best_r2, 4),
    "cv_MAE": round(best_mae, 4),
    "cv_r": round(best_r, 4),
    "train_R2": round(r2_score(y, yt), 4),
    "train_r": round(np.corrcoef(y, yt)[0,1], 4),
    "ridge_CV": {"R2": round(r2r,4), "MAE": round(maer,4), "r": round(cr,4)},
    "gbr_CV": {"R2": round(r2g,4), "MAE": round(maeg,4), "r": round(cg,4)},
}
json.dump(summary, open(os.path.join(OUT, "model_info.json"), "w"), indent=2)

print(f"   Done: {OUT}")
print(f"   {json.dumps(summary, indent=2)}")
