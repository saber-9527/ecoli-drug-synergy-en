# ============================================================
# Full pre-screening pipeline for wet-lab candidate selection
# Results saved to: D:\antibiotics\e-coil group\predictions_all_combos\
# ============================================================
import pandas as pd, numpy as np, os
from collections import defaultdict

OUTDIR = r"../data/predictions_all_combos"

# ---- Step 0: Load everything ----
df = pd.read_csv(os.path.join(OUTDIR, "all_combo_predictions.csv"))
meta = pd.read_csv(r"../data/metadata_merged.csv")
train_pairs = set(meta["drug_pair"].astype(str).str.strip().str.upper().str.replace("-","__"))
train_label = dict(zip(
    meta["drug_pair"].astype(str).str.strip().str.upper().str.replace("-","__"), meta["label"]))

train_drug_cnt = defaultdict(int)
for pair in train_pairs:
    parts = pair.split("__")
    if len(parts) == 2:
        train_drug_cnt[parts[0]] += 1; train_drug_cnt[parts[1]] += 1

# ---- Step 1: Remove non-drugs ----
non_drugs = {
    "NH4CL","HIGHFE","HIGHCOPPER","BILE","ANAEROBIC","UREA","NACL","PEROXIDE","CL","MMS",
    "CHOLATE","DEOXYCHOLATE","TAUROCHOLATE","DMSO","ETHANOL","GLYCEROL","METHANOL",
    "EDTA","EGTA","CITRATE","SDS","TRITONX","TWEEN20","BENZALKONIUM",
    "SUCCINATE","GLUCOSE","FRUCTOSE","SUCROSE","XYLOSE","ARABINOSE","MANNOSE",
    "MALTOSE","GALACTOSE","SORBITOL","INOSITOL","ACETATE","GLUCOSAMINE",
    "N-ACETYLGLUCOSAMINE","GLUCONATE","PYRUVATE","LACTATE","FUMARATE","MALATE",
    "EGCG","PROCAINE","DIBUCAINE","PROPIDIUMIODIDE","ETHIDIUMBROMIDE","CALCOFLUOR",
    "EPINEPHRINE","NOREPINEPHRINE","DOPAMINE","SEROTONIN",
}

def is_drug(p):
    d1, d2 = p.split("__")
    return d1 not in non_drugs and d2 not in non_drugs

df1 = df[df["drug_pair"].apply(is_drug)].copy()
print(f"Step 1 (remove non-drugs): {len(df)} → {len(df1)}")

# ---- Step 2: Exclude training combos ----
df2 = df1[~df1["drug_pair"].isin(train_pairs)].copy()
print(f"Step 2 (exclude training): {len(df1)} → {len(df2)}")

# ---- Step 3: Add training familiarity ----
df2["TR_A"] = df2["drug_pair"].apply(lambda p: train_drug_cnt.get(p.split("__")[0], 0)).astype(int)
df2["TR_B"] = df2["drug_pair"].apply(lambda p: train_drug_cnt.get(p.split("__")[1], 0)).astype(int)
df2["TR_MIN"] = df2[["TR_A","TR_B"]].min(axis=1)

# ---- Separate: model-agree for A/B/D; all for C3 ----
df2["agree"] = df2["pred_199_model"] == df2["pred_411_model"]
df_agree = df2[df2["agree"]].copy()
print(f"  Model agree: {len(df_agree)}, Disagree: {(~df2['agree']).sum()}")

# ---- Diversity selector ----
def diverse(df, sort_col, ascending, max_per_drug=2, n=30):
    df = df.sort_values(sort_col, ascending=ascending)
    sel = []; cnt = defaultdict(int)
    for _, r in df.iterrows():
        d1, d2 = r["drug_pair"].split("__")
        if cnt[d1] < max_per_drug and cnt[d2] < max_per_drug:
            sel.append(r.to_dict()); cnt[d1] += 1; cnt[d2] += 1
        if len(sel) >= n: break
    return pd.DataFrame(sel)

# ---- Tiers ----
results = {}

# A: m2>=0.99, min_train>=3, agree
pool = df_agree[(df_agree["proba_411_model"]>=0.99)&(df_agree["TR_MIN"]>=3)]
sel = diverse(pool, "proba_411_model", False, 2, 30)
sel["Tier"] = "A_Synergy_HiConf"; results["A"] = sel
print(f" A (m2>=0.99,TR>=3): {len(pool)} → {len(sel)}")

# B: m2>=0.90, min_train>=2, agree, not in A
used = set(sel["drug_pair"])
pool = df_agree[(df_agree["proba_411_model"]>=0.90)&(df_agree["TR_MIN"]>=2)&~df_agree["drug_pair"].isin(used)]
sel = diverse(pool, "proba_411_model", False, 2, 20)
sel["Tier"] = "B_Synergy_MedConf"; results["B"] = sel
print(f" B (m2>=0.90,TR>=2): {len(pool)} → {len(sel)}")

# C1: m2<=0.10, min_train>=2, agree (strict antagonism)
pool = df_agree[(df_agree["proba_411_model"]<=0.10)&(df_agree["TR_MIN"]>=2)]
sel = diverse(pool, "proba_411_model", True, 2, 15)
sel["Tier"] = "C1_Antag_Strict"; results["C1"] = sel
print(f" C1 (m2<=0.10,TR>=2,agree): {len(pool)} → {len(sel)}")

# C2: m2<=0.10, min_train>=1, agree (relaxed antagonism)
used = set(sel["drug_pair"]) if len(sel)>0 else set()
pool = df_agree[(df_agree["proba_411_model"]<=0.10)&(df_agree["TR_MIN"]>=1)&~df_agree["drug_pair"].isin(used)]
sel = diverse(pool, "proba_411_model", True, 2, 15)
sel["Tier"] = "C2_Antag_Relaxed"; results["C2"] = sel
print(f" C2 (m2<=0.10,TR>=1,agree): {len(pool)} → {len(sel)}")

# C3: m2<=0.10, min_train>=1, any agree/disagree (loose)
used = used | set(sel["drug_pair"]) if len(sel)>0 else used
pool = df2[(df2["proba_411_model"]<=0.10)&(df2["TR_MIN"]>=1)&~df2["drug_pair"].isin(used)]
sel = diverse(pool, "proba_411_model", True, 2, 15)
sel["Tier"] = "C3_Antag_Loose"; results["C3"] = sel
print(f" C3 (m2<=0.10,TR>=1,any): {len(pool)} → {len(sel)}")

# D: m2 0.30-0.70, min_train>=2, agree
pool = df_agree[(df_agree["proba_411_model"]>=0.30)&(df_agree["proba_411_model"]<=0.70)&(df_agree["TR_MIN"]>=2)]
sel = diverse(pool, "proba_411_model", True, 2, 20)
sel["Tier"] = "D_Borderline"; results["D"] = sel
print(f" D (m2 0.30-0.70,TR>=2): {len(pool)} → {len(sel)}")

# ---- Assemble ----
all_pieces = []
for key in ["A","B","C1","C2","C3","D"]:
    s = results.get(key)
    if s is not None and len(s) > 0:
        s["drug_a"], s["drug_b"] = s["drug_pair"].str.split("__", expand=True).values[:,0], s["drug_pair"].str.split("__", expand=True).values[:,1]
        all_pieces.append(s)

cols = ["Tier","drug_pair","drug_a","drug_b",
        "proba_411_model","proba_199_model","avg_proba",
        "TR_A","TR_B","TR_MIN","pred_411_model","pred_199_model"]
full = pd.concat([p[cols] for p in all_pieces], ignore_index=True)

# ---- Flag: in Supp Table 2? ----
try:
    xl = pd.read_excel(r"../data/1-ecoli.xlsx","Supp Table 2",header=1)
    xl = xl.iloc[2:].copy()
    supp = set()
    for _,r in xl.iterrows():
        d1=str(r.iloc[1]).strip().upper(); d2=str(r.iloc[2]).strip().upper()
        if d1 and d2 and d1!="NAN" and d2!="NAN":
            supp.add(f"{d1}__{d2}"); supp.add(f"{d2}__{d1}")
    full["SuppTable2"] = full["drug_pair"].apply(lambda p:"YES"if p in supp else"-")
except:
    full["SuppTable2"] = "?"

# ---- Rename + save ----
full_out = full.rename(columns={
    "drug_pair":"Drug_Pair","drug_a":"Drug_A","drug_b":"Drug_B",
    "proba_411_model":"M2_411_Proba","proba_199_model":"M1_199_Proba",
    "avg_proba":"Avg_Probas","pred_411_model":"M2_Pred","pred_199_model":"M1_Pred",
    "TR_A":"TrainFreq_A","TR_B":"TrainFreq_B","TR_MIN":"MinTrainFreq",
})

tier_order = ["A_Synergy_HiConf","B_Synergy_MedConf","C1_Antag_Strict","C2_Antag_Relaxed","C3_Antag_Loose","D_Borderline"]

out_xlsx = os.path.join(OUTDIR,"prescreened_candidates.xlsx")
out_csv = os.path.join(OUTDIR,"prescreened_candidates.csv")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    for tier in tier_order:
        sub = full_out[full_out["Tier"]==tier]
        if len(sub)>0: sub.to_excel(w, sheet_name=tier[:31], index=False)
    full_out.to_excel(w, sheet_name="All_Candidates", index=False)
full_out.to_csv(out_csv, index=False, float_format="%.6f")

# ---- Console report ----
print(f"\n{'='*80}")
print("  PRE-SCREENED CANDIDATES — ALL TIERS")
print(f"{'='*80}")
for tier in tier_order:
    n = len(full_out[full_out["Tier"]==tier])
    if n>0: print(f"  {tier:<25s}: {n}")
print(f"  {'TOTAL':<25s}: {len(full_out)}")
print(f"\n  Saved: {out_xlsx}")

# Show all tiers
for tier in tier_order:
    sub = full_out[full_out["Tier"]==tier]
    if len(sub)==0: continue
    print(f"\n{'─'*80}")
    print(f"  {tier} ({len(sub)})")
    print(f"{'─'*80}")
    for i,(_,r) in enumerate(sub.iterrows(),1):
        print(f"  {i:>2}. {r['Drug_Pair']:<50s} M2={r['M2_411_Proba']:.4f}  Tr=({r['TrainFreq_A']},{r['TrainFreq_B']})  SuppT2={r['SuppTable2']}")
