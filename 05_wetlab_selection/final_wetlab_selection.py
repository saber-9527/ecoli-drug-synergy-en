# Final wet-lab candidate selection with strict criteria
# 1. Drug classification cleanup (no more Unknowns leaking through)
# 2. Filter: min_train >= 2 for at least one drug, both >= 1
# 3. Sort by M2 proba (preserve gradient)
# 4. Exclude combos already in Supp Table 2 with significant scores
# 5. Diverse selection: max 3 per drug
import pandas as pd, numpy as np, os
from collections import defaultdict

OUTDIR = r"../data/predictions_all_combos"
df = pd.read_csv(os.path.join(OUTDIR, "all_combo_predictions.csv"))
meta = pd.read_csv(r"../data/metadata_merged.csv")
train_pairs = set(meta["drug_pair"].astype(str).str.strip().str.upper().str.replace("-","__"))
train_cnt = defaultdict(int)
for p in train_pairs:
    ps = p.split("__")
    if len(ps)==2: train_cnt[ps[0]]+=1; train_cnt[ps[1]]+=1

print(f"Total: {len(df)}")

# ============================================================
# COMPREHENSIVE DRUG CLASSIFICATION — NO LEAKS
# ============================================================
# Named antibiotics by class
ANTIBIOTICS = {
    "Beta-Lactam": "AMOXICILLIN AMPICILLIN AZTREONAM CARBENICILLIN CEFACLOR CEFOXITIN CEFSULODIN CEFTAZIDIME MECILLINAM OXACILLIN PENICILLIN CLOXACILLIN DICLOXACILLIN FLUCLOXACILLIN PIPERACILLIN CEFALOTIN CEFAMANDOLE CEFOPERAZONE IMIPENEM MEROPENEM".split(),
    "Aminoglycoside": "AMIKACIN GENTAMICIN KANAMYCIN SPECTINOMYCIN STREPTOMYCIN TOBRAMYCIN NEOMYCIN NETILMICIN PAROMOMYCIN".split(),
    "Macrolide": "AZITHROMYCIN CLARITHROMYCIN ERYTHROMYCIN SPIRAMYCIN CLARYTHROMYCIN TYLOSIN TELITHROMYCIN".split(),
    "Tetracycline": "DOXYCYCLINE MINOCYCLINE TETRACYCLINE OXYTETRACYCLINE CHLORTETRACYCLINE".split(),
    "Quinolone": "CIPROFLOXACIN LEVOFLOXACIN NALIDIXICACID NORFLOXACIN NOVOBIOCIN OFLOXACIN MOXIFLOXACIN ENROFLOXACIN NALIDIXIC".split(),
    "Folate_Inhibitor": "TRIMETHOPRIM SULFAMETHIZOLE SULFAMETHOXAZOLE SULFAMONOMETHOXINE SULFADIAZINE SULFANILAMIDE SULFATHIAZOLE".split(),
    "CellWall_Other": "BACITRACIN CERULENIN CYCLOSERINED FOSFOMYCIN VANCOMYCIN CYCLOSERINE D-CYCLOSERINE PHOSPHOMYCIN TEICOPLANIN DALBAVANCIN".split(),
    "DNA_Damage_RNA": "MITOMYCINC NITROFURANTOIN METRONIDAZOLE RIFAMPICIN CISPLATIN".split(),
    "Membrane": "POLYMYXINB COLISTIN DAPTOMYCIN GRAMICIDIN NISIN".split(),
    "Protein_Synthesis": "CHLORAMPHENICOL FUSIDICACID PUROMYCIN LINEZOLID MUPIROCIN THIOSTREPTON".split(),
    "Other_Antimicrobial": "TRICLOSAN ISONIAZID ETHIONAMIDE THEOPHYLLINE INDOLICIDIN CECROPINB CHIR090 A22".split(),
}

ALL_ANTIBIOTICS = set()
for lst in ANTIBIOTICS.values(): ALL_ANTIBIOTICS.update(lst)

# Known non-antibiotic items that appeared in the screen
NON_ANTIBIOTIC = {
    # Solvents / vehicles
    "DMSO","ETHANOL","GLYCEROL","METHANOL","PROPYLENEGLYCOL",
    # Salts / metals / stress
    "NACL","NH4CL","HIGHFE","HIGHCOPPER","HIGHNICKEL","HIGHCOBALT","LOWFE","LOWCOPPER","LOWZINC",
    "ANAEROBIC","UREA","PEROXIDE","CL","MMS","CCCP","PARAQUAT","PYOCYANIN","AZIDE","ARSENITE","BILE",
    # Chelators
    "EDTA","EGTA","CITRATE","DMSA","DEFEROXAMINE",
    # Detergents
    "SDS","TRITONX","TWEEN20","TWEEN80","BENZALKONIUM","CHLORHEXIDINE","CETRIMIDE",
    # Bile acids
    "CHOLATE","DEOXYCHOLATE","TAUROCHOLATE","CHENODEOXYCHOLATE",
    # Dyes / stains
    "ACRIFLAVINE","PROPIDIUMIODIDE","ETHIDIUMBROMIDE","CALCOFLUOR","CRYSTALVIOLET","METHYLENEBLUE",
    # Local anesthetics
    "PROCAINE","DIBUCAINE","LIDOCAINE","BUPIVACAINE","TETRACAINE","BENZOCAINE",
    # Neurotransmitters / hormones
    "NOREPINEPHRINE","EPINEPHRINE","DOPAMINE","SEROTONIN","ACETYLCHOLINE","HISTAMINE",
    # Carbon sources / metabolites
    "GLUCOSE","FRUCTOSE","SUCROSE","SORBITOL","INOSITOL","GLUCOSAMINE","N-ACETYLGLUCOSAMINE",
    "LACTULOSE","MALTOSE","ARABINOSE","MANNOSE","XYLOSE","GALACTOSE","TREHALOSE",
    "ACETATE","SUCCINATE","PYRUVATE","LACTATE","FUMARATE","MALATE","GLUCONATE",
    # Other non-antibiotic
    "EGCG","CURCUMIN","RESVERATROL","QUERCETIN","VERAPAMIL","LOPERAMIDE",
    "CHLOROPROMAZINE","PMS","PMSF","DTT","HEPES","TRIS","MOPS","PIPES",
    "NIGERICIN","RADICICOL","STREPTONIGRIN","THIOLACTOMYCIN","TUNICAMYCIN",
    "GLUFOSFOMYCIN","BICYCLOMYCIN","ACTINOMYCIND","HYDROXYUREA","STREPTOZOTOCIN",
    "LACTULOSE",
}

def get_class(d):
    for cls_name, lst in ANTIBIOTICS.items():
        if d in lst: return cls_name
    return None

# ============================================================
# STEP 1: KEEP ONLY TRUE ANTIBIOTIC-ANTIBIOTIC PAIRS
# ============================================================
mask_real = df["drug_a"].apply(lambda d: d in ALL_ANTIBIOTICS) & \
            df["drug_b"].apply(lambda d: d in ALL_ANTIBIOTICS)
df_ab = df[mask_real].copy()
print(f"Step 1 (true antibiotic pairs): {mask_real.sum()}")

# ============================================================
# STEP 2: EXCLUDE TRAINING SET PAIRS
# ============================================================
mask_new = ~df_ab["drug_pair"].isin(train_pairs)
df_new = df_ab[mask_new].copy()
print(f"Step 2 (exclude training): {len(df_new)}")

# ============================================================
# STEP 3: ADD TRAINING FAMILIARITY + DRUG CLASS
# ============================================================
df_new["TR_A"] = df_new["drug_a"].apply(lambda d: train_cnt.get(d, 0)).astype(int)
df_new["TR_B"] = df_new["drug_b"].apply(lambda d: train_cnt.get(d, 0)).astype(int)
df_new["TR_MIN"] = df_new[["TR_A","TR_B"]].min(axis=1)
df_new["TR_SUM"] = df_new[["TR_A","TR_B"]].sum(axis=1)
df_new["Class_A"] = df_new["drug_a"].apply(get_class)
df_new["Class_B"] = df_new["drug_b"].apply(get_class)

# Filter: at least one drug trained >= 2
df_fam = df_new[df_new["TR_MIN"] >= 1].copy()
print(f"Step 3 (TR_MIN >= 1): {len(df_fam)}  (mean M2={df_fam.proba_411_model.mean():.3f})")

# ============================================================
# STEP 4: PHARMACOLOGICAL CONFLICT FLAGS (same target / cidal-static)
# ============================================================
S50 = {"ERYTHROMYCIN","AZITHROMYCIN","CLARITHROMYCIN","CLARYTHROMYCIN","SPIRAMYCIN","CHLORAMPHENICOL","PUROMYCIN","FUSIDICACID","LINEZOLID"}
S30 = {"TETRACYCLINE","DOXYCYCLINE","MINOCYCLINE","GENTAMICIN","AMIKACIN","TOBRAMYCIN","SPECTINOMYCIN","STREPTOMYCIN","KANAMYCIN","NEOMYCIN"}
PBP = set(ANTIBIOTICS["Beta-Lactam"] + ANTIBIOTICS["CellWall_Other"])
GYR = set(ANTIBIOTICS["Quinolone"]) - {"NOVOBIOCIN"}
FOL = set(ANTIBIOTICS["Folate_Inhibitor"])

CIDAL = set(ANTIBIOTICS["Beta-Lactam"] + ANTIBIOTICS["Aminoglycoside"] + ANTIBIOTICS["Quinolone"] +
    ANTIBIOTICS["DNA_Damage_RNA"] + ANTIBIOTICS["Membrane"] +
    ["VANCOMYCIN","FOSFOMYCIN","BACITRACIN","POLYMYXINB","COLISTIN","NITROFURANTOIN"])
STATIC = set(ANTIBIOTICS["Macrolide"] + ANTIBIOTICS["Tetracycline"] + ANTIBIOTICS["Folate_Inhibitor"] +
    ANTIBIOTICS["Protein_Synthesis"] + ["CHLORAMPHENICOL","PUROMYCIN","LINEZOLID","CERULENIN"])

def has_conflict(r):
    d1, d2 = r["drug_a"], r["drug_b"]
    # Same target
    if d1 in S50 and d2 in S50: return True
    if d1 in S30 and d2 in S30: return True
    if d1 in PBP and d2 in PBP and r["Class_A"] == r["Class_B"] == "Beta-Lactam": return True
    if d1 in GYR and d2 in GYR: return True
    if d1 in FOL and d2 in FOL and d1 != d2: return True  # different folate inhibitors OK
    # Cidal/Static (only flag if cidal needs active growth)
    if (d1 in CIDAL and d2 in STATIC) or (d2 in CIDAL and d1 in STATIC):
        return True
    return False

mask_clean = ~df_fam.apply(has_conflict, axis=1)
df_clean = df_fam[mask_clean].copy()
print(f"Step 4 (no pharmacological conflicts): {len(df_clean)}")

# ============================================================
# STEP 5: CHECK SUPP TABLE 2 — flag previously tested combos
# ============================================================
xl = pd.read_excel(r"../data/1-ecoli.xlsx", "Supp Table 2", header=1)
xl = xl.iloc[2:].copy()
supp_tested = {}
for _,r in xl.iterrows():
    d1 = str(r.iloc[1]).strip().upper()
    d2 = str(r.iloc[2]).strip().upper()
    try: score = float(r.iloc[14])
    except: score = None
    sign = str(r.iloc[8]).strip() if pd.notna(r.iloc[8]) else None
    if d1 and d2 and d1 != "NAN" and d2 != "NAN":
        supp_tested[f"{d1}__{d2}"] = (sign, score)
        supp_tested[f"{d2}__{d1}"] = (sign, score)

def in_supp_table(pair):
    info = supp_tested.get(pair)
    if info is None: return "NOVEL"
    sign, score = info
    if score is not None and abs(score) > 0.5:  # Strong signal — already confirmed
        return "KNOWN_STRONG"
    if score is not None and abs(score) > 0.2:  # Moderate signal
        return "KNOWN_MODERATE"
    return "SCREENED_ONLY"  # In Supp Table but no strong score

df_clean["SuppTable_Status"] = df_clean["drug_pair"].apply(in_supp_table)

# Keep NOVEL + SCREENED_ONLY (not independently validated)
# Exclude KNOWN_STRONG (already has clear experimental result)
df_final = df_clean[df_clean["SuppTable_Status"] != "KNOWN_STRONG"].copy()
print(f"Step 5 (exclude SuppT2 with strong signal): {len(df_final)}")
print(f"  NOVEL: {(df_final['SuppTable_Status']=='NOVEL').sum()}")
print(f"  SCREENED_ONLY: {(df_final['SuppTable_Status']=='SCREENED_ONLY').sum()}")
print(f"  KNOWN_MODERATE: {(df_final['SuppTable_Status']=='KNOWN_MODERATE').sum()}")

# ============================================================
# STEP 6: M2 PROBA DISTRIBUTION IN FINAL POOL
# ============================================================
print(f"\n=== M2 PROBA IN FINAL POOL ({len(df_final)} combos) ===")
for t in [1.0, 0.99, 0.95, 0.90, 0.80, 0.50, 0.10, 0.05, 0.01, 0.0]:
    n = (df_final.proba_411_model >= t).sum() if t > 0 else len(df_final)
    lbl = f">= {t}"
    print(f"  {lbl:<8s}: {n}")

# ============================================================
# STEP 7: TIER + DIVERSIFY
# ============================================================
def diverse(df_pool, sort_col, asc, n=40, max_drug=3):
    df_sorted = df_pool.sort_values(sort_col, ascending=asc)
    sel = []; cnt = defaultdict(int)
    for _, r in df_sorted.iterrows():
        d1, d2 = r["drug_a"], r["drug_b"]
        if cnt[d1] < max_drug and cnt[d2] < max_drug:
            sel.append(r.to_dict()); cnt[d1] += 1; cnt[d2] += 1
        if len(sel) >= n: break
    return pd.DataFrame(sel)

# Synergy: M2 >= 0.90, TR_MIN >= 1
pool_s = df_final[(df_final["proba_411_model"] >= 0.90) & (df_final["TR_MIN"] >= 1)]
sel_s = diverse(pool_s, "proba_411_model", False, 40, 3)
sel_s["Tier"] = "A_Synergy"
print(f"\nA_Synergy (M2>=0.90, TR>=1): {len(pool_s)} -> {len(sel_s)}")

# Antagonism: M2 <= 0.10, TR_MIN >= 1
pool_a = df_final[(df_final["proba_411_model"] <= 0.10) & (df_final["TR_MIN"] >= 1)]
sel_a = diverse(pool_a, "proba_411_model", True, 25, 3)
sel_a["Tier"] = "B_Antagonism"
print(f"B_Antagonism (M2<=0.10, TR>=1): {len(pool_a)} -> {len(sel_a)}")

# Borderline: M2 0.25-0.75, TR_MIN >= 1
pool_b = df_final[(df_final["proba_411_model"] >= 0.25) & (df_final["proba_411_model"] <= 0.75) & (df_final["TR_MIN"] >= 1)]
sel_b = diverse(pool_b, "proba_411_model", True, 25, 3)
sel_b["Tier"] = "C_Borderline"
print(f"C_Borderline (M2 0.25-0.75, TR>=1): {len(pool_b)} -> {len(sel_b)}")

# ============================================================
# STEP 8: ASSEMBLE + SAVE
# ============================================================
def flush_tier(df_sel, tier_name):
    out = []
    for _,r in df_sel.iterrows():
        pair = r["drug_pair"]
        info = supp_tested.get(pair, (None, None))
        supp_sign, supp_score = info[0] if info else None, info[1] if info else None
        agree = "Yes" if r["pred_199_model"]==r["pred_411_model"] else "No"
        out.append({
            "Tier": tier_name,
            "Drug_Pair": r["drug_pair"],
            "Drug_A": r["drug_a"],
            "Drug_B": r["drug_b"],
            "Class_A": r["Class_A"],
            "Class_B": r["Class_B"],
            "M2_Proba": round(r["proba_411_model"], 6),
            "M1_Proba": round(r["proba_199_model"], 6),
            "M2_Pred": int(r["pred_411_model"]),
            "Models_Agree": agree,
            "TrainCnt_A": int(r["TR_A"]),
            "TrainCnt_B": int(r["TR_B"]),
            "MinTrainCnt": int(r["TR_MIN"]),
            "SuppTable2_Status": r["SuppTable_Status"],
            "SuppTable2_Sign": supp_sign if supp_sign else "",
            "SuppTable2_Score": round(supp_score, 4) if supp_score is not None else "",
        })
    return pd.DataFrame(out)

out_pieces = []
for s, t in [(sel_s, "A_Synergy"), (sel_a, "B_Antagonism"), (sel_b, "C_Borderline")]:
    if len(s) > 0:
        out_pieces.append(flush_tier(s, t))

final_out = pd.concat(out_pieces, ignore_index=True)

# Save
out_xlsx = os.path.join(OUTDIR, "wetlab_final_candidates.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    for tier in ["A_Synergy","B_Antagonism","C_Borderline"]:
        sub = final_out[final_out["Tier"]==tier]
        if len(sub) > 0: sub.to_excel(w, sheet_name=tier, index=False)
    final_out.to_excel(w, sheet_name="All_Candidates", index=False)
    # Also save full pre-filtered pool for reference
    df_final_out = df_final.copy()
    df_final_out["M2_Proba"] = df_final_out["proba_411_model"]
    df_final_out[["Drug_Pair","drug_a","drug_b","M2_Proba","TR_A","TR_B","TR_MIN","SuppTable_Status"]].to_excel(
        w, sheet_name="Full_Filtered_Pool", index=False)

final_out.to_csv(os.path.join(OUTDIR, "wetlab_final_candidates.csv"), index=False, float_format="%.6f")

# ============================================================
# REPORT
# ============================================================
print(f"\n{'='*90}")
print("  FINAL WET-LAB CANDIDATES")
print(f"{'='*90}")
for tier in ["A_Synergy","B_Antagonism","C_Borderline"]:
    sub = final_out[final_out["Tier"]==tier]
    print(f"\n{tier} ({len(sub)})")
    print(f"{'─'*90}")
    for i,(_,r) in enumerate(sub.iterrows(),1):
        agree_s = " ✓" if r["Models_Agree"]=="Yes" else " ✗"
        supp_s = f" [{r['SuppTable2_Status']}]"
        if r["SuppTable2_Sign"]:
            supp_s = f" [SuppT2: {r['SuppTable2_Sign']} {r['SuppTable2_Score']}]"
        print(f"  {i:>2}. {r['Drug_Pair']:<50s} M2={r['M2_Proba']:.4f}  {r['Class_A']}+{r['Class_B']}  Tr=({r['TrainCnt_A']},{r['TrainCnt_B']}){agree_s}{supp_s}")

# Stats
print(f"\n{'='*90}")
print(f"  SUMMARY")
print(f"{'='*90}")
print(f"  Total filtered pool (true AB pairs, clean, TR>=1): {len(df_final)}")
print(f"  A_Synergy:     {len(sel_s)}")
print(f"  B_Antagonism:  {len(sel_a)}")
print(f"  C_Borderline:  {len(sel_b)}")
print(f"  TOTAL selected: {len(final_out)}")
print(f"\n  Supp Table 2 status:")
print(f"    NOVEL (never screened):          {(final_out['SuppTable2_Status']=='NOVEL').sum()}")
print(f"    SCREENED_ONLY (no strong signal): {(final_out['SuppTable2_Status']=='SCREENED_ONLY').sum()}")
print(f"    KNOWN_MODERATE (some evidence):   {(final_out['SuppTable2_Status']=='KNOWN_MODERATE').sum()}")
print(f"\n  Saved: {out_xlsx}")
print(f"  Saved: {os.path.join(OUTDIR, 'wetlab_final_candidates.csv')}")
