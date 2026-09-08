# Build complete table of 1660 clean drug-drug combinations
import pandas as pd, os
from collections import defaultdict, Counter

full = pd.read_excel(r"../data/predictions_all_combos\full_review_candidates.xlsx", sheet_name="Full_5565_Classified")
print(f"Loaded: {full.shape}")

# Clean drug-drug combos
clean = full[full["Flags"].isna()].copy()
print(f"Clean: {len(clean)}")

# --- Drug classification (refined) ---
BETA_LACTAM = set("AMOXICILLIN AMPICILLIN AZTREONAM CARBENICILLIN CEFACLOR CEFOXITIN CEFSULODIN MECILLINAM OXACILLIN PENICILLIN PYRIDINE CEFTAZIDIME".split())
AMINOGLYCO = set("AMIKACIN GENTAMICIN KANAMYCIN SPECTINOMYCIN STREPTOMYCIN TOBRAMYCIN NEOMYCIN".split())
MACROLIDE = set("AZITHROMYCIN CLARITHROMYCIN ERYTHROMYCIN SPIRAMYCIN CLARYTHROMYCIN TYLOSIN".split())
TETRACYCLINE = set("DOXYCYCLINE MINOCYCLINE TETRACYCLINE".split())
QUINOLONE = set("CIPROFLOXACIN LEVOFLOXACIN NALIDIXICACID NORFLOXACIN NOVOBIOCIN OFLOXACIN MOXIFLOXACIN".split())
FOLATE = set("TRIMETHOPRIM SULFAMETHIZOLE SULFAMETHOXAZOLE SULFAMONOMETHOXINE".split())
CELLWALL = set("BACITRACIN CERULENIN CYCLOSERINED FOSFOMYCIN VANCOMYCIN".split())
DNA_DMG = set("MITOMYCINC NITROFURANTOIN CISPLATIN METRONIDAZOLE".split())
MEMBRANE = set("POLYMYXINB COLISTIN DAPTOMYCIN".split())
RNAPOL = set("RIFAMPICIN".split())
PROT_OTHER = set("CHLORAMPHENICOL FUSIDICACID PUROMYCIN".split())
OTHER_DRUG = set("TRICLOSAN ISONIAZID THEOPHYLLINE INDOLICIDIN CECROPINB CHIR090 CYCLOSERINED MMS RADICICOL STREPTONIGRIN THIOLACTOMYCIN TUNICAMYCIN NIGERICIN GLUFOSFOMYCIN BICYCLOMYCIN ACTINOMYCIND HYDROXYUREA STREPTOZOTOCIN".split())

def dc(d):
    if d in BETA_LACTAM: return "Beta-Lactam"
    if d in AMINOGLYCO: return "Aminoglycoside"
    if d in MACROLIDE: return "Macrolide"
    if d in TETRACYCLINE: return "Tetracycline"
    if d in QUINOLONE: return "Quinolone"
    if d in FOLATE: return "Folate_Inhibitor"
    if d in CELLWALL: return "CellWall_Other"
    if d in DNA_DMG: return "DNA_Damage"
    if d in MEMBRANE: return "Membrane"
    if d in RNAPOL: return "RNA_Polymerase"
    if d in PROT_OTHER: return "Protein_Synthesis"
    if d in OTHER_DRUG: return "Other_Antimicrobial"
    return "Unknown"

clean["Drug_Class_A"] = clean["Drug_A"].apply(dc)
clean["Drug_Class_B"] = clean["Drug_B"].apply(dc)

# --- Classification into synergy/antagonism/borderline ---
clean["Prediction_Label"] = ""
clean.loc[(clean["Plausibility"] >= 3) & (clean["M2_Proba"] >= 0.90), "Prediction_Label"] = "High_Confidence_Synergy"
clean.loc[(clean["M2_Proba"] <= 0.10), "Prediction_Label"] = "Likely_Antagonism"
clean.loc[(clean["M2_Proba"] >= 0.30) & (clean["M2_Proba"] <= 0.70), "Prediction_Label"] = "Neutral_Borderline"
clean.loc[clean["Prediction_Label"] == "", "Prediction_Label"] = "Unclassified"

print(f"\n1660 clean combos by prediction:")
print(clean["Prediction_Label"].value_counts())

# --- Diversity selection ---
def diverse(df_pool, sort_col, asc, n=50, max_drug=3):
    df_pool = df_pool.sort_values(sort_col, ascending=asc)
    sel = []; cnt = defaultdict(int)
    for _, r in df_pool.iterrows():
        d1, d2 = r["Drug_A"], r["Drug_B"]
        if cnt[d1] < max_drug and cnt[d2] < max_drug:
            sel.append(r.to_dict()); cnt[d1] += 1; cnt[d2] += 1
        if len(sel) >= n: break
    return pd.DataFrame(sel)

syn_pool = clean[clean["Prediction_Label"] == "High_Confidence_Synergy"]
ant_pool = clean[clean["Prediction_Label"] == "Likely_Antagonism"]
bdl_pool = clean[clean["Prediction_Label"] == "Neutral_Borderline"]

syn_sel = diverse(syn_pool, "M2_Proba", False, 50, 3)
ant_sel = diverse(ant_pool, "M2_Proba", True, 30, 3)
bdl_sel = diverse(bdl_pool, "M2_Proba", True, 25, 3)
syn_sel["Tier"] = "1_Synergy"; ant_sel["Tier"] = "2_Antagonism"; bdl_sel["Tier"] = "3_Borderline"
print(f"\nAfter diversity (max 3/drug): Syn={len(syn_sel)}, Ant={len(ant_sel)}, Bdl={len(bdl_sel)}")

# --- Assemble output ---
sel_out = []
for s, t in [(syn_sel,"1_Synergy"),(ant_sel,"2_Antagonism"),(bdl_sel,"3_Borderline")]:
    s["Tier"] = t
    sel_out.append(s)

cols_sel = ["Tier","Drug_Pair","Drug_A","Drug_B","Drug_Class_A","Drug_Class_B",
    "M2_Proba","M1_Proba","Avg_Proba","M2_Pred","M1_Pred","Agree","InTrain",
    "Plausibility","MinTrainCnt","TrainCnt_A","TrainCnt_B"]
final_sel = pd.concat([s[cols_sel] for s in sel_out], ignore_index=True)

# Clean output columns
clean_cols = ["Drug_Pair","Drug_A","Drug_B","Drug_Class_A","Drug_Class_B",
    "M2_Proba","M1_Proba","Avg_Proba","M2_Pred","M1_Pred","Agree","InTrain",
    "Plausibility","MinTrainCnt","TrainCnt_A","TrainCnt_B","Prediction_Label"]

# --- Save ---
OUT = r"../data/predictions_all_combos"
out_xlsx = os.path.join(OUT, "clean_drug_combinations.xlsx")

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    clean[clean_cols].to_excel(w, sheet_name="All_1660_Clean_Combos", index=False)
    for tier in ["1_Synergy","2_Antagonism","3_Borderline"]:
        sub = final_sel[final_sel["Tier"]==tier]
        if len(sub) > 0: sub.to_excel(w, sheet_name=tier, index=False)
    final_sel.to_excel(w, sheet_name="All_Selected_105", index=False)

final_sel.to_csv(os.path.join(OUT, "clean_drug_combinations_selected.csv"), index=False)

# --- Report summary ---
print(f"\n=== 2485 REAL DRUG COMBOS FROM 5565 PREDICTIONS ===")
print(f"  Clean (no pharmacological conflicts):  1,660 (66.8%)")
print(f"  Flagged (same-target, cidal/static, etc.):  825 (33.2%)")
print(f"  Total real drug pool:                    2,485")

print(f"\n=== 1660 CLEAN COMBOS BY PREDICTION ===")
for label in ["High_Confidence_Synergy","Likely_Antagonism","Neutral_Borderline","Unclassified"]:
    n = (clean["Prediction_Label"] == label).sum()
    pct = n / len(clean) * 100
    print(f"  {label:<30s}: {n:>5d} ({pct:5.1f}%)")

# Drug class distribution
print(f"\n=== TOP 15 DRUG CLASS PAIRS IN 1660 CLEAN POOL ===")
pairs = [f"{r['Drug_Class_A']}+{r['Drug_Class_B']}" for _,r in clean.iterrows()]
for pair, n in Counter(pairs).most_common(15):
    print(f"  {pair:<50s} {n}")

print(f"\nSaved: {out_xlsx}")
print(f"Sheets: All_1660_Clean_Combos | 1_Synergy | 2_Antagonism | 3_Borderline | All_Selected_105")
