# Full pharmacological review — ALL 5565 predictions
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

# --- Classifications ---
BETA_LACTAM = "AMOXICILLIN AMPICILLIN AZTREONAM CARBENICILLIN CEFACLOR CEFOXITIN CEFSULODIN MECILLINAM OXACILLIN PENICILLIN".split()
AMINOGLYCO = "AMIKACIN GENTAMICIN KANAMYCIN SPECTINOMYCIN STREPTOMYCIN TOBRAMYCIN NEOMYCIN".split()
MACROLIDE = "AZITHROMYCIN CLARITHROMYCIN ERYTHROMYCIN SPIRAMYCIN CLARYTHROMYCIN TYLOSIN".split()
TETRACYCLINES = "DOXYCYCLINE MINOCYCLINE TETRACYCLINE".split()
QUINOLONES = "CIPROFLOXACIN LEVOFLOXACIN NALIDIXICACID NORFLOXACIN NOVOBIOCIN OFLOXACIN MOXIFLOXACIN".split()
FOLATE_INH = "TRIMETHOPRIM SULFAMETHIZOLE SULFAMETHOXAZOLE SULFAMONOMETHOXINE SULFADIAZINE".split()
CELLWALL = "BACITRACIN CERULENIN CYCLOSERINED FOSFOMYCIN VANCOMYCIN".split()
DNA_DMG = "MITOMYCINC NITROFURANTOIN CISPLATIN METRONIDAZOLE".split()
MEMBRANE = "POLYMYXINB COLISTIN DAPTOMYCIN NISIN".split()
RNAPOL = "RIFAMPICIN RIFABUTIN RIFAXIMIN".split()
PROT_OTHER = "CHLORAMPHENICOL FUSIDICACID LINEZOLID PUROMYCIN".split()
ANTICANCER = "DOXORUBICIN BLEOMYCIN PHLEOMYCIN METHOTREXATE AZIDOTHYMIDINE".split()
STRESSORS = "CCCP PARAQUAT PYOCYANIN PEROXIDE HIGHFE HIGHCOPPER BILE ANAEROBIC UREA NH4CL SDS A22".split()
CHANNEL = "VERAPAMIL LOPERAMIDE".split()
ANESTH = "PROCAINE DIBUCAINE LIDOCAINE".split()
DYES = "ACRIFLAVINE PROPIDIUMIODIDE ETHIDIUMBROMIDE CALCOFLUOR".split()
NEURO = "NOREPINEPHRINE EPINEPHRINE DOPAMINE SEROTONIN".split()
DETERGENTS = "BENZALKONIUM TRITONX TWEEN20 CHLORHEXIDINE".split()
CHELATORS = "EDTA EGTA CITRATE".split()
BILE_AC = "CHOLATE DEOXYCHOLATE TAUROCHOLATE".split()
SOLVENTS = "DMSO ETHANOL GLYCEROL METHANOL".split()
SUGARS = "GLUCOSE FRUCTOSE SUCROSE SORBITOL INOSITOL GLUCOSAMINE N.ACETYLGLUCOSAMINE LACTULOSE".replace(".","-").split()
METABOL = "ACETATE SUCCINATE PYRUVATE LACTATE FUMARATE MALATE".split()
POLYPH = "EGCG".split()
OTHER = "TRICLOSAN ISONIAZID THEOPHYLLINE INDOLICIDIN CECROPINB CHIR090 CYCLOSERINED".split()

S50 = set("ERYTHROMYCIN AZITHROMYCIN CLARITHROMYCIN CLARYTHROMYCIN SPIRAMYCIN CHLORAMPHENICOL PUROMYCIN FUSIDICACID LINEZOLID".split())
S30 = set("TETRACYCLINE DOXYCYCLINE MINOCYCLINE GENTAMICIN AMIKACIN TOBRAMYCIN SPECTINOMYCIN STREPTOMYCIN KANAMYCIN NEOMYCIN".split())
PBP_SET = set(BETA_LACTAM)
GYR_SET = set(QUINOLONES) - {"NOVOBIOCIN"}
NOVO = {"NOVOBIOCIN"}
FOL_SET = set(FOLATE_INH)

ALL_CLASSES = [
    ("Beta-Lactam",BETA_LACTAM),("Aminoglycoside",AMINOGLYCO),("Macrolide",MACROLIDE),
    ("Tetracycline",TETRACYCLINES),("Quinolone",QUINOLONES),("Folate_Inhibitor",FOLATE_INH),
    ("CellWall",CELLWALL),("DNA_Damage",DNA_DMG),("Membrane",MEMBRANE),("RNA_Pol",RNAPOL),
    ("Protein_Other",PROT_OTHER),("Anticancer",ANTICANCER),("Stress",STRESSORS),
    ("Channel",CHANNEL),("Anesthetic",ANESTH),("Dye",DYES),("Neuro",NEURO),
    ("Detergent",DETERGENTS),("Chelator",CHELATORS),("Bile_Acid",BILE_AC),
    ("Solvent",SOLVENTS),("Sugar",SUGARS),("Metabolite",METABOL),("Polyphenol",POLYPH),("Other",OTHER),
]

def classify(d):
    for cls_name, lst in ALL_CLASSES:
        if d in lst: return cls_name
    return "Unknown"

CIDAL = set(BETA_LACTAM + AMINOGLYCO + QUINOLONES + DNA_DMG + MEMBRANE + "VANCOMYCIN FOSFOMYCIN POLYMYXINB COLISTIN NITROFURANTOIN RIFAMPICIN BACITRACIN METRONIDAZOLE".split())
STATIC = set(MACROLIDE + TETRACYCLINES + FOLATE_INH + PROT_OTHER + "CHLORAMPHENICOL PUROMYCIN ERYTHROMYCIN AZITHROMYCIN CLARITHROMYCIN CLARYTHROMYCIN SPIRAMYCIN SPECTINOMYCIN CERULENIN LINEZOLID".split())
NON_DRUGS = set(STRESSORS + DYES + SOLVENTS + SUGARS + METABOL + BILE_AC + CHELATORS + DETERGENTS + POLYPH + NEURO + "PROCAINE DIBUCAINE VERAPAMIL".split())
G_NEG = set("CHIR090 AZTREONAM COLISTIN POLYMYXINB".split())
G_POS = set("VANCOMYCIN BACITRACIN DAPTOMYCIN".split())

def get_targets(d):
    t = []
    if d in S50: t.append("50S")
    if d in S30: t.append("30S")
    if d in PBP_SET: t.append("PBP")
    if d in GYR_SET: t.append("Gyrase")
    if d in NOVO: t.append("GyrB")
    if d in FOL_SET: t.append("Folate")
    return t

# Process all
rows = []
for _, r in df.iterrows():
    d1, d2 = r["drug_a"], r["drug_b"]
    m2, m1 = r["proba_411_model"], r["proba_199_model"]
    agree = r["pred_199_model"] == r["pred_411_model"]
    c1, c2 = classify(d1), classify(d2)
    t1_list, t2_list = get_targets(d1), get_targets(d2)

    flags = []
    issues = []
    plausibility = 3  # baseline

    # Non-drug
    if d1 in NON_DRUGS or d2 in NON_DRUGS:
        flags.append("NonDrug"); issues.append("非典型药物/胁迫/代谢物"); plausibility -= 2

    # Same target
    same_targets = [t for t in t1_list if t in t2_list]
    if same_targets:
        flags.append(f"SameTarget({','.join(same_targets)})")
        issues.append(f"竞争{','.join(same_targets)}靶点"); plausibility -= 3

    # Cidal/Static
    is_c1 = d1 in CIDAL; is_c2 = d2 in CIDAL
    is_s1 = d1 in STATIC; is_s2 = d2 in STATIC
    if (is_s1 and is_c2) or (is_s2 and is_c1):
        flags.append("Cidal_Static"); issues.append("杀菌/抑菌冲突"); plausibility -= 2

    # Cross species
    if (d1 in G_NEG and d2 in G_POS) or (d2 in G_NEG and d1 in G_POS):
        flags.append("CrossSpecies"); issues.append("G-专属+G+专属"); plausibility -= 3

    # Anticancer
    if d1 in ANTICANCER or d2 in ANTICANCER:
        flags.append("Anticancer"); issues.append("抗癌药,抗菌意义有限"); plausibility -= 1

    # Known rifampin interaction
    if "RIFAMPICIN" in (d1,d2):
        other_d = d2 if d1 == "RIFAMPICIN" else d1
        if other_d in MACROLIDE or other_d == "CHLORAMPHENICOL":
            flags.append("Known_Interaction"); issues.append("利福平诱导代谢/外排"); plausibility -= 1

    # Different classes bonus
    if c1 != c2 and c1 != "Unknown" and c2 != "Unknown":
        plausibility += 1

    # Training familiarity bonus
    min_tr = min(train_cnt.get(d1,0), train_cnt.get(d2,0))
    if min_tr >= 3:
        plausibility += 1

    if not agree:
        plausibility -= 1

    in_train = r["drug_pair"] in train_pairs

    rows.append({
        "Drug_Pair": r["drug_pair"], "Drug_A": d1, "Drug_B": d2,
        "Class_A": c1, "Class_B": c2,
        "M2_Proba": m2, "M1_Proba": m1, "Avg_Proba": (m1+m2)/2,
        "M2_Pred": int(r["pred_411_model"]), "M1_Pred": int(r["pred_199_model"]),
        "Agree": "Yes" if agree else "No",
        "InTrain": "Yes" if in_train else "No",
        "Flags": " | ".join(flags) if flags else "",
        "Issues": "; ".join(issues) if issues else "",
        "Plausibility": plausibility,
        "MinTrainCnt": min_tr,
        "TrainCnt_A": train_cnt.get(d1,0), "TrainCnt_B": train_cnt.get(d2,0),
    })

full = pd.DataFrame(rows)

# Stats
print("\n=== FULL REVIEW OF 5565 PREDICTIONS ===")
print(f"Total: {len(full)}")
print(f"  Same target:     {(full['Plausibility']<=0) & full['Flags'].str.contains('SameTarget')} combos -> {((full['Plausibility']<=0) & full['Flags'].str.contains('SameTarget')).sum()}")
print(f"  Same-target pairs: {(full['Flags'].str.contains('SameTarget')).sum()}")
print(f"  Cidal/Static pairs: {(full['Flags'].str.contains('Cidal_Static')).sum()}")
print(f"  Non-drug pairs:     {(full['Flags'].str.contains('NonDrug')).sum()}")
print(f"  Cross-species:      {(full['Flags'].str.contains('CrossSpecies')).sum()}")
print(f"  Anticancer:         {(full['Flags'].str.contains('Anticancer')).sum()}")
print(f"  Known interaction:  {(full['Flags'].str.contains('Known_Interaction')).sum()}")

# Clean subsets
clean = full[full["Flags"] == ""].copy()
print(f"\nClean (no flags): {len(clean)}")

syn_hi = clean[(clean["Plausibility"] >= 3) & (clean["M2_Proba"] >= 0.90)].sort_values(["Plausibility","M2_Proba"], ascending=[False,False])
ant_hi = clean[(clean["M2_Proba"] <= 0.10)].sort_values("M2_Proba")
bdl_hi = clean[(clean["M2_Proba"] >= 0.30) & (clean["M2_Proba"] <= 0.70)].sort_values("M2_Proba")

print(f"  Clean Synergy (Plaus>=3, M2>=0.90): {len(syn_hi)}")
print(f"  Clean Antagonism (M2<=0.10):        {len(ant_hi)}")
print(f"  Clean Borderline (0.30-0.70):       {len(bdl_hi)}")

# Diversity
def diverse(df_pool, sort_col, asc, n=40, max_drug=2):
    df_pool = df_pool.sort_values(sort_col, ascending=asc)
    sel = []; cnt = defaultdict(int)
    for _, r in df_pool.iterrows():
        if cnt[r["Drug_A"]] < max_drug and cnt[r["Drug_B"]] < max_drug:
            sel.append(r.to_dict()); cnt[r["Drug_A"]] += 1; cnt[r["Drug_B"]] += 1
        if len(sel) >= n: break
    return pd.DataFrame(sel)

syn_sel = diverse(syn_hi, "M2_Proba", False, 40)
ant_sel = diverse(ant_hi, "M2_Proba", True, 25)
bdl_sel = diverse(bdl_hi, "M2_Proba", True, 20)
syn_sel["Tier"] = "Synergy"; ant_sel["Tier"] = "Antagonism"; bdl_sel["Tier"] = "Borderline"

print(f"\n=== AFTER DIVERSITY ===")
print(f"  Synergy: {len(syn_sel)} | Antagonism: {len(ant_sel)} | Borderline: {len(bdl_sel)}")

# Assemble
sels = []
for df_s, tier in [(syn_sel,"Synergy"), (ant_sel,"Antagonism"), (bdl_sel,"Borderline")]:
    df_s["Tier"] = tier
    sels.append(df_s)

out_cols = ["Tier","Drug_Pair","Drug_A","Drug_B","Class_A","Class_B",
    "M2_Proba","M1_Proba","Avg_Proba","M2_Pred","M1_Pred","Agree","InTrain",
    "Flags","Issues","Plausibility","MinTrainCnt","TrainCnt_A","TrainCnt_B"]
final = pd.concat([s[out_cols] for s in sels], ignore_index=True)

out_xlsx = os.path.join(OUTDIR, "full_review_candidates.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    for tier in ["Synergy","Antagonism","Borderline"]:
        sub = final[final["Tier"]==tier]
        if len(sub) > 0: sub.to_excel(w, sheet_name=tier, index=False)
    final.to_excel(w, sheet_name="All_Selected", index=False)
    # Also save full classified
    full.to_excel(w, sheet_name="Full_5565_Classified", index=False)

final.to_csv(os.path.join(OUTDIR, "full_review_candidates.csv"), index=False)

print(f"\nSaved: {out_xlsx}")

# Show
for tier in ["Synergy","Antagonism","Borderline"]:
    sub = final[final["Tier"]==tier]
    print(f"\n{'='*80}")
    print(f"  {tier} ({len(sub)})")
    print(f"{'='*80}")
    for i,(_,r) in enumerate(sub.iterrows(),1):
        train_tag = " [TRAIN]" if r["InTrain"]=="Yes" else ""
        agree_tag = " ✓" if r["Agree"]=="Yes" else " ✗"
        print(f"  {i:>2}. {r['Drug_Pair']:<48s} M2={r['M2_Proba']:.4f}  {r['Class_A']}+{r['Class_B']}{agree_tag}{train_tag}")

# Bad predictions that model confidently got wrong
print(f"\n\n{'='*80}")
print(f"  ⚠ LIKELY WRONG PREDICTIONS (plausibility <= 1, M2>=0.95 confidence)")
print(f"{'='*80}")
wrong = full[(full["Plausibility"] <= 1) & (full["M2_Proba"] >= 0.95)].sort_values("Plausibility")
wrong = diverse(wrong, "Plausibility", True, 20, 3)
for i,(_,r) in enumerate(wrong.iterrows(),1):
    print(f"  {i:>2}. {r['Drug_Pair']:<48s} M2={r['M2_Proba']:.4f}  {r['Issues']}")
