# ============================================================
# Full prediction review: flag all 85 candidates by pharmacological class
# Output: prescreened_candidates_reviewed.xlsx
# ============================================================
import pandas as pd, os

df = pd.read_csv(r"../data/predictions_all_combos\prescreened_candidates.csv")

# ---- Drug classification ----
BETA_LACTAM = {"AMOXICILLIN","AMPICILLIN","AZTREONAM","CARBENICILLIN","CEFACLOR",
    "CEFOXITIN","CEFSULODIN","MECILLINAM","OXACILLIN","PENICILLIN"}
AMINOGLYCOSIDE = {"AMIKACIN","GENTAMICIN","KANAMYCIN","SPECTINOMYCIN","STREPTOMYCIN","TOBRAMYCIN"}
MACROLIDE = {"AZITHROMYCIN","CLARITHROMYCIN","ERYTHROMYCIN","SPIRAMYCIN"}
TETRACYCLINE = {"DOXYCYCLINE","MINOCYCLINE","TETRACYCLINE"}
QUINOLONE = {"CIPROFLOXACIN","LEVOFLOXACIN","NALIDIXICACID","NORFLOXACIN","NOVOBIOCIN"}
FOLATE = {"SULFAMETHIZOLE","SULFAMETHOXAZOLE","SULFAMONOMETHOXINE","TRIMETHOPRIM"}
CELL_WALL_OTHER = {"BACITRACIN","CERULENIN","CYCLOSERINED","FOSFOMYCIN","VANCOMYCIN"}
DNA_DAMAGE = {"BLEOMYCIN","CISPLATIN","DOXORUBICIN","MITOMYCINC","NITROFURANTOIN","PHLEOMYCIN"}
MEMBRANE = {"POLYMYXINB","COLISTIN","FUSIDICACID"}
RNA_POL = {"RIFAMPICIN"}
TRANSLATION = {"PUROMYCIN","CHLORAMPHENICOL","CHIR090"}
PROTONOPHORE = {"CCCP"}
OTHER = {"VERAPAMIL","PARAQUAT","PYOCYANIN","TRICLOSAN","INDOLICIDIN","DIBUCAINE",
    "THEOPHYLLINE","ACRIFLAVINE","ETHIDIUMBROMIDE","PROPIDIUMIODIDE","CALCOFLUOR",
    "EGCG","NOREPINEPHRINE","EPINEPHRINE","CHOLATE","DEOXYCHOLATE","TAUROCHOLATE",
    "N-ACETYLGLUCOSAMINE","GLUCOSAMINE","SUCCINATE","HIGHCOPPER","ANAEROBIC"}

def classify(drug):
    for cls, dset in {"β-lactam":BETA_LACTAM,"Aminoglycoside":AMINOGLYCOSIDE,
        "Macrolide":MACROLIDE,"Tetracycline":TETRACYCLINE,"Quinolone":QUINOLONE,
        "Folate":FOLATE,"CellWall_Other":CELL_WALL_OTHER,"DNA_Damage":DNA_DAMAGE,
        "Membrane":MEMBRANE,"RNA_Pol":RNA_POL,"Translation":TRANSLATION,
        "Protonophore":PROTONOPHORE,"Other":OTHER}.items():
        if drug in dset: return cls
    return "Unknown"

# ---- Bactericidal vs Bacteriostatic ----
CIDAL = BETA_LACTAM | AMINOGLYCOSIDE | QUINOLONE | DNA_DAMAGE | MEMBRANE | {"VANCOMYCIN","FOSFOMYCIN","POLYMYXINB","COLISTIN","NITROFURANTOIN","RIFAMPICIN","BACITRACIN"}
STATIC = MACROLIDE | TETRACYCLINE | FOLATE | {"CHLORAMPHENICOL","PUROMYCIN","ERYTHROMYCIN","AZITHROMYCIN","CLARITHROMYCIN","SPIRAMYCIN","SPECTINOMYCIN","STREPTOMYCIN","SULFAMETHIZOLE","SULFAMETHOXAZOLE","SULFAMONOMETHOXINE","TRIMETHOPRIM","CERULENIN"}

def cidal_static(drug):
    if drug in CIDAL: return "Cidal"
    if drug in STATIC: return "Static"
    return "?"

# ---- Same target check ----
# Ribosome 50S
S50 = {"ERYTHROMYCIN","AZITHROMYCIN","CLARITHROMYCIN","SPIRAMYCIN","CHLORAMPHENICOL","PUROMYCIN","FUSIDICACID"}
# Ribosome 30S
S30 = {"TETRACYCLINE","DOXYCYCLINE","MINOCYCLINE","GENTAMICIN","AMIKACIN","TOBRAMYCIN","SPECTINOMYCIN","STREPTOMYCIN"}
# Cell wall (peptidoglycan)
CW_PBP = {"AMOXICILLIN","AMPICILLIN","CEFACLOR","CEFOXITIN","CEFSULODIN","MECILLINAM","OXACILLIN","AZTREONAM","CARBENICILLIN"}
CW_OTHER = {"VANCOMYCIN","BACITRACIN","CYCLOSERINED","FOSFOMYCIN","CERULENIN"}
# DNA gyrase
GYR = {"CIPROFLOXACIN","LEVOFLOXACIN","NALIDIXICACID","NORFLOXACIN","NOVOBIOCIN"}

# ---- Build review ----
rows = []
for _, r in df.iterrows():
    d1, d2 = r["Drug_A"], r["Drug_B"]
    c1, c2 = classify(d1), classify(d2)
    cs1, cs2 = cidal_static(d1), cidal_static(d2)
    m2, m1 = r["M2_411_Proba"], r["M1_199_Proba"]
    tier = r["Tier"]

    # Flags
    flags = []
    issues = []
    recommend = ""

    # 1. Same target
    if d1 in S50 and d2 in S50: flags.append("同靶点_50S"); issues.append("竞争50S同一结合位点")
    if d1 in S30 and d2 in S30: flags.append("同靶点_30S"); issues.append("竞争30S同一结合位点")
    if d1 in CW_PBP and d2 in CW_PBP: flags.append("同靶点_PBP"); issues.append("竞争PBP同一靶点")
    if d1 in GYR and d2 in GYR: flags.append("同靶点_Gyrase"); issues.append("竞争DNA旋转酶")

    # 2. Cidal/Static conflict
    if cs1 == "Cidal" and cs2 == "Static": flags.append("杀菌_抑菌冲突"); issues.append(f"{d1}(杀菌)需要活跃生长,{d2}(抑菌)阻止生长")
    if cs2 == "Cidal" and cs1 == "Static": flags.append("杀菌_抑菌冲突"); issues.append(f"{d2}(杀菌)需要活跃生长,{d1}(抑菌)阻止生长")

    # 3. Both anticancer/not antibiotics
    non_antibiotic = {"DOXORUBICIN","BLEOMYCIN","CISPLATIN","METHOTREXATE","AZIDOTHYMIDINE"}
    if d1 in non_antibiotic or d2 in non_antibiotic:
        flags.append("非抗菌药物"); issues.append("抗癌药/非抗生素,抗菌协同意义有限")

    # 4. Protonophore/stressor
    if d1 in PROTONOPHORE or d2 in PROTONOPHORE:
        flags.append("质子载体胁迫"); issues.append("CCCP不是治疗药物,是实验工具药")

    if d1 in {"PARAQUAT","PYOCYANIN"} or d2 in {"PARAQUAT","PYOCYANIN"}:
        flags.append("氧化胁迫剂"); issues.append("氧化胁迫工具药,非临床抗生素")

    # 5. Different species targets
    g_neg_only = {"CHIR090","AZTREONAM","COLISTIN"}
    g_pos_only = {"VANCOMYCIN","BACITRACIN","DAPTOMYCIN"}
    if (d1 in g_neg_only and d2 in g_pos_only) or (d2 in g_neg_only and d1 in g_pos_only):
        flags.append("靶向不同菌种"); issues.append(f"G⁻专属药+G⁺专属药,同株菌无法同时作用")

    # 6. Known antagonism mechanism
    if "RIFAMPICIN" in (d1,d2) and ("MACROLIDE" in f"{c1}{c2}" or "CHLORAMPHENICOL" in (d1,d2)):
        flags.append("已知药物拮抗"); issues.append("利福平诱导代谢/外排,降低合用药浓度")

    # 7. PHLEOMYCIN is bleomycin-family (anticancer, not antibacterial)
    if "PHLEOMYCIN" in (d1,d2):
        flags.append("抗癌抗生素"); issues.append("博来霉素类抗癌药,非抗菌治疗药物")

    # 8. Very likely true synergy (scoring positive)
    if not flags:
        if c1 != c2 and cs1 == "Cidal" and cs2 == "Cidal":
            recommend = "★★★ 协同可能性高"; issues.append("两类杀菌剂,不同靶点,机制协同")
        elif c1 != c2 and "同靶点" not in " ".join(flags):
            recommend = "★★ 协同可能性中"; issues.append("不同类别,可能协同")
        elif c1 == c2 and "同靶点" not in " ".join(flags):
            recommend = "★★ 同类别,需验证"; issues.append(f"同为{c1},但不是同一靶点")

    # M2 vs M1 flag
    m_flag = ""
    if m2 >= 0.99 and m1 >= 0.99: m_flag = "双模型满分"
    elif m2 >= 0.99: m_flag = "M2满分_M1中等"
    elif m2 <= 0.01: m_flag = "M2极低_M2判拮抗"
    elif 0.30 <= m2 <= 0.70: m_flag = "M2不确定(中间态)"

    issue_text = "; ".join(issues) if issues else ""
    flag_text = ", ".join(flags) if flags else ""

    rows.append({
        "Drug_Pair": r["Drug_Pair"],
        "Drug_A": d1, "Drug_B": d2,
        "Class_A": c1, "Class_B": c2,
        "Action_A": cs1, "Action_B": cs2,
        "Tier": tier,
        "M2_411_Proba": m2,
        "M1_199_Proba": m1,
        "Model_Signal": m_flag,
        "Flags": flag_text,
        "Issues": issue_text,
        "Recommendation": recommend,
        "TrainFreq_A": r["TrainFreq_A"],
        "TrainFreq_B": r["TrainFreq_B"],
        "SuppTable2": r.get("SuppTable2","?"),
    })

review = pd.DataFrame(rows)

# Sort: clean first, then flagged
review["_sort"] = review["Flags"].apply(lambda x: 0 if x == "" else (1 if "同靶点" in x else 2 if "杀菌_抑菌" in x else 3))
review = review.sort_values(["_sort", "Tier", "M2_411_Proba"], ascending=[True, True, False]).drop(columns="_sort")

# Save
OUT = r"../data/predictions_all_combos"

with pd.ExcelWriter(os.path.join(OUT,"prescreened_candidates_reviewed.xlsx"), engine="openpyxl") as w:
    # Sheet 1: Clean recommendations
    clean = review[review["Flags"] == ""]
    clean.to_excel(w, sheet_name="Clean_Predictions", index=False)

    # Sheet 2: All with review
    review.to_excel(w, sheet_name="All_Reviewed", index=False)

    # Sheet 3: By tier
    for tier in ["A_Synergy_HiConf","B_Synergy_MedConf","C3_Antag_Loose","D_Borderline"]:
        sub = review[review["Tier"] == tier]
        if len(sub) > 0:
            sub.to_excel(w, sheet_name=tier[:31], index=False)

review.to_csv(os.path.join(OUT,"prescreened_candidates_reviewed.csv"), index=False)

# Console report
print(f"Total candidates: {len(review)}")
print(f"  Clean (no flags): {len(clean)}")
print(f"  Same target conflict: {(review['Flags'].str.contains('同靶点')).sum()}")
print(f"  Cidal/Static conflict: {(review['Flags'].str.contains('杀菌_抑菌')).sum()}")
print(f"  Non-antibiotic: {(review['Flags'].str.contains('非抗菌药物')).sum()}")
print(f"  Protonophore/stress: {(review['Flags'].str.contains('质子载体|氧化胁迫')).sum()}")
print(f"  Cross-species target: {(review['Flags'].str.contains('不同菌种')).sum()}")
print(f"  Known antagonism: {(review['Flags'].str.contains('已知药物拮抗')).sum()}")
print(f"  Anticancer: {(review['Flags'].str.contains('抗癌抗生素')).sum()}")

print("\n=== CLEAN PREDICTIONS (most promising for wet-lab) ===")
for i,(_,r) in enumerate(clean.iterrows(),1):
    print(f"{i:>2}. {r['Drug_Pair']:<50s} M2={r['M2_411_Proba']:.4f}  {r['Class_A']}+{r['Class_B']}  {r['Action_A']}/{r['Action_B']}  {r['Recommendation']}")
    if i >= 40: break

print("\n=== FLAGGED — Same Target ===")
st = review[review["Flags"].str.contains("同靶点")]
for i,(_,r) in enumerate(st.iterrows(),1):
    print(f"  {i}. {r['Drug_Pair']:<50s} {r['Flags']} → {r['Issues']}")

print("\n=== FLAGGED — Cidal/Static ===")
cs = review[review["Flags"].str.contains("杀菌_抑菌")]
for i,(_,r) in enumerate(cs.iterrows(),1):
    print(f"  {i}. {r['Drug_Pair']:<50s} {r['Flags']} → {r['Issues']}")

print(f"\nSaved: {os.path.join(OUT,'prescreened_candidates_reviewed.xlsx')}")
