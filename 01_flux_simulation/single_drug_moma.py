# %%
# =========================================================
# SINGLE-DRUG MOMA + NON-ADDITIVE EFFECT
#
# Step 1: Select middle-dose single-drug gene-fitness CSVs
# Step 2: Run MOMA (same algorithm as combo pipeline)
# Step 3: Compute non-additive effect:
#         NonAdditive_ΔFlux = Combo_ΔFlux - Single_A_ΔFlux - Single_B_ΔFlux
# =========================================================

import os
import re
import shutil
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

# --- Paths ---
MODEL_FILE        = r"../models/iJO1366.json"
SINGLE_INPUT_DIR  = r"../data/mapped-gene-second"     # raw multi-dose CSVs
SINGLE_SELECTED   = r"../data/single-moma-input"      # middle-dose CSVs
SINGLE_OUTPUT_DIR = r"../data/continuous_bounded_moma_single"  # MOMA results
COMBO_OUTPUT_DIR  = r"../data/continuous_bounded_moma_full-new" # combo MOMA results (existing)
OUTPUT_MATRIX_DIR = r"../data/non_additive_matrix"                # final ML matrices

for d in [SINGLE_SELECTED, SINGLE_OUTPUT_DIR, OUTPUT_MATRIX_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Solver ---
PREFERRED_SOLVER = "gurobi"

# --- MOMA parameters (identical to combo pipeline) ---
MIN_ACTIVITY       = 0.5
MAX_ACTIVITY       = 1.0
ACTIVITY_THRESHOLD = 0.8
WT_FLUX_THRESHOLD  = 0.01
SLACK_FACTOR       = 0.5
CHANGE_THRESHOLD   = 1e-6

# --- Middle-dose selection ---
KEEP_UNMATCHED     = True   # keep drugs whose name doesn't end with '-DIGIT'

# =========================================================
# STEP 0: Load model + WT baseline (shared)
# =========================================================

import cobra
from cobra.flux_analysis import pfba, moma

print("=" * 70)
print("  Loading model and computing WT baseline ...")
print("=" * 70)
model = cobra.io.load_json_model(MODEL_FILE)

# Solver setup
def configure_solver(model, preferred="gurobi"):
    qp_capable = {"gurobi", "cplex", "osqp", "quad", "ipopt", "mosek"}
    candidates = [preferred, "osqp", "cplex", "gurobi"]
    current = str(model.solver).lower()
    if any(qp in current for qp in qp_capable):
        print(f"  Solver           : {model.solver}  (QP-capable)")
        return True
    for c in candidates:
        try:
            model.solver = c
            print(f"  Solver           : {model.solver}  (QP-capable)")
            return True
        except Exception:
            continue
    print("  !  No QP solver - MOMA may fail.")
    return False

configure_solver(model, PREFERRED_SOLVER)

# WT baseline
baseline = pfba(model)
baseline_flux = pd.Series(baseline.fluxes, name="Flux_baseline")

objective_rxns = [r.id for r in model.reactions if r.objective_coefficient != 0]
biomass_rxn = objective_rxns[0]
wt_growth = baseline.fluxes[biomass_rxn]
print(f"  Biomass rxn     : {biomass_rxn}")
print(f"  WT growth       : {wt_growth:.6f}")

# Reaction map
reaction_map = {r.id: (r.name, r.subsystem) for r in model.reactions}


# =========================================================
# UTILITY: GPR parser (same as pipeline)
# =========================================================

class GPRParser:
    def __init__(self, rule: str):
        self.tokens = self._tokenize(rule)
        self.pos = 0
        self.gene_activity = {}

    def _tokenize(self, rule: str):
        s = str(rule).lower().strip()
        s = s.replace("(", " ( ").replace(")", " ) ").replace(",", " ")
        return [t for t in s.split() if t]

    def evaluate(self, gene_activity):
        if not self.tokens:
            return 1.0
        self.pos = 0
        self.gene_activity = gene_activity
        return self._expr()

    def _expr(self):
        left = self._term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "or":
            self.pos += 1
            right = self._term()
            left = max(left, right)
        return left

    def _term(self):
        left = self._factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "and":
            self.pos += 1
            right = self._factor()
            left = min(left, right)
        return left

    def _factor(self):
        if self.pos >= len(self.tokens):
            return 1.0
        tok = self.tokens[self.pos]
        if tok == "(":
            self.pos += 1
            val = self._expr()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ")":
                self.pos += 1
            return val
        self.pos += 1
        return self.gene_activity.get(tok, 1.0)


def fitness_to_activity(fitness):
    sigmoid = 1.0 / (1.0 + np.exp(-fitness))
    return MIN_ACTIVITY + (MAX_ACTIVITY - MIN_ACTIVITY) * sigmoid


def gene_to_rxn_activity(model, gene_activity):
    rxn_activity = {}
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if not rule or not rxn.genes:
            rxn_activity[rxn.id] = 1.0
            continue
        local_act = {}
        for g in rxn.genes:
            gid = g.id.lower()
            local_act[gid] = gene_activity.get(gid, 1.0)
        try:
            score = GPRParser(rule).evaluate(local_act)
        except Exception:
            score = 1.0
        rxn_activity[rxn.id] = score
    return rxn_activity


# =========================================================
# STEP 1: SELECT MIDDLE-DOSE SINGLE DRUGS
# =========================================================

print("\n" + "=" * 70)
print("  Step 1: Select middle-dose single drugs")
print("=" * 70)

def parse_drug_and_dose(fname):
    """Parse drug name and numeric dose from filename."""
    name = os.path.splitext(fname)[0].upper().strip()
    # Pattern: DRUG-XX.XX maybe with trailing %
    match = re.match(r"^(.+)-([\d]+(?:\.[\d]+)?)%?$", name)
    if match:
        drug = match.group(1)
        dose = float(match.group(2))
        return drug, dose
    # No dose suffix
    return name, 0.0


def is_combo_file(drug_name):
    """Detect combo files (double underscore or patterns suggesting 2 drugs)."""
    # Combo files have '__' in the original data
    if "__" in drug_name:
        return True
    return False


# Scan all CSVs
drug_dict = {}
combo_skipped = 0

for f in os.listdir(SINGLE_INPUT_DIR):
    if not f.endswith(".csv"):
        continue
    try:
        drug, dose = parse_drug_and_dose(f)
    except Exception:
        continue

    if is_combo_file(drug):
        combo_skipped += 1
        continue

    drug_dict.setdefault(drug, []).append((f, dose))

# Select middle dose for each drug
def select_middle(lst):
    lst = sorted(lst, key=lambda x: x[1])
    n = len(lst)
    if n == 1:
        return lst[0]
    elif n == 2:
        return lst[0]   # take lower dose
    else:
        return lst[n // 2]

selected_count = 0
for drug, lst in sorted(drug_dict.items()):
    src_file, dose = select_middle(lst)
    src_path = os.path.join(SINGLE_INPUT_DIR, src_file)
    dst_name = f"{drug}.csv"
    dst_path = os.path.join(SINGLE_SELECTED, dst_name)

    if os.path.exists(dst_path):
        continue

    shutil.copy(src_path, dst_path)
    selected_count += 1
    print(f"  {drug:<25} -> {src_file}  (dose={dose})")

print(f"\n  Copied           : {selected_count} single-drug CSVs")
print(f"  Combo skipped    : {combo_skipped}")
print(f"  To               : {SINGLE_SELECTED}")


# =========================================================
# STEP 2: RUN MOMA ON SINGLE DRUGS
# =========================================================

print("\n" + "=" * 70)
print("  Step 2: Run MOMA on single drugs")
print("=" * 70)


def process_single_drug(fname: str):
    """Run MOMA for a single-drug CSV (same logic as combo pipeline)."""
    try:
        fname_clean = fname.replace(".csv", "")
        print(f"\n  [FILE]  {fname_clean}")

        path = os.path.join(SINGLE_SELECTED, fname)

        # Load CSV
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        gene_col = df.columns[0]
        value_col = df.columns[1]

        df[gene_col] = df[gene_col].astype(str).str.strip().str.lower()
        df = df[df[gene_col].str.startswith("b")]
        df["fitness"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["fitness"])
        print(f"  Valid genes      : {len(df)}")

        # Gene -> activity
        gene_activity = {row[gene_col]: fitness_to_activity(row["fitness"])
                         for _, row in df.iterrows()}

        # Match model genes
        model_gene_ids = {g.id.lower() for g in model.genes}
        common = model_gene_ids & set(gene_activity.keys())
        if len(common) == 0:
            print("  x  No matched genes — skipping.")
            return fname_clean, None

        # Copy model & apply perturbation
        local_model = model.copy()
        rxn_activity = gene_to_rxn_activity(local_model, gene_activity)

        perturbed_rxns = 0
        for rxn in local_model.reactions:
            activity = rxn_activity.get(rxn.id, 1.0)
            if activity >= ACTIVITY_THRESHOLD:
                continue
            wt_flux = baseline.fluxes[rxn.id]
            if abs(wt_flux) < WT_FLUX_THRESHOLD:
                continue
            perturbed_rxns += 1
            bound_ratio = SLACK_FACTOR + (1.0 - SLACK_FACTOR) * activity
            if wt_flux > 0:
                rxn.upper_bound = min(rxn.upper_bound, max(wt_flux * bound_ratio, 1e-6))
            else:
                rxn.lower_bound = max(rxn.lower_bound, min(wt_flux * bound_ratio, -1e-6))

        # Run MOMA
        sol = moma(local_model, solution=baseline)

        # Build output
        cond_flux = pd.Series(sol.fluxes, name="Flux_condition")
        true_growth = sol.fluxes[biomass_rxn]
        out = pd.DataFrame({"Reaction": [r.id for r in local_model.reactions]})
        out["Name"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[0])
        out["Subsystem"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[1])
        out = out.merge(baseline_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
        out = out.merge(cond_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
        out["ΔFlux"] = out["Flux_condition"] - out["Flux_baseline"]
        out["Abs_ΔFlux"] = out["ΔFlux"].abs()
        out["Growth_Rate"] = true_growth
        out["WT_Growth"] = wt_growth
        out["Growth_Ratio"] = true_growth / wt_growth if wt_growth else np.nan

        # Save
        out_path = os.path.join(SINGLE_OUTPUT_DIR, f"{fname_clean}_FLUX.csv")
        out.to_csv(out_path, index=False)
        print(f"  Perturbed rxns   : {perturbed_rxns}")
        print(f"  Growth ratio     : {true_growth / wt_growth:.4f}" if wt_growth else "")
        print(f"  OK  Saved")

        return fname_clean, out

    except Exception as e:
        print(f"  x  Failed: {e}")
        traceback.print_exc()
        return fname.replace(".csv", ""), None


single_flux_results = {}  # drug_name -> ΔFlux Series

files = sorted([f for f in os.listdir(SINGLE_SELECTED) if f.endswith(".csv")])
print(f"\n  Total single-drug CSVs : {len(files)}")

for i, fname in enumerate(files):
    print(f"\n{'─' * 60}")
    print(f"  [{i+1}/{len(files)}]")
    drug_name, out_df = process_single_drug(fname)
    if out_df is not None:
        single_flux_results[drug_name] = out_df.set_index("Reaction")["ΔFlux"]

print(f"\n{'=' * 70}")
print(f"  OK  MOMA complete for {len(single_flux_results)} / {len(files)} single drugs")
print(f"{'=' * 70}")


# =========================================================
# STEP 3: BUILD NON-ADDITIVE EFFECT MATRIX
# =========================================================

print("\n" + "=" * 70)
print("  Step 3: Compute non-additive effect matrix")
print("=" * 70)

# Load combo drug names from metadata or flux files
COMBO_LABEL_FILE = r"../data/metadata.csv"

if os.path.exists(COMBO_LABEL_FILE):
    meta = pd.read_csv(COMBO_LABEL_FILE)
    meta.columns = meta.columns.str.strip().str.lower()
    if "drug_pair" in meta.columns and "label" in meta.columns:
        # Normalise: hyphens -> underscores for matching flux files
        meta["drug_pair"] = meta["drug_pair"].astype(str).str.strip().str.upper().str.replace("-", "_")
    combo_names = meta["drug_pair"].unique()
    label_map = dict(zip(meta["drug_pair"], meta["label"]))
    print(f"  Combo pairs from metadata : {len(combo_names)}")
else:
    # Fallback: scan combo output directory
    combo_names = sorted(set(
        f.replace("_FLUX.csv", "")
        for f in os.listdir(COMBO_OUTPUT_DIR)
        if f.endswith("_FLUX.csv")
    ))
    label_map = None
    print(f"  Combo pairs from flux dir : {len(combo_names)}")

non_additive_rows = []

for pair in combo_names:
    # Parse drug A and B (format: AMIKACIN__AMOXICILLIN, double underscore)
    parts = pair.split("__")
    if len(parts) != 2:
        # Fallback for edge cases with single/different separators
        print(f"  !  Cannot parse combo pair: {pair} — skipping")
        continue
    drug_a, drug_b = parts[0], parts[1]

    # Try to find matching single-drug results
    # (allow fuzzy match in case of naming differences)
    def find_single(name):
        """Match drug name to single-drug results (case-insensitive)."""
        name_upper = name.upper()
        # Exact match
        if name_upper in single_flux_results:
            return single_flux_results[name_upper]
        # Check if name exists as a key (already uppercase from filename)
        for key in single_flux_results:
            if key.upper() == name_upper:
                return single_flux_results[key]
        # Try without known problematic suffixes
        return None

    flux_a = find_single(drug_a)
    flux_b = find_single(drug_b)

    # Load combo DeltaFlux (CSV)
    combo_path = os.path.join(COMBO_OUTPUT_DIR, f"{pair}_FLUX.csv")
    if not os.path.exists(combo_path):
        continue
    combo_df = pd.read_csv(combo_path)
    if "ΔFlux" not in combo_df.columns:
        continue
    combo_flux = combo_df.set_index("Reaction")["ΔFlux"]

    # If both single drugs found, compute non-additive effect
    if flux_a is not None and flux_b is not None:
        Δ_non_additive = combo_flux - flux_a - flux_b
    elif flux_a is not None:
        Δ_non_additive = combo_flux - flux_a
    elif flux_b is not None:
        Δ_non_additive = combo_flux - flux_b
    else:
        # Neither single drug found — use raw combo ΔFlux
        print(f"  !  No singles for {pair} ({drug_a}, {drug_b}) — using raw ΔFlux")
        Δ_non_additive = combo_flux

    row = {"drug_pair": pair}
    for rxn_id, val in Δ_non_additive.items():
        row[str(rxn_id)] = val

    # Attach label if available
    if label_map is not None and pair in label_map:
        row["label"] = label_map[pair]

    non_additive_rows.append(row)

# Build matrix
X_na = pd.DataFrame(non_additive_rows).fillna(0)
print(f"\n  Non-additive matrix shape : {X_na.shape}")

# --- Save raw non-additive matrix ---
na_raw_path = os.path.join(OUTPUT_MATRIX_DIR, "non_additive_raw.csv")
X_na.to_csv(na_raw_path, index=False)
print(f"  Saved            : {na_raw_path}")

# --- Save normalised non-additive matrix ---
id_cols = [c for c in ["drug_pair", "label"] if c in X_na.columns]
feat_cols = [c for c in X_na.columns if c not in id_cols]

# Zero-variance filter
stds = X_na[feat_cols].std()
feat_cols_std = stds[stds > 0].index.tolist()

# L1 row normalisation
X_norm_vals = X_na[feat_cols_std].div(
    X_na[feat_cols_std].abs().sum(axis=1), axis=0
).fillna(0)
X_norm = pd.concat([X_na[id_cols].reset_index(drop=True), X_norm_vals], axis=1)
na_norm_path = os.path.join(OUTPUT_MATRIX_DIR, "non_additive_normalized.csv")
X_norm.to_csv(na_norm_path, index=False)
print(f"  Saved normalized : {na_norm_path}")

# --- Label distribution ---
if "label" in X_na.columns:
    print(f"\n  Label distribution:")
    print(f"    synergy=1  : {(X_na['label'] == 1).sum()}")
    print(f"    non-synergy=0: {(X_na['label'] == 0).sum()}")
    print(f"    unknown     : {X_na['label'].isna().sum()}")

# --- Summary ---
print(f"\n  OK  Single-drug MOMA + non-additive effect complete.")
print(f"\n  Outputs:")
print(f"    Single MOMA results     : {SINGLE_OUTPUT_DIR}")
print(f"    Non-additive matrix (raw)      : {na_raw_path}")
print(f"    Non-additive matrix (norm)     : {na_norm_path}")
print(f"\n  {'=' * 70}")
print(f"  Done  All done.")
print(f"  {'=' * 70}")
