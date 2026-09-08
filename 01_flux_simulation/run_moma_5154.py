# =========================================================
# Run MOMA for 5154 combos that have gene files in gene_combo-2
# Uses the same MOMA algorithm as flux_ml_pipeline.py
# Saves each combo's _FLUX.csv to continuous_bounded_moma_all/
# --- BATCH MODE: exits after BATCH_SIZE combos to avoid OOM ---
# =========================================================
import os, sys, gc, traceback, warnings
import numpy as np, pandas as pd, cobra
from cobra.flux_analysis import pfba, moma

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ---- CONFIG ----
MODEL_FILE  = r"../models/iJO1366.json"
GENE_DIR    = r"../data/gene_combo-2"
OUTPUT_DIR  = r"../data/continuous_bounded_moma_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_ACTIVITY = 0.5
MAX_ACTIVITY = 1.0
ACTIVITY_THRESHOLD = 0.8
WT_FLUX_THRESHOLD = 0.01
SLACK_FACTOR = 0.5
CHANGE_THRESHOLD = 1e-6
PREFERRED_SOLVER = "gurobi"

BATCH_SIZE = 300  # exit after this many combos, wrapper restarts with fresh memory

# ---- GPR Parser (same as flux_ml_pipeline.py) ----
class GPRParser:
    def __init__(self, rule: str):
        self.tokens = self._tokenize(rule)
        self.pos = 0
        self.gene_activity = {}
    def _tokenize(self, rule):
        s = str(rule).lower().strip()
        s = s.replace("(", " ( ").replace(")", " ) ").replace(",", " ")
        return [t for t in s.split() if t]
    def evaluate(self, gene_activity):
        if not self.tokens: return 1.0
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
        if self.pos >= len(self.tokens): return 1.0
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
        local_activity = {}
        for g in rxn.genes:
            gid = g.id.lower()
            local_activity[gid] = gene_activity.get(gid, 1.0)
        try:
            parser = GPRParser(rule)
            rxn_activity[rxn.id] = parser.evaluate(local_activity)
        except:
            rxn_activity[rxn.id] = 1.0
    return rxn_activity

# ---- Determine combos to run ----
already_done = set(f.replace("_FLUX.csv","") for f in os.listdir(OUTPUT_DIR) if f.endswith("_FLUX.csv"))
gene_files = [f.replace(".csv","") for f in os.listdir(GENE_DIR) if f.endswith(".csv") and "__" in f]
to_run = sorted([c for c in gene_files if c not in already_done])

print(f"Already done: {len(already_done)}")
print(f"Total gene combos available: {len(gene_files)}")
print(f"To run: {len(to_run)}")
print(f"Batch size: {BATCH_SIZE}")

if len(to_run) == 0:
    print("All done!")
    sys.exit(0)

# ---- Limit to batch ----
to_run = to_run[:BATCH_SIZE]
print(f"This batch: {len(to_run)}")

# ---- Load model & baseline (once) ----
print("\nLoading model...")
model = cobra.io.load_json_model(MODEL_FILE)
try:
    model.solver = "gurobi"
    print("Gurobi OK")
except:
    print("Default solver")

biomass_rxn = [r.id for r in model.reactions if r.objective_coefficient != 0][0]
baseline = pfba(model)
baseline_flux = pd.Series(baseline.fluxes, name="Flux_baseline")
wt_growth = baseline.fluxes[biomass_rxn]
reaction_map = {r.id: (r.name, r.subsystem) for r in model.reactions}

print(f"\nWT growth: {wt_growth:.6f}")
print(f"Biomass: {biomass_rxn}")

# ---- Process each combo ----
error_count = 0
skipped_common = 0
for i, combo_name in enumerate(to_run):
    try:
        if (i+1) % 50 == 0:
            print(f"\n--- Progress: {i+1}/{len(to_run)} ({100*(i+1)/len(to_run):.1f}%) ---")

        # Load gene fitness
        path = os.path.join(GENE_DIR, f"{combo_name}.csv")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        gene_col = df.columns[0]
        value_col = df.columns[1]
        df[gene_col] = df[gene_col].astype(str).str.strip().str.lower()
        df = df[df[gene_col].str.startswith("b")]
        df["fitness"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["fitness"])

        gene_activity = {row[gene_col]: fitness_to_activity(row["fitness"]) for _, row in df.iterrows()}
        model_gene_ids = {g.id.lower() for g in model.genes}
        common = model_gene_ids & set(gene_activity.keys())
        if len(common) == 0:
            skipped_common += 1
            continue

        local_model = model.copy()
        rxn_activity = gene_to_rxn_activity(local_model, gene_activity)

        for rxn in local_model.reactions:
            activity = rxn_activity.get(rxn.id, 1.0)
            if activity >= ACTIVITY_THRESHOLD:
                continue
            wt_flux = baseline.fluxes[rxn.id]
            if abs(wt_flux) < WT_FLUX_THRESHOLD:
                continue

            bound_ratio = SLACK_FACTOR + (1.0 - SLACK_FACTOR) * activity
            if wt_flux > 0:
                new_upper = wt_flux * bound_ratio
                rxn.upper_bound = min(rxn.upper_bound, max(new_upper, 1e-6))
            else:
                new_lower = wt_flux * bound_ratio
                rxn.lower_bound = max(rxn.lower_bound, min(new_lower, -1e-6))

        sol = moma(local_model, solution=baseline)
        if sol.status != "optimal":
            error_count += 1
            del local_model
            gc.collect()
            continue

        cond_flux = pd.Series(sol.fluxes, name="Flux_condition")
        out = pd.DataFrame({"Reaction": [r.id for r in local_model.reactions]})
        out["Name"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[0])
        out["Subsystem"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[1])
        out = out.merge(baseline_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
        out = out.merge(cond_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
        out["ΔFlux"] = out["Flux_condition"] - out["Flux_baseline"]
        out["Abs_ΔFlux"] = out["ΔFlux"].abs()
        out["Growth_Rate"] = sol.fluxes[biomass_rxn]
        out["WT_Growth"] = wt_growth
        out["Growth_Ratio"] = out["Growth_Rate"] / wt_growth if wt_growth else np.nan
        out["Solution_Status"] = sol.status

        out.to_csv(os.path.join(OUTPUT_DIR, f"{combo_name}_FLUX.csv"), index=False)

        # --- CRITICAL: release C-level Gurobi model memory ---
        del local_model, sol, cond_flux, out
        gc.collect()

    except Exception:
        error_count += 1
        try: del local_model
        except: pass
        gc.collect()

print(f"\nBATCH DONE. Processed {len(to_run)} combos.")
print(f"Errors: {error_count}, Skipped (no common genes): {skipped_common}")
print(f"Remaining after this batch: {len(gene_files) - len(already_done) - len(to_run)}")
# Exit with 0 so wrapper can restart for next batch
