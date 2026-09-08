# %%
# Run MOMA for 212 new combos only, using existing results for the other 147
import os, re, traceback, warnings
import numpy as np, pandas as pd, cobra
from cobra.flux_analysis import pfba, moma
warnings.filterwarnings("ignore")

# ---- CONFIG ----
MODEL_FILE  = r"../models/iJO1366.json"
INPUT_DIR   = r"../data/gene-combo-all"
OUTPUT_DIR  = r"../data/continuous_bounded_moma_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_ACTIVITY = 0.5; MAX_ACTIVITY = 1.0; ACTIVITY_THRESHOLD = 0.8
WT_FLUX_THRESHOLD = 0.01; SLACK_FACTOR = 0.5; CHANGE_THRESHOLD = 1e-6

# ---- Determine which combos need MOMA ----
all_input = set(f.replace('.csv','') for f in os.listdir(INPUT_DIR) if f.endswith('.csv'))
already_done = set(f.replace('_FLUX.csv','') for f in os.listdir(OUTPUT_DIR) if f.endswith('_FLUX.csv'))
to_run = sorted(all_input - already_done)

print(f'All combos:   {len(all_input)}')
print(f'Already done: {len(already_done)}')
print(f'To run:       {len(to_run)}')

if len(to_run) == 0:
    print('Nothing to run! All done.')
    exit(0)

# ---- Load model & baseline (once) ----
print('\nLoading model...')
model = cobra.io.load_json_model(MODEL_FILE)
try: model.solver = "gurobi"; print('Gurobi OK')
except: print('Default solver')

biomass_rxn = [r.id for r in model.reactions if r.objective_coefficient != 0][0]
baseline = pfba(model)
baseline_flux = pd.Series(baseline.fluxes, name="Flux_baseline")
wt_growth = baseline.fluxes[biomass_rxn]
reaction_map = {r.id: (r.name, r.subsystem) for r in model.reactions}

# ---- GPR parser ----
class GPRParser:
    def __init__(self, rule): self.tokens = self._tokenize(rule); self.pos = 0; self.gene_activity = {}
    def _tokenize(self, rule):
        s = str(rule).lower().strip().replace("("," ( ").replace(")"," ) ").replace(","," ")
        return [t for t in s.split() if t]
    def evaluate(self, ga):
        if not self.tokens: return 1.0
        self.pos = 0; self.gene_activity = ga; return self._expr()
    def _expr(self):
        left = self._term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "or": self.pos += 1; left = max(left, self._term())
        return left
    def _term(self):
        left = self._factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "and": self.pos += 1; left = min(left, self._factor())
        return left
    def _factor(self):
        if self.pos >= len(self.tokens): return 1.0
        tok = self.tokens[self.pos]
        if tok == "(": self.pos += 1; v = self._expr(); self.pos += (self.pos < len(self.tokens) and self.tokens[self.pos] == ")"); return v
        self.pos += 1; return self.gene_activity.get(tok, 1.0)

def fitness_to_activity(f):
    s = 1.0 / (1.0 + np.exp(-f))
    return MIN_ACTIVITY + (MAX_ACTIVITY - MIN_ACTIVITY) * s

def gene_to_rxn_activity(model, ga):
    ra = {}
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if not rule or not rxn.genes: ra[rxn.id] = 1.0; continue
        la = {}
        for g in rxn.genes: gid = g.id.lower(); la[gid] = ga.get(gid, 1.0)
        try: ra[rxn.id] = GPRParser(rule).evaluate(la)
        except: ra[rxn.id] = 1.0
    return ra

# ---- Process each ----
for idx, fname in enumerate(to_run):
    fname_csv = f'{fname}.csv'
    print(f'\n[{(idx+1)}/{(len(to_run))}] {fname}')

    df = pd.read_csv(os.path.join(INPUT_DIR, fname_csv))
    df.columns = df.columns.str.strip()
    gene_col = df.columns[0]; value_col = df.columns[1]
    df[gene_col] = df[gene_col].astype(str).str.strip().str.lower()
    df = df[df[gene_col].str.startswith("b")]
    df["fitness"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["fitness"])

    gene_activity = {row[gene_col]: fitness_to_activity(row["fitness"]) for _, row in df.iterrows()}
    common = {g.id.lower() for g in model.genes} & set(gene_activity)
    if not common: print('  No matched genes'); continue

    local_model = model.copy()
    rxn_activity = gene_to_rxn_activity(local_model, gene_activity)

    perturbed = 0
    for rxn in local_model.reactions:
        activity = rxn_activity.get(rxn.id, 1.0)
        if activity >= ACTIVITY_THRESHOLD: continue
        wf = baseline.fluxes[rxn.id]
        if abs(wf) < WT_FLUX_THRESHOLD: continue
        perturbed += 1
        br = SLACK_FACTOR + (1.0 - SLACK_FACTOR) * activity
        if wf > 0: rxn.upper_bound = min(rxn.upper_bound, max(wf * br, 1e-6))
        else: rxn.lower_bound = max(rxn.lower_bound, min(wf * br, -1e-6))

    try:
        sol = moma(local_model, solution=baseline)
    except Exception as e:
        print(f'  MOMA failed: {e}'); continue

    cond_flux = pd.Series(sol.fluxes, name="Flux_condition")
    tg = sol.fluxes[biomass_rxn]

    out = pd.DataFrame({"Reaction": [r.id for r in local_model.reactions]})
    out["Name"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[0])
    out["Subsystem"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[1])
    out = out.merge(baseline_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
    out = out.merge(cond_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
    out["DFlux"] = out["Flux_condition"] - out["Flux_baseline"]
    out["Abs_DFlux"] = out["DFlux"].abs()
    out["Reaction_Activity"] = out["Reaction"].map(rxn_activity).fillna(1.0)
    out["Growth_Rate"] = tg; out["WT_Growth"] = wt_growth
    out["Growth_Ratio"] = tg / wt_growth if wt_growth else np.nan
    out["MOMA_Objective"] = sol.objective_value; out["Solution_Status"] = sol.status

    out.to_csv(os.path.join(OUTPUT_DIR, f"{fname}_FLUX.csv"), index=False)

    ratio = tg / wt_growth if wt_growth else 0
    print(f'  Perturbed={perturbed}  Growth={tg:.4f}  Ratio={ratio:.4f}  OK')

print(f'\nDone. {len(to_run)} combos processed.')
