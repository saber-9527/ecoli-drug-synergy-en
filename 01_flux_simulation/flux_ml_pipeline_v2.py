# %%
# =========================================================
# MOMA -> FLUX FEATURE MATRIX PIPELINE  —  v2 (tighter bounds)
#
# MIN_ACTIVITY = 0.3, SLACK_FACTOR = 0.3
#   fitness -> activity   ∈ [0.3, 1.0]
#   activity -> bound_ratio = 0.3 + 0.7*activity  ∈ [0.3, 1.0]
#
# Compared to v1 (0.5/0.5): more aggressive perturbation,
#   weaker genes are more strongly inhibited.
# =========================================================
#
# Part A: MOMA simulation with soft gene-activity constraints
# Part B: Build ML-ready feature matrix from flux results
#
# Improvements over originals (5.21.ipynb + 6.4.ipynb):
#   1. Proper GPR boolean parsing (recursive descent)
#   2. Lower/dynamic WT_FLUX_THRESHOLD (captures secondary metabolism)
#   3. Proportional MIN_FREQUENCY (adapts to dataset size)
#   4. Both raw + row-normalized feature matrices (retain magnitude + pattern)
#   5. Comprehensive reaction_metadata.csv (for downstream enrichment)
#   6. Solver QP capability check (prevents silent MOMA failure)
#   7. Configurable SLACK_FACTOR + optional sensitivity analysis
#   8. Unified configuration block (single source of truth)
# =========================================================

import os
import re
import traceback
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import cobra
from cobra.flux_analysis import pfba, moma

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION  (single source of truth)
# =========================================================

# --- Paths ---
MODEL_FILE  = r"../models/iJO1366.json"
INPUT_DIR   = r"../data/gene-combo-all"
OUTPUT_DIR  = r"../data/continuous_bounded_moma_all_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Solver ---
PREFERRED_SOLVER = "gurobi"          # fallback chain: gurobi -> cplex -> osqp -> default

# --- Part A: MOMA parameters ---
MIN_ACTIVITY      = 0.3              # floor of sigmoid mapping [fitness -> activity]  (v2: tighter)
MAX_ACTIVITY      = 1.0              # ceiling
ACTIVITY_THRESHOLD = 0.8             # only constrain reactions with activity below this
WT_FLUX_THRESHOLD  = 0.01            # only constrain reactions with |WT flux| above this
                                     #   NOTE: 0.01 captures secondary metabolism
                                     #   (original 1.0 only caught central carbon)
SLACK_FACTOR      = 0.3              # residual capacity fraction when activity -> 0  (v2: tighter)
                                     #   activity=0 -> 30% retained  (continuous inhibition)
                                     #   activity=1 -> 100% retained (no constraint)
CHANGE_THRESHOLD  = 1e-6             # minimum |ΔFlux| to count as "changed"

# --- Part B: Feature matrix parameters ---
FEATURE_COLUMN     = "ΔFlux"
MIN_FREQUENCY_RATIO = 0.1            # keep reaction if changed in >= ratio of samples
MIN_FREQUENCY_ABSOLUTE = 5           #   but at least this many (adaptable to dataset)
OUTPUT_RAW         = True            # export flux_feature_matrix_raw.csv
OUTPUT_NORMALIZED  = True            # export flux_feature_matrix_normalized.csv
OUTPUT_METADATA    = True            # export reaction_metadata.csv

# --- Label / metadata ---
METADATA_FILE = r"../data/metadata_merged.csv"
# metadata label meanings:
#   1  -> synergistic (effective drug combination)
#   0  -> non-synergistic (ineffective drug combination)

# =========================================================
# IMPROVED GPR BOOLEAN PARSER
# =========================================================

class GPRParser:
    """
    Recursive-descent parser for GPR boolean expressions.

    Grammar:
        expr  ::= term ('or' term)*
        term  ::= factor ('and' factor)*
        factor ::= '(' expr ')' | gene_name

    Semantics:
        AND  -> min(activities)
        OR   -> max(activities)

    This correctly handles nested rules like:
        '(A or B) and C'   -> min(max(A,B), C)
        '(A and B) or C'   -> max(min(A,B), C)
        'A or (B and C)'   -> max(A, min(B,C))

    The original code (``if 'and' in rule: min() else max()``)
    treated '(A or B) and C' as min(A, B, C) — which is wrong.
    """

    def __init__(self, rule: str):
        self.tokens = self._tokenize(rule)
        self.pos = 0
        self.gene_activity: dict[str, float] = {}

    # -----------------------------------------------------------------
    def _tokenize(self, rule: str) -> list[str]:
        """Normalise spacing around parentheses and split into tokens."""
        s = str(rule).lower().strip()
        s = s.replace("(", " ( ").replace(")", " ) ")
        s = s.replace(",", " ")
        return [t for t in s.split() if t]

    # -----------------------------------------------------------------
    def evaluate(self, gene_activity: dict[str, float]) -> float:
        """Parse and evaluate the rule, returning a single activity score."""
        if not self.tokens:
            return 1.0
        self.pos = 0
        self.gene_activity = gene_activity
        return self._expr()

    # --- Recursive descent ---
    def _expr(self) -> float:
        left = self._term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "or":
            self.pos += 1
            right = self._term()
            left = max(left, right)
        return left

    def _term(self) -> float:
        left = self._factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "and":
            self.pos += 1
            right = self._factor()
            left = min(left, right)
        return left

    def _factor(self) -> float:
        if self.pos >= len(self.tokens):
            return 1.0
        tok = self.tokens[self.pos]
        if tok == "(":
            self.pos += 1
            val = self._expr()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ")":
                self.pos += 1
            return val
        # Gene name -> lookup activity (default 1.0 = no constraint)
        self.pos += 1
        return self.gene_activity.get(tok, 1.0)


# =========================================================
# UTILITY: GPR rule complexity stats
# =========================================================

def gpr_complexity_stats(model) -> dict:
    """
    Classify all reactions by GPR rule complexity.
    Useful for gauging how much the improved parser matters in practice.
    """
    simple, has_and, has_or, nested = 0, 0, 0, 0
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if not rule:
            continue
        if "(" in rule:
            nested += 1
        elif "and" in rule:
            has_and += 1
        elif "or" in rule:
            has_or += 1
        else:
            simple += 1
    total = simple + has_and + has_or + nested
    return {
        "total":               total,
        "simple (1 gene)":     f"{simple} ({simple/total*100:.1f}%)" if total else "0",
        "simple AND / OR":     f"{has_and + has_or} ({(has_and+has_or)/total*100:.1f}%)" if total else "0",
        "nested parentheses":  f"{nested} ({nested/total*100:.1f}%)" if total else "0",
    }


# =========================================================
# UTILITY: solver / QP check
# =========================================================

def configure_solver(model, preferred: str = "gurobi") -> bool:
    """
    Verify solver and attempt to switch to a QP-capable one.
    MOMA requires quadratic programming; GLPK/CBC do NOT support QP.

    Returns True if the active solver supports QP.
    """
    qp_capable_names = {"gurobi", "cplex", "osqp", "quad", "ipopt", "mosek"}
    solver_candidates = [preferred, "osqp", "cplex", "gurobi"]  # osqp ships with cobrapy

    try:
        solver_name = str(model.solver.interface)
    except Exception:
        solver_name = str(type(model.solver).__name__)
    is_qp = any(qp in solver_name.lower() for qp in qp_capable_names)
    print(f"  Current solver ... : {solver_name}")

    if is_qp:
        print(f"  v QP-capable       : {solver_name}")
        return True

    # Try switching
    for candidate in solver_candidates:
        try:
            model.solver = candidate
            try:
                new_name = str(model.solver.interface)
            except Exception:
                new_name = str(type(model.solver).__name__)
            print(f"  -> Switched to      : {new_name}  (QP-capable)")
            return True
        except Exception:
            continue

    print("  !  No QP solver available - MOMA step may fail.")
    return False


# =========================================================
# UTILITY: sigmoid fitness -> activity
# =========================================================

def fitness_to_activity(fitness: float) -> float:
    """
    Map gene fitness (unbounded real) to reaction activity [MIN_ACTIVITY, MAX_ACTIVITY].

    rationale: sigmoid(fitness) ∈ (0, 1); then rescale to [min, max].
    activity=1 -> no constraint; activity->min -> strong inhibition.
    """
    sigmoid = 1.0 / (1.0 + np.exp(-fitness))
    return MIN_ACTIVITY + (MAX_ACTIVITY - MIN_ACTIVITY) * sigmoid


# =========================================================
# UTILITY: gene-level -> reaction-level activity  (improved)
# =========================================================

def gene_to_rxn_activity(model, gene_activity: dict[str, float]) -> dict[str, float]:
    """
    Map gene activity -> reaction activity using the (improved) GPR rule parser.

    For reactions with no matching genes or an empty rule, returns 1.0
    (no constraint).
    """
    rxn_activity: dict[str, float] = {}

    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if not rule or not rxn.genes:
            rxn_activity[rxn.id] = 1.0
            continue

        # Build a lookup of gene -> activity for ALL genes in this reaction
        local_activity = {}
        for g in rxn.genes:
            gid = g.id.lower()
            if gid in gene_activity:
                local_activity[gid] = gene_activity[gid]
            else:
                local_activity[gid] = 1.0  # unseen gene -> no constraint

        try:
            parser = GPRParser(rule)
            score = parser.evaluate(local_activity)
        except Exception:
            # Fallback: unconstrained
            score = 1.0

        rxn_activity[rxn.id] = score

    return rxn_activity


# =========================================================
# UTILITY: sensitivity table for SLACK_FACTOR
# =========================================================

def slack_sensitivity_table():
    """
    Print an illustrative table of what each SLACK_FACTOR value means
    for different activity levels.  Not executed by default; call
    explicitly if you want the reference.
    """
    header = f"{'Slack':>6} | {'act=0.0':>10} {'act=0.2':>10} {'act=0.5':>10} {'act=0.8':>10} {'act=1.0':>10}"
    sep = "-" * len(header)
    print("Perturbation strength (% WT flux retained):")
    print(header)
    print(sep)
    for slack in [0.1, 0.3, 0.5, 0.7, 0.9]:
        vals = [slack + (1 - slack) * a for a in [0.0, 0.2, 0.5, 0.8, 1.0]]
        row = f"{slack:>6.1f} | " + " ".join(f"{v*100:>8.1f}%" for v in vals)
        print(row)
    print()

    print("Interpretation:")
    print("  activity=0.0  -> complete gene inhibition")
    print("  activity=1.0  -> no inhibition (all retained)")
    print("  The closer SLACK_FACTOR is to 0, the tighter the constraint at low activity.")
    print()


# =========================================================
# -- PART A: MOMA SIMULATION ---------------------------------
# =========================================================

# --- Load model ---
print("=" * 70)
print("  Loading model ...")
print("=" * 70)
model = cobra.io.load_json_model(MODEL_FILE)
configure_solver(model, preferred=PREFERRED_SOLVER)

# --- Identify biomass reaction ---
objective_rxns = [r.id for r in model.reactions if r.objective_coefficient != 0]
biomass_rxn = objective_rxns[0]
print(f"\n  Biomass reaction : {biomass_rxn}")

# --- WT baseline (pFBA) ---
print("\n" + "=" * 70)
print("  WT baseline (pFBA) ...")
print("=" * 70)
baseline = pfba(model)
print(f"  Status           : {baseline.status}")
baseline_flux = pd.Series(baseline.fluxes, name="Flux_baseline")
wt_growth = baseline.fluxes[biomass_rxn]
print(f"  WT growth rate   : {wt_growth:.6f}")

# --- Pre-build reaction metadata lookup ---
reaction_map = {r.id: (r.name, r.subsystem) for r in model.reactions}

# --- GPR complexity stats (informational) ---
stats = gpr_complexity_stats(model)
print("\n  GPR complexity distribution:")
for k, v in stats.items():
    print(f"    {k}: {v}")


# =========================================================
# PROCESS ONE GENE-FITNESS FILE  (improved)
# =========================================================

def process_file(fname: str):
    """Run MOMA for a single gene-fitness CSV and save results to OUTPUT_DIR."""

    try:
        print("\n" + "=" * 70)
        print(f"  [FILE]  {fname}")
        print("=" * 70)

        path = os.path.join(INPUT_DIR, fname)

        # --- Load input ---
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        gene_col = df.columns[0]
        value_col = df.columns[1]

        # Normalise gene IDs
        df[gene_col] = df[gene_col].astype(str).str.strip().str.lower()
        # Keep E. coli (b-number) genes
        df = df[df[gene_col].str.startswith("b")]
        df["fitness"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["fitness"])

        print(f"  Valid genes      : {len(df)}")

        # --- Build gene-activity dict ---
        gene_activity = {row[gene_col]: fitness_to_activity(row["fitness"])
                         for _, row in df.iterrows()}

        # --- Match with model ---
        model_gene_ids = {g.id.lower() for g in model.genes}
        common = model_gene_ids & set(gene_activity.keys())
        print(f"  Matched genes    : {len(common)} / {len(gene_activity)}")

        if len(common) == 0:
            print("  x  No matched genes — skipping.")
            return

        # --- Copy & apply perturbation ---
        local_model = model.copy()

        # Reaction-level activities (improved GPR parser)
        rxn_activity = gene_to_rxn_activity(local_model, gene_activity)

        perturbed_rxns = 0
        debug_rows = []

        for rxn in local_model.reactions:
            activity = rxn_activity.get(rxn.id, 1.0)

            # Skip if activity is high (weak / no perturbation)
            if activity >= ACTIVITY_THRESHOLD:
                continue

            wt_flux = baseline.fluxes[rxn.id]

            # Skip near-zero WT flux (reaction not used in WT)
            if abs(wt_flux) < WT_FLUX_THRESHOLD:
                continue

            perturbed_rxns += 1

            # Apply soft bound:
            #   new_bound = WT_flux × [SLACK_FACTOR + (1 - SLACK_FACTOR) × activity]
            #   At activity=1 -> 100% retained (no constraint)
            #   At activity=0 -> SLACK_FACTOR fraction retained (continuous inhibition)
            bound_ratio = SLACK_FACTOR + (1.0 - SLACK_FACTOR) * activity

            if wt_flux > 0:
                direction = "forward"
                new_upper = wt_flux * bound_ratio
                rxn.upper_bound = min(rxn.upper_bound, max(new_upper, 1e-6))
            else:
                direction = "reverse"
                new_lower = wt_flux * bound_ratio
                rxn.lower_bound = max(rxn.lower_bound, min(new_lower, -1e-6))

            debug_rows.append({
                "Reaction":            rxn.id,
                "Name":                rxn.name,
                "Subsystem":           rxn.subsystem,
                "WT_Flux":             wt_flux,
                "Direction":           direction,
                "Activity":            activity,
                "Upper_Bound_After":   rxn.upper_bound,
                "Lower_Bound_After":   rxn.lower_bound,
                "Original_Upper":      model.reactions.get_by_id(rxn.id).upper_bound,
                "Original_Lower":      model.reactions.get_by_id(rxn.id).lower_bound,
                "GPR":                 rxn.gene_reaction_rule,
            })

        print(f"  Perturbed rxns   : {perturbed_rxns}")

        # --- Run MOMA ---
        print("  Running MOMA ...")
        try:
            sol = moma(local_model, solution=baseline)
            print(f"  MOMA status      : {sol.status}")
        except Exception as e:
            print(f"  !  MOMA failed    : {e}")
            return

        # --- Build output table ---
        cond_flux = pd.Series(sol.fluxes, name="Flux_condition")
        true_growth = sol.fluxes[biomass_rxn]

        out = pd.DataFrame({"Reaction": [r.id for r in local_model.reactions]})
        out["Name"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[0])
        out["Subsystem"] = out["Reaction"].map(lambda x: reaction_map.get(x, ("", ""))[1])
        out = out.merge(baseline_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
        out = out.merge(cond_flux.reset_index().rename(columns={"index": "Reaction"}), on="Reaction")
        out["ΔFlux"]         = out["Flux_condition"] - out["Flux_baseline"]
        out["Abs_ΔFlux"]     = out["ΔFlux"].abs()
        out["Reaction_Activity"] = out["Reaction"].map(rxn_activity).fillna(1.0)
        out["Growth_Rate"]   = true_growth
        out["WT_Growth"]     = wt_growth
        out["Growth_Ratio"]  = true_growth / wt_growth if wt_growth else np.nan
        out["MOMA_Objective"] = sol.objective_value
        out["Solution_Status"] = sol.status

        # --- Compute summary ---
        total_rewiring = out["Abs_ΔFlux"].sum()
        changed_rxns   = (out["Abs_ΔFlux"] > CHANGE_THRESHOLD).sum()

        # --- Pathway summary ---
        pathway_summary = (
            out.groupby("Subsystem")["Abs_ΔFlux"]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
        )

        # --- Top 100 rewired ---
        top_rewiring = out.sort_values("Abs_ΔFlux", ascending=False).head(100)

        # --- Summary table ---
        summary_df = pd.DataFrame({
            "Metric": [
                "WT_Growth", "Condition_Growth", "Growth_Ratio",
                "Total_Rewiring", "Mean_Rewiring", "Max_Rewiring",
                "Changed_Reactions", "Perturbed_Reactions",
            ],
            "Value": [
                wt_growth, true_growth, true_growth / wt_growth,
                total_rewiring, out["Abs_ΔFlux"].mean(), out["Abs_ΔFlux"].max(),
                changed_rxns, perturbed_rxns,
            ],
        })

        # --- Name for output files ---
        name = os.path.splitext(fname)[0].upper()

        # --- Save (CSV — faster, no openpyxl segfault) ---
        out.to_csv(os.path.join(OUTPUT_DIR, f"{name}_FLUX.csv"), index=False)
        pd.DataFrame(debug_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"{name}_PERTURBATION.csv"), index=False)
        pathway_summary.to_csv(os.path.join(OUTPUT_DIR, f"{name}_PATHWAY.csv"))
        top_rewiring.to_csv(os.path.join(OUTPUT_DIR, f"{name}_TOP_REWIRING.csv"), index=False)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_SUMMARY.csv"), index=False)

        # --- Print diagnostics ---
        print(f"\n  Growth rate      : {true_growth:.6f}")
        print(f"  Growth ratio     : {true_growth / wt_growth:.4f}" if wt_growth else "  Growth ratio     : N/A")
        print(f"  Total rewiring   : {total_rewiring:.4f}")
        print(f"  Changed rxns     : {changed_rxns}")
        print(f"  OK  {name}")
        print()

    except Exception:
        print(f"\n  x  Error processing: {fname}")
        traceback.print_exc()


# %%
# =========================================================
# -- PART B: FEATURE MATRIX CONSTRUCTION  (improved) --------
# =========================================================

def build_feature_matrices():
    """
    Read all _FLUX.xlsx files from OUTPUT_DIR and build ML-ready matrices.

    Saves:
        flux_feature_matrix_raw.csv         — raw ΔFlux values   (+ label)
        flux_feature_matrix_normalized.csv  — row-normalised ΔFlux values  (+ label)
        reaction_metadata.csv               — per-reaction summary for downstream analysis
        reaction_frequency.csv              — how often each reaction changes
    """

    print("\n" + "=" * 70)
    print("  Building feature matrices ...")
    print("=" * 70)

    # --- Load metadata / labels ---
    # Metadata uses hyphens:      "AMIKACIN-AMOXICILLIN"
    # Flux files use double underscore: "AMIKACIN__AMOXICILLIN_FLUX.csv"
    label_map = None
    if os.path.exists(METADATA_FILE):
        meta_df = pd.read_csv(METADATA_FILE)
        meta_df.columns = meta_df.columns.str.strip().str.lower()
        if "drug_pair" in meta_df.columns and "label" in meta_df.columns:
            # Normalise: hyphens -> double underscores, uppercase
            meta_df["drug_pair"] = (
                meta_df["drug_pair"]
                .astype(str).str.strip().str.upper()
                .str.replace("-", "__")
            )
            label_map = dict(zip(meta_df["drug_pair"], meta_df["label"]))
            n_pos = (meta_df["label"] == 1).sum()
            n_neg = (meta_df["label"] == 0).sum()
            print(f"  Labels loaded    : {len(meta_df)} total "
                  f"(synergy=1: {n_pos}, non-synergy=0: {n_neg})")
        else:
            print(f"  !  metadata.csv columns: {list(meta_df.columns)} — expected 'drug_pair', 'label'")
    else:
        print(f"  !  metadata.csv not found at:\n     {METADATA_FILE}")

    # --- Locate flux files ---
    flux_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_FLUX.csv")])
    print(f"  Flux files found : {len(flux_files)}")

    all_rows = []

    for fname in flux_files:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
        pair_name = fname.replace("_FLUX.csv", "").upper()
        row = {"drug_pair": pair_name}
        for _, r in df.iterrows():
            row[str(r["Reaction"])] = r[FEATURE_COLUMN]
        all_rows.append(row)

    # --- Build raw matrix ---
    X = pd.DataFrame(all_rows).fillna(0)

    # Attach label
    if label_map is not None:
        X["label"] = X["drug_pair"].map(label_map)
        matched = X["label"].notna().sum()
        print(f"  Labels matched   : {matched} / {len(X)}")
        # Reorder columns: drug_pair, label, then features
        cols = ["drug_pair", "label"] + [c for c in X.columns if c not in ("drug_pair", "label")]
        X = X[cols]
    feature_cols = [c for c in X.columns if c not in ("drug_pair", "label")]
    print(f"  Raw shape        : {X.shape}  ({len(feature_cols)} reactions × {len(X)} samples)")

    # --- Remove zero-variance reactions ---
    stds = X[feature_cols].std()
    kept_std = stds[stds > 0].index.tolist()
    print(f"  After zero-var   : ({len(X)}, {len(kept_std) + 1})  "
          f"(removed {len(feature_cols) - len(kept_std)})")
    feature_cols = kept_std

    # --- Frequency filter (proportional) ---
    n_samples = len(X)
    freq_threshold = max(MIN_FREQUENCY_ABSOLUTE, round(MIN_FREQUENCY_RATIO * n_samples))
    freq = (np.abs(X[feature_cols]) > CHANGE_THRESHOLD).sum(axis=0)
    kept_freq = freq[freq >= freq_threshold].index.tolist()
    print(f"  Frequency thresh : {freq_threshold}  "
          f"(={MIN_FREQUENCY_RATIO*100:.0f}% of {n_samples} samples, min {MIN_FREQUENCY_ABSOLUTE})")
    print(f"  After freq filt  : ({len(X)}, {len(kept_freq) + 1})  "
          f"(removed {len(feature_cols) - len(kept_freq)})")
    feature_cols = kept_freq

    # --- Columns to always carry (drug_pair ± label) ---
    id_cols = [c for c in ["drug_pair", "label"] if c in X.columns]

    # --- (1) Save raw matrix ---
    if OUTPUT_RAW:
        X_raw = X[id_cols + feature_cols].copy()
        raw_path = os.path.join(OUTPUT_DIR, "flux_feature_matrix_raw.csv")
        X_raw.to_csv(raw_path, index=False)
        print(f"  Saved raw        : {raw_path}")

    # --- (2) Save normalised matrix (row L1 norm) ---
    if OUTPUT_NORMALIZED:
        X_norm_vals = X[feature_cols].div(
            X[feature_cols].abs().sum(axis=1), axis=0
        ).fillna(0)
        X_norm = pd.concat([X[id_cols].reset_index(drop=True), X_norm_vals], axis=1)
        norm_path = os.path.join(OUTPUT_DIR, "flux_feature_matrix_normalized.csv")
        X_norm.to_csv(norm_path, index=False)
        print(f"  Saved normalized : {norm_path}")

    # --- (3) Reaction metadata file ---
    if OUTPUT_METADATA:
        # Collect per-reaction stats from the ORIGINAL full matrix (before frequency filter)
        # so we capture ALL reactions, not just the surviving ones
        X_full = pd.DataFrame(all_rows).fillna(0)
        full_features = [c for c in X_full.columns if c not in ("drug_pair", "label")]
        X_sub = X_full[full_features]

        # Read the first flux file for Name / Subsystem lookup
        first_flux = pd.read_csv(os.path.join(OUTPUT_DIR, flux_files[0]))
        rxn_info = first_flux[["Reaction", "Name", "Subsystem"]].drop_duplicates("Reaction")

        # Full frequency (across all samples, before any filtering)
        full_freq = (np.abs(X_sub.values) > CHANGE_THRESHOLD).sum(axis=0)
        full_std = X_sub.std().values
        full_mean = X_sub.mean().values
        full_mean_abs = X_sub.abs().mean().values

        meta = pd.DataFrame({
            "Reaction":          full_features,
            "Frequency":         full_freq,
            "Frequency_Ratio":   full_freq / n_samples,
            "Mean_ΔFlux":        full_mean,
            "Mean_Abs_ΔFlux":    full_mean_abs,
            "Std_ΔFlux":         full_std,
            "Zero_Variance":     (full_std == 0).astype(int),
            "Pass_Frequency_Filter": [1 if f >= freq_threshold else 0 for f in full_freq],
        })
        meta = meta.merge(rxn_info, on="Reaction", how="left")
        meta = meta.sort_values("Frequency", ascending=False)

        meta_path = os.path.join(OUTPUT_DIR, "reaction_metadata.csv")
        meta.to_csv(meta_path, index=False)
        print(f"  Saved metadata   : {meta_path}")

        # Also save the reaction frequency file (original behaviour)
        freq_df = meta[["Reaction", "Name", "Subsystem", "Frequency", "Frequency_Ratio",
                         "Mean_Abs_ΔFlux", "Std_ΔFlux"]].copy()
        freq_path = os.path.join(OUTPUT_DIR, "reaction_frequency.csv")
        freq_df.to_csv(freq_path, index=False)
        print(f"  Saved frequency  : {freq_path}")

        # --- Print top 20 for quick check ---
        print("\n  Top 20 most-frequently changed reactions:")
        print(f"  {'Reaction':>12} {'Name':>40} {'Subsystem':>25} {'Freq':>6} {'Ratio':>6}")
        print(f"  {'-'*12} {'-'*40} {'-'*25} {'-'*6} {'-'*6}")
        for _, r in meta.head(20).iterrows():
            name = (str(r.get("Name", ""))[:38] if pd.notna(r.get("Name")) else "")
            sub  = (str(r.get("Subsystem", ""))[:23] if pd.notna(r.get("Subsystem")) else "")
            print(f"  {r['Reaction']:>12} {name:>40} {sub:>25} {r['Frequency']:>6} {r['Frequency_Ratio']:>5.0%}")

    print(f"\n  OK  Feature matrices ready in:\n     {OUTPUT_DIR}")


# %%
# =========================================================
# -- MAIN ---------------------------------------------------
# =========================================================

if __name__ == "__main__":

    # -------------------------------------------------------
    # [OPTIONAL] Print slack sensitivity table for reference
    # -------------------------------------------------------
    # Uncomment the line below to see how each SLACK_FACTOR
    # translates to % WT flux retained at different activities:
    #
    # slack_sensitivity_table()

    # -------------------------------------------------------
    # PART A: Run MOMA for all gene-combination files
    # -------------------------------------------------------
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
    print(f"\n  Total gene-combo files : {len(files)}")

    for i, fname in enumerate(files):
        print(f"\n{'─' * 70}")
        print(f"  [{i+1} / {len(files)}]")
        process_file(fname)

    print(f"\n{'=' * 70}")
    print("  OK  PART A complete — all MOMA simulations done.")
    print(f"{'=' * 70}")

    # -------------------------------------------------------
    # PART B: Build feature matrices from flux results
    # -------------------------------------------------------
    build_feature_matrices()

    print(f"\n{'=' * 70}")
    print("  **  Pipeline finished successfully.")
    print(f"{'=' * 70}")
