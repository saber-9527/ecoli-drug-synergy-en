# E. coli Antibiotic Combination Synergy Prediction

A workflow for predicting **antibiotic combination synergy in *E. coli* (K-12, iJO1366)** built on **MOMA / iMAT flux balance analysis** and **machine learning**.

Starting from gene fitness data, the pipeline uses metabolic network simulation (MOMA) to derive the flux perturbation of each drug combination, builds a non-additive flux feature matrix, trains classification/regression models to predict combination synergy, and finally selects candidate combinations suitable for wet-lab validation.

> **Sign convention: `score < 0` = synergy, `score > 0` = antagonism.** This is easy to
> invert by accident, so verify it against the source paper's `Interacion sign` column
> before interpreting any score in this repo.

## Pipeline overview

```
Gene fitness data
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 01_flux_simulation     MOMA / iMAT flux simulation│
│   (iJO1366 + GPR parsing + pfba/moma/iMAT)        │
└─────────────────────────────────────────────────┘
    │  Per combination: _FLUX.csv (2583 reactions × ΔFlux)
    ▼
┌─────────────────────────────────────────────────┐
│ 02_feature_matrix      Non-additive feature matrix │
│   ΔFlux_combo - ΔFlux_A - ΔFlux_B                 │
│   Zero-variance / frequency filter + L1 row norm   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 03_model_training      Model training & deployment │
│   Logistic Regression (L1/L2), XGBoost, RF,        │
│   Ridge/GBR regression, nested cross-validation    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 04_model_evaluation    Evaluation & interpretability│
│   5-fold CV, ROC/PR-AUC, MCC, SHAP, ext. validation│
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 05_wetlab_selection    Wet-lab candidate selection │
│   Pharmacology review + training coverage          │
│   + literature exclusion + diversity              │
└─────────────────────────────────────────────────┘
    │
    ▼
Wet-lab validation → relabel → retrain (active-learning loop)
```

## Repository layout

```
├── 01_flux_simulation/       # MOMA/iMAT flux simulation
│   ├── flux_ml_pipeline.py        # MOMA→features pipeline v1 (MIN_ACTIVITY=0.5)
│   ├── flux_ml_pipeline_v2.py     # MOMA→features pipeline v2 (MIN_ACTIVITY=0.3, tighter)
│   ├── run_moma_5154.py           # Batch MOMA, 300 combos/batch to avoid OOM
│   ├── run_new_combos.py          # Incremental run for newly added combos
│   ├── single_drug_moma.py        # Single-drug MOMA + non-additive effects
│   ├── imat-new.py / imat_pairs_new2.py  # iMAT alternative method
│   └── gimme.py                   # GIMME algorithm implementation
├── 02_feature_matrix/        # Feature matrix construction
│   ├── step3_only.py / step3_only_filtered.py   # Rebuild non-additive matrix
│   ├── build_feature_matrix_only.py             # Build from existing results
│   ├── build_combination_features.py            # Combination features (reaction→subsystem)
│   ├── build_v2_and_benchmark.py                # v2 matrix + comparison
│   └── merge_and_benchmark.py                   # Merge old & new data
├── 03_model_training/        # Model training & deployment
│   ├── final_model_train.py      # Two-stage final model training
│   ├── train_regression.py       # Continuous-score regression (Ridge/GBR)
│   └── predict_all_combos.py     # Predict all combos with saved models
├── 04_model_evaluation/      # Evaluation & interpretability
│   ├── ml_benchmark.py / ml_svm.py            # Multi-model benchmark
│   ├── benchmark_nonadditive.py               # LR/RF/XGB comparison
│   ├── benchmark_compact_vs_full.py           # Feature-subset comparison
│   ├── feature_sweep.py / _save_results.py    # Feature-count sweep
│   ├── ml_best_model_analysis.py              # SHAP interpretability
│   ├── external_validation.py                 # Cross-study validation
│   └── single_feature_test.py                 # Single-feature diagnostics
├── 05_wetlab_selection/      # Wet-lab candidate selection
│   ├── full_review.py            # Full pharmacology review
│   ├── prescreen_candidates.py   # Prescreening pipeline
│   ├── review_predictions.py     # Candidate annotation
│   ├── clean_drug_table.py       # Clean combination table
│   └── final_wetlab_selection.py # Final candidates
├── 06_utils/                 # Utilities
│   ├── xlsx_to_csv.py            # _FLUX.xlsx → _FLUX.csv
│   └── run_moma_loop.ps1         # Batch loop script (OOM-safe restart)
├── models/                   # Deployed models
│   ├── best_logreg_pipeline_enhanced.joblib   # 199-sample L1 model (968 features)
│   ├── pipeline_compact.joblib                # 411-sample Compact model (120 features)
│   ├── pipeline_full.joblib                   # Full model (800 features)
│   ├── regression_model.joblib / regression_scaler.joblib  # Regression model
│   ├── svm_reaction_synergy_with_features.pkl # SVM model
│   └── iJO1366.json                          # E. coli K-12 metabolic model
├── notebooks/                # Core Jupyter Notebooks
└── data/                     # Data directory (add your own, see below)
```

## Environment & dependencies

Python 3.11 + Miniforge. Core dependencies are listed in `requirements.txt`. Flux simulation needs a QP solver:

| Solver | Purpose |
|--------|---------|
| **Gurobi** (recommended) | QP solving for MOMA; requires commercial/academic license |
| GLPK / osqp | Fallback solvers (v1 pipeline) |

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate ecoli-synergy
```

## Usage

### 1. Data preparation

Put the data files in `data/` (this corresponds to the `../data/` paths in the code):

- Gene fitness CSVs (e.g. `gene_combo-2/`)
- Single-drug / combination MOMA results (`continuous_bounded_moma_single/`, `continuous_bounded_moma_all/`)
- Training labels `metadata_merged.csv`
- Literature data `1-ecoli.xlsx` (Nature Communications 2020, Supplementary Table 2)
- External-validation data `desktop_new_pairs.xlsx` — see [Data sources](#data-sources)
  for provenance and the **sign convention**, which is easy to get backwards.

### 2. Flux simulation

```bash
# Single-drug MOMA
python 01_flux_simulation/single_drug_moma.py

# Batch combination MOMA (300/batch; use the PS1 loop on Windows to avoid OOM)
python 01_flux_simulation/run_moma_5154.py
./06_utils/run_moma_loop.ps1        # PowerShell
```

### 3. Feature matrix

```bash
python 02_feature_matrix/step3_only.py
```

### 4. Model training & prediction

```bash
python 03_model_training/final_model_train.py   # Train
python 03_model_training/predict_all_combos.py  # Predict all combos
```

### 5. Wet-lab candidates

```bash
python 05_wetlab_selection/full_review.py
python 05_wetlab_selection/final_wetlab_selection.py
```

## Key methods & parameters

| Stage | Method | Key parameters |
|-------|--------|----------------|
| Flux simulation | MOMA (soft gene-activity constraint) | MIN_ACTIVITY=0.5 (v1) / 0.3 (v2), SLACK_FACTOR=0.5, ACTIVITY_THRESHOLD=0.8 |
| Alternative | iMAT, GIMME | kappa/rho/epsilon |
| Features | Non-additive ΔFlux = Combo − A − B | Zero-variance filter + frequency filter (≥10%) + L1 row norm + StandardScaler |
| Classification | Logistic Regression (L1/L2), XGBoost, RF, SVM | Nested CV, stratified 5-fold |
| Regression | Ridge, GradientBoosting | Continuous Interaction Score |
| Selection | Pharmacology review + training coverage + literature exclusion | M2 probability tiers + diversity |

## Score sign convention — read before interpreting any score

**In this project's data, `score < 0` means SYNERGY and `score > 0` means ANTAGONISM.**

This is the single easiest thing to get backwards, and getting it backwards silently
inverts every conclusion. It is verified with zero exceptions against the source
paper's own `Interacion sign` column (`1-ecoli.xlsx`, Supplementary Table 2, n = 381):

| `Interacion score` | `Interacion sign` | count |
|---|---|---|
| < 0 | Synergy | 161 |
| > 0 | Antagonism | 220 |

A sanity check: `Trimethoprim + Sulfamonomethoxine = −0.500` and is labelled Synergy —
the classic folate double-block, as expected.

Consequently:

| Quantity | Meaning |
|---|---|
| `label = 1` in `metadata*.csv` | synergy (`label = 0` = antagonism) |
| `proba_199_model`, `proba_411_model` | **P(synergy)** — column names are accurate |
| Ranking candidates | sort by **descending** `proba_411_model` |

Both literature sources used here share this direction (84 % directional agreement
on the 19 drug pairs they have in common).

## Data sources

- **iJO1366**: genome-scale metabolic model of *E. coli* K-12 MG1655 (Orth et al., 2011)
- **Gene fitness**: large-scale gene-knockout fitness data
- **Training labels** (`metadata_merged.csv`, 411 pairs × 51 drugs):
  high-throughput drug-interaction screen from *Nature Communications* 2020
  (BW25113 / iAi1 strains), file `1-ecoli.xlsx` Supplementary Table 2.
  Labels are **experimental**, not model output — verified against the raw
  interaction scores with zero exceptions.
- **External validation** (`desktop_new_pairs.xlsx`):
  Chandrasekaran et al., *Chemogenomics and orthology-based design of antibiotic
  combination therapies*, **Mol Syst Biol 12:872 (2016), Dataset EV1**
  (published file `MSB-12-872-s003.xlsx`). 170 published rows plus one manual row
  (`AMK + CEF`): **153 map onto the MOMA drug set**, 18 involve `H22` which is not
  in that set. 85 of the 153 duplicate a training pair (deduplicated by
  `merge_and_benchmark.py`; the cross-platform comparison is written to
  `external_vs_train_labels.csv`).

  ⚠️ Three caveats when using this file:
  - Abbreviations must be expanded before matching MOMA output. **`CEF` = CEFOXITIN,
    not Cefsulodin** — confirmed by matching against the Nature Communications sign
    column (CEFOXITIN agrees 84 %, CEFSULODIN 73 %). Getting this wrong silently
    drops 79 of the 153 usable pairs.
  - The scores saturate at `4.3940` (four pairs share this exact value) and `H22`
    scores 2.0–4.4 against all 18 of its partners, which is a hub artefact rather
    than 18 genuine synergies. Prefer Spearman correlation against the continuous
    score over hard binary MCC.
  - The two platforms agree on only **88 %** of the 85 shared pairs. Treat
    disagreements as measurement noise, not as labels to "correct".
- **Not experimental — do not use for validation**: `MSB-12-872-s004.xlsx`
  (Dataset EV2, 2627 pairs × 73 drugs × 3 species) contains **INDIGO model
  predictions**, not measurements. `MSB-12-872-s005.xlsx` (EV3) holds the gene
  rankings INDIGO uses internally.

## Notes

- All code uses relative paths (`../data/`, `../models/`); place the data in the corresponding directories first
- MOMA solving depends on Gurobi; without a license some scripts fall back to the default solver
- `run_moma_loop.ps1` is a Windows PowerShell script; use an equivalent shell loop on Linux

## License

MIT License — see the LICENSE file.
