# Pipeline & Results

End-to-end description of the E. coli antibiotic-combination synergy pipeline:
what each stage consumes and produces, the results obtained, and the traps to avoid.

---

## 0. Sign convention — read this first

**`score < 0` = SYNERGY, `score > 0` = ANTAGONISM.**

Every conclusion in this project depends on this, and it is easy to invert by accident.
Verified with zero exceptions against the source paper's own `Interacion sign` column
(`1-ecoli.xlsx` Supplementary Table 2, n = 381):

| `Interacion score` | `Interacion sign` | n |
|---|---|---|
| < 0 | Synergy | 161 |
| > 0 | Antagonism | 220 |

Derived quantities:

| Quantity | Meaning |
|---|---|
| `label = 1` in `metadata*.csv` | synergy |
| `proba_199_model`, `proba_411_model` | **P(synergy)** |
| candidate ranking | sort by **descending** `proba_411_model` |

---

## 1. End-to-end flow

```
Gene fitness data
      │
      ▼  01_flux_simulation   (MOMA on iJO1366 + GPR parsing)
Per-combination _FLUX.csv        2583 reactions × ΔFlux
      │
      ▼  02_feature_matrix    ΔFlux_combo − ΔFlux_A − ΔFlux_B
Non-additive feature matrix      zero-variance + frequency(≥10%) filter + L1 row norm
      │
      ▼  03_model_training    ClipOutliers → StandardScaler → LogisticRegression
pipeline_compact.joblib          120 features, C = 0.5623
      │
      ▼  predict_all_combos.py
All-combination predictions       5565 pairs = C(106,2), P(synergy) each
      │
      ▼  05_wetlab_selection
Wet-lab candidate list
```

---

## 2. Stage by stage

### Stage 1 — Flux simulation

| | |
|---|---|
| Method | MOMA (soft gene-activity constraint) on iJO1366 |
| Input | gene-knockout fitness data (`gene_combo-2/`) |
| Output | `continuous_bounded_moma_single/` — 128 single drugs<br>`continuous_bounded_moma_all/` — 5671 combinations |
| Key params | `MIN_ACTIVITY` 0.5 (v1) / 0.3 (v2), `SLACK_FACTOR` 0.5, `ACTIVITY_THRESHOLD` 0.8 |
| Solver | Gurobi (QP); GLPK / osqp as fallback |

Each `_FLUX.csv` is one simulation: 2583 reactions × `ΔFlux` relative to the
unperturbed model.

### Stage 2 — Non-additive feature matrix

The feature for reaction *r* is the **non-additive** flux shift:

```
ΔFlux_r(combo) − ΔFlux_r(drug A) − ΔFlux_r(drug B)
```

This removes the additive part of the response, so the model sees only the
interaction. Filters applied, in order:

1. drop zero-variance features
2. drop features changed in fewer than 10 % of samples
3. L1-normalise each row (feature vector sums to 1)
4. `StandardScaler` inside the model pipeline

Output for the training set: **411 pairs × 975 columns → 120 retained features**
(compact model) or 800 (full model).

### Stage 3 — Model training

| | |
|---|---|
| Data | `non_additive_normalized.csv` — **411 pairs, 51 drugs, 168 positive (synergy)** |
| Pipeline | `ClipOutliers(0.1/99.9) → StandardScaler → LogisticRegression(L2)` |
| Feature count | 120 (compact) / 800 (full) |
| C | 0.5623 (compact), selected by `GridSearchCV` |
| Artifacts | `models/pipeline_compact.joblib`, `models/pipeline_full.joblib` |

### Stage 4 — Evaluation

**Cross-validated AUC** (reproduced with the 120-feature compact model, C = 0.5623):

| Evaluation | AUC | Note |
|---|---|---|
| Pair-level stratified 5-fold CV | 0.741 | optimistic — pairs share drugs |
| **Drug-disjoint 5-fold CV** | **0.662** | test drugs never seen in training |

With only 51 drugs and 411 pairs, the pairs share drugs heavily, so pair-level CV
leaks across folds. **Use the drug-disjoint number when judging the ability to
predict a genuinely new drug pair.**

> ⚠️ **`model_comparison.csv` is void.** It reports ROC-AUC 0.937 (compact) and
> 0.856 (full), but those figures came from a run with data leakage and do not
> reproduce — the same data gives 0.741 pair-level. Ignore that file.
>
> The leakage was in the *evaluation*, not the training: the saved
> `pipeline_compact.joblib` scores 0.787 on its own training set against 0.741
> cross-validated, a normal overfitting gap. A leak in training would show up as
> an in-sample score near 1.0. The deployed predictions are therefore not
> affected — only the reported metric was.

### Stage 5 — All-combination prediction

`predict_all_combos.py` scores every one of the 106 drugs against every other —
**5565 pairs = C(106,2)** — writing `all_combo_predictions.csv`.

> ⚠️ **`proba_199_model` is constant 1.0 across all 5565 rows.** The 199-sample /
> 968-feature model carries no information on this data. Discard that column;
> use `proba_411_model` only.

> ⚠️ The top of the `proba_411` ranking is dominated by a few "hub" drugs
> (`FUSIDICACID` appears in 42 of the highest-scoring pairs, `CEFSULODIN` in 33).
> A drug scoring high against almost everything is more likely an embedding
> artefact than 40 genuine synergies. Always cross-check the ranking against
> mechanism, and hand-exclude same-target pairs (β-lactam × β-lactam etc.) —
> measured same-class pairs are consistently *antagonistic*.

### Stage 6 — Wet-lab candidate selection

Combines the model probability with pharmacology review, training-set coverage
(has this drug been seen?), literature exclusion, and chemical diversity. See
`05_wetlab_selection/`.

---

## 3. External validation

Source: Chandrasekaran et al., *Mol Syst Biol* **12**:872 (2016), Dataset EV1 —
measured on a different experimental platform. See the Data sources section of
the README for full provenance.

### 3.1 The merge, step by step

| Step | Rows | Note |
|---|---|---|
| Training set | 411 | 51 drugs, 168 synergy |
| External pairs matching the MOMA drug set | 153 | 18 further pairs involve `H22` and are unmappable |
| Concatenated | 564 | |
| − duplicates | −85 | pairs already in the training set: **identical MOMA features, labels from a different platform** |
| **Merged, deduplicated** | **479** | 184 synergy / 295 antagonism |

The 85 overlapping pairs are kept aside as `external_vs_train_labels.csv`.
The two platforms agree on **75/85 = 88.2 %** of them; the 10 conflicts are
genuine inter-platform disagreement, not a coding error.

> ⚠️ Skipping the dedup step inflates the benchmark. With the duplicates left in
> (564 rows) XGB reaches ROC 0.822 / MCC 0.458 — but the duplicated pairs put an
> *identical* feature row in both the training and the test fold, so the model is
> partly scored on rows it has memorised. 479 is the honest number.

### 3.2 Benchmark — 5-fold CV, hyperparameters tuned per fold

| Model | Rows | Accuracy | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|
| RF · old 411 | 411 | 0.701 | 0.589 | **0.778** | 0.690 | 0.371 |
| RF · merged | 479 | 0.703 | 0.561 | 0.752 | 0.651 | 0.352 |
| XGB · old 411 | 411 | 0.708 | 0.606 | 0.768 | 0.698 | 0.386 |
| XGB · merged | 479 | 0.718 | **0.627** | 0.756 | 0.673 | **0.404** |
| LR · old 411 | 411 | 0.679 | 0.585 | 0.730 | 0.660 | 0.329 |
| LR · merged | 479 | 0.699 | 0.597 | 0.735 | 0.664 | 0.366 |

**Reading:** adding the external data is roughly **neutral**. MCC improves for
XGB (+0.018) and LR (+0.037) but drops slightly for RF (−0.019); ROC-AUC is flat
to slightly down. The 68 genuinely new pairs are not enough to move the needle,
and the second platform carries its own noise.

**What the fix actually bought:** before it, the merge collapsed MCC from ~0.34
to ~0.19 because the two label conventions contradicted each other. That failure
is gone — the merged set now trains and evaluates normally.

### 3.3 Generalisation to unseen drug pairs

The 411 training pairs share drugs heavily (only 51 drugs), so any pair-level CV
leaks and overstates performance. Two independent estimates of what actually
matters — predicting a combination of two drugs the model has not seen:

| Estimate | AUC | What it measures |
|---|---|---|
| Drug-disjoint 5-fold CV (internal) | **0.66** | new drug pairs, same platform |
| External held-out test (cross-platform) | 0.60 | new drug pairs, different study **and** assay |

The external test is the one `04_model_evaluation/external_validation.py` runs:
train on the 411, test on the 68 deduplicated held-out pairs.

| Model | ROC-AUC | MCC | Spearman ρ vs measured score |
|---|---|---|---|
| RF | 0.613 | +0.039 | −0.277 |
| XGB | 0.606 | +0.135 | −0.288 (p = 0.017) |
| LR | 0.602 | +0.061 | −0.239 (p = 0.050) |

**Read the direction carefully:** ρ is negative here and that is correct — the
source convention is `score < 0` = synergy, so the raw score must anti-correlate
with P(synergy). As a direct check, the mean predicted P(synergy) is higher for
the synergistic pairs than for the antagonistic ones in all three models
(XGB 0.48 vs 0.38; RF 0.40 vs 0.36; LR 0.43 vs 0.34).

**But the sample is too small to size the effect.** Of the 68 held-out pairs only
**16 are synergistic**, and the binary group comparison reaches no significance
(Mann-Whitney p = 0.22–0.34). The continuous Spearman correlation, which uses the
full score instead of a threshold, does reach significance (p = 0.017 for XGB) —
so the model is picking up real signal, on a sample too small to pin its
magnitude down. Treat 0.60 as a low-confidence lower bound, not a contradiction
of the 0.66 internal estimate.

**When evaluating on the external set, prefer the continuous Spearman correlation
over binary MCC/AUC** — the binarisation throws away most of the information
available from only 16 positive pairs.

---

## 4. Known issues (fixed)

### 4.1 Label inversion in `merge_and_benchmark.py`

```python
# before — inverts every new pair
r["label"] = 1 if score > 0 else 0

# after — score < 0 is synergy
r["label"] = 1 if score < 0 else 0
```

The external source's scores follow the same convention as the training labels
(`score < 0` = synergy), so `score > 0` assigned label 1 to every *antagonistic*
pair. Because the old 411 rows already carried correct labels, the merged file
contained two contradictory conventions for the same drug pairs — no model can fit
that, which is exactly the MCC collapse in the table above.

### 4.2 Wrong abbreviation expansion

```python
# before
"CEF": "CEFSULODIN"
# after
"CEF": "CEFOXITIN"
```

Determined by matching the 19 drug pairs the external set shares with the Nature
Communications screen: CEFOXITIN agrees 84 % directionally, CEFSULODIN 73 %.
Fixing this also raised the number of matchable external pairs from 74 to 153,
because CEFOXITIN has broader MOMA coverage than CEFSULODIN.

### 4.3 Duplicate drug pairs with conflicting labels

The external set shares 85 drug pairs with the training set. Both rows carry the
**same** non-additive MOMA features (same simulation) but labels from two
different platforms — 10 of them contradict outright. Concatenating the two sets
therefore fed the model contradictory examples of the same input.

`merge_and_benchmark.py` now deduplicates: for an overlapping pair the training
row is kept and the external row dropped, and the pair is written to
`external_vs_train_labels.csv` for cross-platform comparison. An assertion checks
that exactly `len(old) + len(new) − len(overlap)` rows survive, so a silent
mis-drop cannot recur.

> A first attempt at this dedup was wrong: the alignment step runs `.fillna(0)`,
> which turns the old rows' `NaN` score into `0.0`, so filtering on
> `score.isna()` silently matched nothing and deleted the *training* rows of the
> 85 overlapping pairs instead. Hence the assertion.

### 4.4 Stale inter-study block in `external_validation.py`

That script recomputed cross-platform label agreement by intersecting its own
train and test frames — which, after the dedup in § 4.3, are disjoint by
construction. The loop found nothing and the report crashed on a division by
zero. It now reads `external_vs_train_labels.csv` (exported by
`merge_and_benchmark.py`) and reports the overlap from there.

The dedup also fixed a subtler problem in the same script: its test set used to
include the 85 pairs that were still in training. It now holds only the 68 pairs
the model has genuinely not seen, so the external evaluation is leakage-free.

---

## 5. Reproducing

```bash
# 1. flux simulation
python 01_flux_simulation/single_drug_moma.py
python 01_flux_simulation/run_moma_5154.py

# 2. non-additive feature matrix
python 02_feature_matrix/step3_only.py

# 3. train + predict
python 03_model_training/final_model_train.py
python 03_model_training/predict_all_combos.py

# 4. merge external data and benchmark
python 02_feature_matrix/merge_and_benchmark.py

# 5. wet-lab candidates
python 05_wetlab_selection/full_review.py
python 05_wetlab_selection/final_wetlab_selection.py
```

Requires placing the data in `data/` — see README § Data preparation. MOMA needs
Gurobi for QP solving.

---

## 6. Open items

1. **Verify the remaining `ABBR_MAP` abbreviations.** Only `CEF` was checked; the
   external set covers just 19 drugs, so the others cannot be cross-validated
   from the data available here — check them against the source paper.
2. **`H22` is unresolved.** It appears in 18 pairs of the external set but is not
   in the MOMA drug set, so those pairs cannot be mapped. Identify what H22 is.
3. **External-set caveats.** Scores saturate at 4.3940 and H22 behaves as a hub.
   Prefer Spearman correlation against the continuous score over binary MCC.
4. **Dataset EV2 must not be used for validation** — it holds INDIGO model
   predictions, not measurements.

### Resolved

- ~~`model_comparison.csv` reports ROC-AUC 0.937, which does not reproduce.~~
  Confirmed as a **data-leakage artefact** from an earlier run; that file is void
  and the leakage was confined to the evaluation (see § 2, Stage 4). The saved
  model and the deployed predictions are unaffected.
