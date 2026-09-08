#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cobra
import pandas as pd
import os, re, traceback
from itertools import combinations
from cobra.flux_analysis import pfba
from math import ceil
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ======== Path setup ========
model_file = r"../models/iJO1366.xml"
input_dir = r"../data/mapped-gene-2_mapped"
output_dir = r"../data/imat_result_pairs-new"
pair_log_path = os.path.join(output_dir, "pair_log.xlsx")

# ======== Parameter setup ========
kappa = 0.025
rho = 0.025
epsilon = 1.0
keep_baseline_fraction = 0.20
down_threshold = -1.5
up_threshold = 1.5
batch_size = 5


# In[ ]:


# ======== Load model ========
os.makedirs(output_dir, exist_ok=True)
model = cobra.io.read_sbml_model(model_file)
orig_bounds = {r.id: (r.lower_bound, r.upper_bound) for r in model.reactions}

baseline_sol = model.optimize()
baseline_flux = pd.Series(baseline_sol.fluxes, name="Flux_baseline")

print(f"✅ 模型载入完成: {model.id}")
print(f"   反应数: {len(model.reactions)}, 基线生长率: {baseline_sol.objective_value:.6f}\n")


# In[ ]:


# ======== Extract drug names ========
all_files = [f for f in os.listdir(input_dir) if f.endswith("_genes_mapped.xlsx")]

file_info = {}
for f in all_files:
    m = re.search(r"TableS2-FinalData_([A-Za-z0-9\-]+?)(?:-[0-9]+)?\s*-", f)
    if m:
        base_drug = m.group(1)
        file_info[f] = base_drug
    else:
        print(f"⚠️ 无法识别药物名: {f}")


# In[ ]:


# ======== Load existing pair_log.xlsx directly ========
print(f"📂 正在加载已有组合记录: {pair_log_path}")
pair_log_df = pd.read_excel(pair_log_path, index_col=0)

# Rebuild file_pairs (for later batch processing)
file_pairs = list(zip(pair_log_df["FileA"], pair_log_df["FileB"]))

# If drug names or output filenames are needed later, they can be read from the DataFrame
# e.g. pair_log_df[["DrugA", "DrugB", "Output_File"]]

print(f"✅ 成功加载 {len(file_pairs)} 个组合\n")

# ======== Number of batches ========
total = len(file_pairs)
num_batches = ceil(total / batch_size)


# In[ ]:


def process_pair(fname1, fname2):
    try:
        import gc

        # Use the global model; do not reload the SBML
        local_model = model.copy # Reuse the globally loaded model

        # Reset the bounds of every reaction before each combination
        for rxn in local_model.reactions:
            lb, ub = orig_bounds[rxn.id]
            rxn.lower_bound = lb
            rxn.upper_bound = ub

        # Combination name and output file
        drugA = fname1.replace("TableS2-FinalData_", "").replace(" - _genes_mapped.xlsx", "").strip()
        drugB = fname2.replace("TableS2-FinalData_", "").replace(" - _genes_mapped.xlsx", "").strip()
        combo_name = f"{drugA}+{drugB}"
        # Replace special characters to avoid invalid filenames
        combo_name = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', combo_name)
        output_file = os.path.join(output_dir, f"flux_results_{combo_name}.xlsx")

        if os.path.exists(output_file):
            print(f"⚠️ 已存在结果: {combo_name}")
            return

        print(f"🔹 处理组合: {combo_name}")

        # ======== Read expression data ========
        dfA = pd.read_excel(os.path.join(input_dir, fname1))
        dfB = pd.read_excel(os.path.join(input_dir, fname2))
        for df in [dfA, dfB]:
            df.rename(columns={"YourGene": "ECK", "GEM_Gene_ID": "b_id", "Score": "fitness"}, inplace=True)
            df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")

        # ======== Classify gene state of drugs A and B ========
        def build_expr_dict(df):
            expr_dict = {}
            for _, row in df.iterrows():
                if pd.isna(row["fitness"]):
                    continue
                f = row["fitness"]
                if f <= down_threshold:
                    expr_dict[row["b_id"]] = "important"
                elif f >= up_threshold:
                    expr_dict[row["b_id"]] = "suppress"
                else:
                    expr_dict[row["b_id"]] = "neutral"
            return expr_dict

        expr_dictA = build_expr_dict(dfA)
        expr_dictB = build_expr_dict(dfB)

        del dfA, dfB
        gc.collect()

        # ======== Define combination gene-state function (weak_important -> neutral) ========
        def combine_gene_states(state1, state2):
            if state1 == "suppress" and state2 == "suppress":
                return "suppress"
            elif state1 == "important" and state2 == "important":
                return "important"
            elif (state1 == "suppress" and state2 == "important") or (state1 == "important" and state2 == "suppress"):
                return "partial"
            elif (state1 == "suppress" and state2 == "neutral") or (state1 == "neutral" and state2 == "suppress"):
                return "weak_suppress"
            # All remaining cases become neutral, including the former weak_important
            else:
                return "neutral"

        # ======== Apply combination constraints ========
        for rxn in local_model.reactions:
            genes = [g.id for g in rxn.genes if g.id in expr_dictA or g.id in expr_dictB]
            if not genes:
                continue

            combined_states = []
            for g in genes:
                stateA = expr_dictA.get(g, "neutral")
                stateB = expr_dictB.get(g, "neutral")
                combined_states.append(combine_gene_states(stateA, stateB))

            lb0, ub0 = orig_bounds[rxn.id]

            # All suppress: strong inhibition
            if all(s == "suppress" for s in combined_states):
                if rxn.upper_bound > 0:
                    rxn.upper_bound *= kappa
                if rxn.lower_bound < 0:
                    rxn.lower_bound *= kappa

            # All important: keep original bounds
            elif all(s == "important" for s in combined_states):
                rxn.lower_bound = lb0
                rxn.upper_bound = ub0

            # Other combination cases: partial, weak_suppress, neutral
            else:
                for s in combined_states:
                    if s == "partial":
                        rxn.lower_bound *= 0.5
                        rxn.upper_bound *= 0.5
                    elif s == "weak_suppress" and rxn.upper_bound > 0:
                        rxn.upper_bound *= (1 + kappa) / 2
                    # neutral or other cases: keep original bounds, no extra constraint

        # ======== Biomass handling ========
        biomass_rxns = [r for r in local_model.reactions if "biomass" in r.id.lower()]
        min_growth = baseline_sol.objective_value * keep_baseline_fraction
        for br in biomass_rxns:
            print(f"   ⚙️ biomass 上限: {br.upper_bound:.6f}, 参考下限: {min_growth:.6f}")
            # Optional: ensure the lower bound stays above a certain fraction
            # br.lower_bound = max(br.lower_bound, min_growth)

        # ======== Solve ========
        try:
            sol = pfba(local_model)
        except Exception as e:
            print(f"⚠️ pfba 出错，尝试 optimize: {e}")
            sol = local_model.optimize()

        if sol is None or sol.fluxes is None:
            print(f"⚠️ 求解失败，保存空表格: {combo_name}")
            flux_df = pd.DataFrame({"Reaction":[r.id for r in local_model.reactions]})
        else:
            if sol.objective_value < baseline_sol.objective_value * 0.01:
                print(f"⚠️ 组合 {combo_name} 生长率很低: {sol.objective_value:.6f}")

            condition_flux = pd.Series(sol.fluxes, name="Flux_condition")

            flux_df = pd.DataFrame({
                "Reaction": [r.id for r in local_model.reactions],
                "Name": [r.name for r in local_model.reactions],
                "Subsystem": [getattr(r, "subsystem", "") for r in local_model.reactions]
            })

            flux_df = flux_df.join(baseline_flux, on="Reaction").join(condition_flux, on="Reaction")
            flux_df["ΔFlux"] = flux_df["Flux_condition"] - flux_df["Flux_baseline"]

        flux_df.to_excel(output_file, index=False)
        print(f"   ✅ 已保存结果: {output_file} （包括低生长率组合）\n")

        # Manually free memory
        del expr_dictA, expr_dictB, combined_states, flux_df, condition_flux
        gc.collect()

    except Exception as e:
        print("\n❌ 出错组合:", fname1, "+", fname2)
        print("错误类型:", type(e).__name__)
        print("错误信息:", e)
        import traceback
        traceback.print_exc()


# In[ ]:


# %%
import psutil, os, gc, time, sys, json

process = psutil.Process(os.getpid())

# ===== Config parameters (monitoring only; no longer enforced) =====
PROGRESS_FILE = "progress_log.json"
start_batch = 0

# ===== Read last progress =====
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
        start_batch = progress.get("last_batch", 0)
        print(f"🔁 检测到上次进度，从第 {start_batch+1} 批继续。")
else:
    print("🚀 从头开始处理。")

print(f"🚦 初始内存: {process.memory_info().rss / 1024**3:.2f} GB")
start_time = time.time()

# ===== Batch processing =====
for batch_idx in range(start_batch, num_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, total)
    batch_pairs = file_pairs[start:end]

    print(f"\n🚀 开始第 {batch_idx+1}/{num_batches} 批（组合 {start+1} ~ {end}）...")
    print("-" * 60)

    for f1, f2 in batch_pairs:
        try:
            process_pair(f1, f2)
        except Exception as e:
            print(f"❌ 处理 {f1} + {f2} 时出错: {e}")
            continue  # Skip the error and continue to the next combination

    # ===== Manually free memory =====
    gc.collect()
    mem_now = process.memory_info().rss / 1024**3
    print(f"🎯 第 {batch_idx+1}/{num_batches} 批完成，共 {len(batch_pairs)} 个组合。")
    print(f"💾 当前内存占用: {mem_now:.2f} GB")
    print("=" * 80 + "\n")

    # ===== Save progress only; no memory check or exit =====
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_batch": batch_idx}, f)

# ===== After all batches complete =====
end_time = time.time()
print(f"🎉 所有文件组合处理完成！总用时 {(end_time - start_time)/60:.1f} 分钟。")

# Optional: remove the progress file (marks the task as finished)
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print("🧹 已清理进度文件。")

