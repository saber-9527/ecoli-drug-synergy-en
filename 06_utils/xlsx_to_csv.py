# Convert _FLUX.xlsx -> _FLUX.csv only (the main data Part B needs)
import os, gc
import pandas as pd

OUTPUT_DIR = r"../data/continuous_bounded_moma_full-new"

xlsx_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_FLUX.xlsx")])
print(f"Found {len(xlsx_files)} _FLUX.xlsx files")

done = 0
for i, fname in enumerate(xlsx_files):
    csv_name = fname.replace(".xlsx", ".csv")
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    if os.path.exists(csv_path):
        done += 1
        continue
    df = pd.read_excel(os.path.join(OUTPUT_DIR, fname))
    df.to_csv(csv_path, index=False)
    del df
    done += 1
    if (i + 1) % 25 == 0:
        gc.collect()
        print(f"  {done}/{len(xlsx_files)}")

print(f"Done. {done}/{len(xlsx_files)} converted.")
