import sqlite3
import pandas as pd
import numpy as np
import shutil
from modules.utilities.sql_utils import *
import os
"""

def sample_balanced(
    db_path: str,
    n_per_class: int = 50,
    label_col: str = "label",
    seed: int = 42,
    output_path: str = "balanced_sample_indices.txt",
) -> pd.DataFrame:

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT discrepancy_id, label FROM discrepancies INNER JOIN reports_inventory ON reports_inventory.signal_id = discrepancies.signal_id WHERE reports_inventory.signal_type = 0", 
        conn
    )
    conn.close()

    rng = np.random.default_rng(seed)
    selected = []

    for label, group in df.groupby(label_col):
        if len(group) >= n_per_class:
            idx = rng.choice(len(group), size=n_per_class, replace=False)
            sample = group.iloc[idx]
            selected.append(sample)

    result = pd.concat(selected).sort_values("discrepancy_id").reset_index(drop=True)

    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Balanced sample — {n_per_class} examples per class\n")
        f.write(f"Total selected : {len(result)}\n")
        f.write("-" * 30 + "\n")
        for label, group in result.groupby(label_col):
            f.write(f"\nClass {label} ({len(group)} examples):\n")
            f.write(", ".join(group["discrepancy_id"].astype(str).tolist()) + "\n")

    print(f"Selected {len(result)} examples saved to {output_path}")

sample_balanced(
    db_path="/workspace/kakhmouch/data/database_analog_F1.db",
    n_per_class=20,
    label_col="label",
    seed = 42,
    output_path="/workspace/kakhmouch/SOTS_IMTS/experiments/results/validation_indices_binary.txt",
)



txt_path="/workspace/kakhmouch/SOTS_IMTS/experiments/results/validation_indices_binary.txt"

ids = []
with open(txt_path, "r") as f:
    for line in f:
        line = line.strip()
        # Skip header lines
        if not line or line.startswith("Balanced") or \
            line.startswith("Total") or line.startswith("-") or \
            line.startswith("Class"):
            continue
        # Parse comma-separated ids
        ids.extend([int(x.strip()) for x in line.split(",")])

database_path = "/workspace/kakhmouch/data/database_analog_F1.db"
database_validation_path = "/workspace/kakhmouch/data/database_validation_binary.db"
database_training_path = "/workspace/kakhmouch/data/database_training_binary.db"
dataset_val_name = "dataset_sots_val"
dataset_name = "dataset_sots"   

shutil.copy2(database_path, database_validation_path)

DatabaseV = automatic_reports_analysis_db("v0", database_validation_path)
DatabaseV.delete_table("dataset_analog_F1")

DatabaseV.create_table_dataset(dataset_val_name)
for x in ids:
    DatabaseV.insert_time_series(dataset_val_name, int(x))
"""

txt_path="/workspace/kakhmouch/SOTS_IMTS/experiments/results/validation_indices_binary.txt"
database_path = "/workspace/kakhmouch/data/database_analog_F1.db"
database_validation_path = "/workspace/kakhmouch/data/database_validation_binary.db"
database_training_path = "/workspace/kakhmouch/data/database_training_binary.db"
dataset_val_name = "dataset_sots_val"
dataset_name = "dataset_sots" 
output_path = "/workspace/kakhmouch/SOTS_IMTS/experiments/results/training_indices_binary.txt"

# Load all ids from database
conn = sqlite3.connect(database_path)
df = pd.read_sql_query(
        "SELECT discrepancy_id, label FROM discrepancies INNER JOIN reports_inventory ON reports_inventory.signal_id = discrepancies.signal_id WHERE reports_inventory.signal_type = 0", 
        conn
    )
conn.close()
all_ids = set(df["discrepancy_id"].astype(int).tolist())

# Load selected ids from .txt
selected_ids = set()
with open(txt_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("Balanced") or \
            line.startswith("Total") or line.startswith("-") or \
            line.startswith("Class"):
            continue
        selected_ids.update([int(x.strip()) for x in line.split(",")])

# Remaining = all - selected
remaining_ids = sorted(all_ids - selected_ids)
print(f"Total ids      : {len(all_ids)}")
print(f"Selected ids   : {len(selected_ids)}")
print(f"Remaining ids  : {len(remaining_ids)}")

# Save to .txt
os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
with open(output_path, "w") as f:
    f.write(f"Remaining discrepancy ids\n")
    f.write(f"Total remaining : {len(remaining_ids)}\n")
    f.write("-" * 30 + "\n")
    f.write(", ".join(str(x) for x in remaining_ids) + "\n")

print(f"Remaining ids saved to {output_path}")


with open(output_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("Remaining") or \
            line.startswith("Total") or line.startswith("-"):
            continue
        # Single line of comma-separated ids
        ids2 = [int(x.strip()) for x in line.split(",")]
        break


shutil.copy2(database_path, database_training_path)

Database = automatic_reports_analysis_db("v0", database_training_path)
Database.delete_table("dataset_analog_F1")

Database.create_table_dataset(dataset_name)

for x in ids2:
    Database.insert_time_series(dataset_name, int(x))