import numpy as np
import os
import shutil

from modules.coreset_selection.main import import_points
from modules.model.transformer import transformer
from modules.utilities.sql_utils import *
from modules.utilities.generate_dataset import generate_dataset


PERCENTAGES = [0.5, 1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
N_TOTAL = 14566


def main(signal_type, version, cs_method):

# =======================================================================================
# Paths
# =======================================================================================
    data_path    = "../data/"
    database_train_name = "database_training.db"
    database_val_name   = "database_validation.db"

    if signal_type == "binary":
        database_train_name = "database_training_binary.db"
        database_val_name   = "database_validation_binary.db"

    discrepancies_selected_path = f"../data/{signal_type}/{version}/discrepancies_selected.txt"
    dataset_val_path            = f"../data/{signal_type}/{version}/dataset_sots_val/"
    database_validation_path    = f"../data/{signal_type}/{version}/database_validation.db"
    dataset_val_name = "dataset_sots_val"
    dataset_name     = "dataset_sots"

    window_percentage     = 0.01
    max_time_serie_length = 400
    epochs          = 150
    model_batch_size = 64

# =======================================================================================
# Load selected ids (in selection order)
# =======================================================================================
    with open(discrepancies_selected_path, "r") as f:
        all_selected_ids = list(map(int, f.read().split()))

    print(f"Loaded {len(all_selected_ids)} selected discrepancy ids")
    print()

# =======================================================================================
# Generate validation dataset once
# =======================================================================================
    database_val_src = os.path.join(data_path, database_val_name)
    if not os.path.exists(database_validation_path):
        shutil.copy2(database_val_src, database_validation_path)

    DatabaseV = automatic_reports_analysis_db("v0", database_validation_path)
    if not os.path.exists(dataset_val_path):
        generate_dataset(
            DatabaseV, data_path, dataset_val_name,
            dataset_val_path, window_percentage,
            max_time_serie_length=max_time_serie_length
        )
        print("Validation dataset generated")
    else:
        print("Validation dataset already exists — skipping")
    print()

# =======================================================================================
# Train on each percentage checkpoint
# =======================================================================================
    database_train_src = os.path.join(data_path, database_train_name)

    for pct in PERCENTAGES:
        k_pct = max(1, int(round(pct / 100 * N_TOTAL)))
        ids_pct = all_selected_ids[:k_pct]

        pct_str = str(pct).replace(".", "p")  # e.g. "0.5" -> "0p5"

        # Paths specific to this percentage
        pct_folder           = f"../data/{signal_type}/{version}/{cs_method}_k_pcts/{pct_str}pct/"
        dataset_train_path   = os.path.join(pct_folder, "dataset/")
        database_selected_path = os.path.join(pct_folder, "database_selected.db")
        model_path           = f"../data/models/{signal_type}/{version}/{cs_method}_k{pct_str}pct/"

        os.makedirs(pct_folder, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        print(f"{'='*60}")
        print(f"Training on {pct}% coreset — k={k_pct} points")
        print(f"{'='*60}")

        # --- Build training database ---
        shutil.copy2(database_train_src, database_selected_path)
        Database = automatic_reports_analysis_db("v0", database_selected_path)
        Database.delete_table(dataset_name)
        Database.create_table_dataset(dataset_name)
        for x in ids_pct:
            Database.insert_time_series(dataset_name, x)
        print(f"Training database: {Database.len_table(dataset_name)} samples")

        # --- Generate training dataset ---
        generate_dataset(
            Database, data_path, dataset_name,
            dataset_train_path, window_percentage,
            max_time_serie_length=max_time_serie_length
        )
        print("Training dataset generated")

        # --- Train transformer ---
        transformer(
            id=f"{version}_{cs_method}_k{pct_str}pct",
            model_path=model_path,
            dataset_train_path=dataset_train_path,
            dataset_val_path=dataset_val_path,
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_classes=12,
            n_modalities=7,
            epochs=epochs,
            batch_size=model_batch_size,
            lr=1e-3,
            device="cuda"
        )
        print(f"Done, results saved in {model_path}")
        print()


if __name__ == "__main__":
    version     = int(input("Input version of the run: "))
    signal_type = input("Input signal_type ('binary' or 'analog'): ")
    cs_method   = input("Input cs_method (greedy/weighted/herding/facility_location/clustering/random): ")

    main(signal_type=signal_type, version=version, cs_method=cs_method)
