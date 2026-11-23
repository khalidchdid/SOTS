import numpy as np
import os
import shutil

from modules.data_preprocessing.discrepancies_table import discrepancies_table
from modules.data_preprocessing.interpolation import Interpolator

from modules.signatures.signature_computer import *

from modules.coreset_selection.greedy_cs import *
from modules.coreset_selection.weighted_cs import *
from modules.coreset_selection.clustering import *
from modules.coreset_selection.main import import_points

from modules.model.transformer import transformer

from modules.utilities.sql_utils import *
from modules.utilities.generate_dataset import generate_dataset
from modules.utilities.save_hyperparameters import *

def main(signal_type, version):

# =======================================================================================
# Paths
# =======================================================================================
    data_path = "../data/"                                                                          # parent data folder
    database_train_name = "database_training.db"                                                    # initial discrepancies database                                     
    database_val_name = "database_validation.db" 
    path_interpolation = f"../data/{signal_type}/{version}/aligned_discrepancies.csv"               # csv for the interpolated discrepancies
    path_signatures = f"../data/{signal_type}/{version}/signatures.csv"                             # csv for the signatures
    database_selected_path = f"../data/{signal_type}/{version}/database_selected.db"                # database containing the training dataset
    database_validation_path = f"../data/{signal_type}/{version}/database_validation.db"            # database containing the validation dataset
    discrepancies_path = f"../data/{signal_type}/{version}/discrepancies.txt"                       # all discrepancy ids that will be selected from
    discrepancies_selected_path = f"../data/{signal_type}/{version}/discrepancies_selected.txt"     # selected discrepancy ids
    sw_path = f"../data/{signal_type}/{version}/sw_steps.txt"
    weights_path = f"../data/{signal_type}/{version}/weights_steps.txt"
    hyperparams_path = f"../data/{signal_type}/{version}/hyperparams.txt"
    model_path = f"../data/models/{signal_type}/{version}"                                          # trained model
    dataset_train_path = f"../data/{signal_type}/{version}/dataset_sots/"                           # path of the training dataset
    dataset_val_path = f"../data/{signal_type}/{version}/dataset_sots_val/"                         # path of the validation dataset                                                      # dataset_name (3rd table) in the initial database
    dataset_name = "dataset_sots"                                                                   # name for the training dataset
    dataset_val_name = "dataset_sots_val"                                                           # name for the validation dataset                                       

# =======================================================================================
# Hyperparameters
# =======================================================================================
    k = 14566
    L = 100
    n_iter_weighted = 5000
    wasserstein_order = 2
    grid_method = "exact"
    cs_method = "greedy"
    transport = "sliced"
    eps_sinkhorn = 0.1
    n_iter_sinkhorn = 100
    sk_chunk_size = 50
    cs_batch_size = 500
    window_percentage = 0.01
    max_time_serie_length = 400
    epochs = 150
    model_batch_size = 64
    signature_order = 2
    projection_method = "uniform"


    save_hyperparameters(
        hyperparams_path,
        transport=transport,
        k=k,
        L=L,
        projection_method=projection_method,
        n_iter_weighted=n_iter_weighted,
        wasserstein_order=wasserstein_order,
        signature_order=signature_order,
        grid_method=grid_method,
        cs_method=cs_method,
        cs_batch_size=cs_batch_size,
        window_percentage=window_percentage,
        max_time_serie_length=max_time_serie_length,
        epochs=epochs,
        model_batch_size=model_batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

# =======================================================================================
# Data_preprocessing
# =======================================================================================

    os.makedirs(f"../data/{signal_type}/{version}/", exist_ok=True)

    # --- Discrepancies Table ---
    print()
    if signal_type == "binary":
        database_train_name = "database_training_binary.db"                                                    # initial discrepancies database                                     
        database_val_name = "database_validation_binary.db"

    discrepancy_table = discrepancies_table(data_path=data_path, 
                                database_name=database_train_name, 
                                dataset_name=dataset_name, 
                                window_percentage=window_percentage
                                )
    table = discrepancy_table.discrepancies_table
    discrepancies = [discrepancy[0] for discrepancy in table]
    with open(discrepancies_path, "w") as f:
        f.write(" ".join(map(str, discrepancies)))
    print(f"Loaded {len(table)} discrepancies")
    print()

    # --- Interpolated discrepancies table ---
    print()
    preprocessor = Interpolator()
    aligned = preprocessor.process(table)
    print(f"Aligned {len(aligned)} discrepancies (empty signals discarded)")
    print()
    
    preprocessor.export(aligned=aligned, output_path=path_interpolation)
    print()

# =======================================================================================
# Signature_computation
# =======================================================================================

    computer = SignatureComputer(order=signature_order)
    aligned = computer.extract(input_path=path_interpolation)
    X, original_indices = computer.process(aligned)
    print(f"Computed signatures — shape {X.shape}")
    print()
    
    computer.export(X, original_indices, path_signatures)

# =======================================================================================
# Coreset Selection
# =======================================================================================

    original_indices, points = import_points(path_signatures)
    print()
    print(f"{len(points)} Points imported successfully")
    print()

    # --- Discrepancies Selection ---
    if cs_method == "greedy":
        selector = GreedyCoresetSelector(points, p=wasserstein_order, transport=transport, eps=eps_sinkhorn, n_iter=n_iter_sinkhorn)
        selected_idx, history = selector.select(k=k, L=L, method=grid_method, log_path=sw_path, 
                                                candidate_batch_size=cs_batch_size, 
                                                sinkhorn_chunk_size=sk_chunk_size,
                                                projection_method=projection_method)
    elif cs_method == "weighted":
        selector = WeightedCoresetSelector(points, p=wasserstein_order)
        selected_idx, weights, history = selector.select(k=k, L=L, sparsity=True, 
                                                         n_iter=n_iter_weighted, lr=0.05, 
                                                         log_path=weights_path)
    elif cs_method == "clustering":
        selector = KMeansCoresetSelector(points, random_state=42)
        selected_idx, cluster_labels = selector.select(k=k)
    else:
        rng = np.random.default_rng(seed=42)
        selected_idx = rng.choice(len(points), size=k, replace=False)

    original_ids_selected = [original_indices[i] for i in selected_idx]

    print(f"\nFinal coreset size : {len(selected_idx)}")
    print()

    with open(discrepancies_selected_path, "w") as f:
        f.write(" ".join(map(str, original_ids_selected)))
    
    print("Selected discrepancies saved!")

# =======================================================================================
# Generating Training/Validation datasets
# =======================================================================================

    database_train_path = os.path.join(data_path, database_train_name)
    database_val_path = os.path.join(data_path, database_val_name)
    shutil.copy2(database_train_path, database_selected_path)
    shutil.copy2(database_val_path, database_validation_path)

    # --- Training Dataset ---
    Database = automatic_reports_analysis_db("v0", database_selected_path)
    Database.delete_table(dataset_name)

    Database.create_table_dataset(dataset_name)

    with open(discrepancies_selected_path, "r") as f:
        for x in f.read().split():
            Database.insert_time_series(dataset_name, int(x))
    
    print(f"Training database updated successfully !")
    print(f"{Database.len_table(dataset_name)} training samples")

    generate_dataset(Database, data_path, dataset_name, dataset_train_path, window_percentage, max_time_serie_length=max_time_serie_length)

    print(f"Training dataset generated successfully !")

    # --- Validation Dataset ---
    DatabaseV = automatic_reports_analysis_db("v0", database_validation_path)

    print(f"{DatabaseV.len_table(dataset_val_name)} validation samples")

    generate_dataset(DatabaseV, data_path, dataset_val_name, dataset_val_path, window_percentage, max_time_serie_length=max_time_serie_length)

    print(f"Validation dataset generated successfully !")

# =======================================================================================
# Training Model
# =======================================================================================

    transformer(
        id=version, 
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
        device="cuda")

if __name__ == "__main__":
    version = int(input("Input version of the run: "))
    signal_type = input("Input signal_type('binary' for binary, 'analog' for analog, leave blank if you only want the discrepancies in the dataset present in the initial database): ")

    main(signal_type=signal_type, version=version)


