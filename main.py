import numpy as np
import os
import pickle
import torch

from modules.data_preprocessing.discrepancies_table import discrepancies_table
from modules.data_preprocessing.interpolation import Interpolator

from modules.signatures.signature_computer import *

from modules.coreset_selection.greedy_cs import GreedyCoresetSelector
from modules.coreset_selection.weighted_cs import WeightedCoresetSelector
from modules.coreset_selection.clustering import KMeansCoresetSelector
from modules.coreset_selection.herding import HerdingCoresetSelector
from modules.coreset_selection.facility_location import FacilityLocationCoresetSelector
from modules.coreset_selection.main import import_points

from modules.model.transformer import transformer

from modules.utilities.sql_utils import *
from modules.utilities.generate_dataset import generate_dataset
from modules.utilities.save_hyperparameters import *


def main(signal_type, version, resume_from, target_k):

# =======================================================================================
# Paths
# =======================================================================================
    data_path                   = "../data/"
    database_train_name         = "database_training.db"
    path_interpolation          = f"../data/{signal_type}/{version}/aligned_discrepancies.csv"
    path_signatures             = f"../data/{signal_type}/{version}/signatures.csv"
    discrepancies_path          = f"../data/{signal_type}/{version}/discrepancies.txt"
    discrepancies_selected_path = f"../data/{signal_type}/{version}/discrepancies_selected.txt"
    sw_path                     = f"../data/{signal_type}/{version}/sw_steps.txt"
    weights_path                = f"../data/{signal_type}/{version}/weights_steps.txt"
    hyperparams_path            = f"../data/{signal_type}/{version}/hyperparams.txt"
    checkpoint_path             = f"../data/{signal_type}/{version}/checkpoint.pkl"
    dataset_name                = "dataset_sots"

# =======================================================================================
# Hyperparameters
# =======================================================================================
    k                       = target_k
    L                       = 100
    n_iter_weighted         = 5000
    wasserstein_order       = 2
    grid_method             = "exact"
    cs_method               = "greedy"
    transport               = "sliced"
    eps_sinkhorn            = 0.1
    n_iter_sinkhorn         = 100
    sk_chunk_size           = 50
    cs_batch_size           = 500
    window_percentage       = 0.01
    signature_order         = 2
    projection_method       = "uniform"

    save_hyperparameters(
        hyperparams_path,
        transport=transport,
        k=k,
        resume_from=resume_from,
        target_k=target_k,
        L=L,
        projection_method=projection_method,
        n_iter_weighted=n_iter_weighted,
        wasserstein_order=wasserstein_order,
        signature_order=signature_order,
        grid_method=grid_method,
        cs_method=cs_method,
        cs_batch_size=cs_batch_size,
        window_percentage=window_percentage,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

# =======================================================================================
# Data_preprocessing --- skip if resuming
# =======================================================================================
    os.makedirs(f"../data/{signal_type}/{version}/", exist_ok=True)

    if signal_type == "binary":
        database_train_name = "database_training_binary.db"

    if resume_from == 0:
        # --- Discrepancies Table ---
        print()
        discrepancy_table = discrepancies_table(
            data_path=data_path,
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

        # --- Interpolation ---
        print()
        preprocessor = Interpolator()
        aligned = preprocessor.process(table)
        print(f"Aligned {len(aligned)} discrepancies (empty signals discarded)")
        preprocessor.export(aligned=aligned, output_path=path_interpolation)
        print()

        # --- Signatures ---
        computer = SignatureComputer(order=signature_order)
        aligned  = computer.extract(input_path=path_interpolation)
        X, original_indices = computer.process(aligned)
        print(f"Computed signatures : shape {X.shape}")
        computer.export(X, original_indices, path_signatures)
        print()

    else:
        print(f"Resuming from checkpoint k={resume_from}, skipping preprocessing and signatures")
        print()

# =======================================================================================
# Coreset Selection
# =======================================================================================

    original_indices, points = import_points(path_signatures)
    print(f"{len(points)} Points imported successfully")
    print()

    # ------------------------------------------------------------------
    # Sequential methods — support checkpointing
    # ------------------------------------------------------------------
    if cs_method in ["greedy", "weighted", "herding", "facility_location"]:

        if resume_from > 0:
            print(f"Loading checkpoint from {checkpoint_path} ...")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            selected_idx  = checkpoint["selected_idx"]
            remaining_idx = checkpoint["remaining_idx"]
            extra_state   = checkpoint.get("extra_state", {})
            print(f"Checkpoint loaded : {len(selected_idx)} points already selected")
            print()
        else:
            selected_idx  = []
            remaining_idx = list(range(len(points)))
            extra_state   = {}

        steps_to_do = target_k - resume_from

        if cs_method == "greedy":
            selector = GreedyCoresetSelector(
                points, p=wasserstein_order,
                transport=transport,
                eps=eps_sinkhorn,
                n_iter=n_iter_sinkhorn
            )
            selected_idx, history = selector.select(
                k=steps_to_do,
                L=L,
                method=grid_method,
                log_path=sw_path,
                candidate_batch_size=cs_batch_size,
                sinkhorn_chunk_size=sk_chunk_size,
                projection_method=projection_method,
                initial_selected_idx=selected_idx,
                initial_remaining_idx=remaining_idx,
                checkpoint_path=checkpoint_path,
            )

        elif cs_method == "weighted":
            selector = WeightedCoresetSelector(points, p=wasserstein_order)
            selected_idx, weights, history = selector.select(
                k=steps_to_do,
                L=L,
                sparsity=True,
                n_iter=n_iter_weighted,
                lr=0.05,
                log_path=weights_path,
                initial_selected_idx=selected_idx,
                initial_remaining_idx=remaining_idx,
                checkpoint_path=checkpoint_path,
            )

        elif cs_method == "herding":
            selector = HerdingCoresetSelector(points)
            selected_idx, history = selector.select(
                k=steps_to_do,
                log_path=sw_path,
                initial_selected_idx=selected_idx,
                initial_remaining_idx=remaining_idx,
                checkpoint_path=checkpoint_path,
            )

        elif cs_method == "facility_location":
            selector = FacilityLocationCoresetSelector(points)
            selected_idx, history = selector.select(
                k=steps_to_do,
                log_path=sw_path,
                initial_selected_idx=selected_idx,
                initial_remaining_idx=remaining_idx,
                extra_state=extra_state,
                checkpoint_path=checkpoint_path,
            )

    # ------------------------------------------------------------------
    # Non-sequential methods — no checkpointing needed
    # ------------------------------------------------------------------
    elif cs_method == "clustering":
        selector = KMeansCoresetSelector(points, random_state=42)
        selected_idx, cluster_labels = selector.select(k=target_k)

    elif cs_method == "random":
        rng = np.random.default_rng(seed=42)
        selected_idx = rng.choice(len(points), size=target_k, replace=False)

    else:
        raise ValueError(f"Unknown cs_method: {cs_method}")

    # --- Save selected discrepancy ids ---
    original_ids_selected = [original_indices[i] for i in selected_idx]
    print(f"\nFinal coreset size : {len(selected_idx)}")
    with open(discrepancies_selected_path, "w") as f:
        f.write(" ".join(map(str, original_ids_selected)))
    print("Selected discrepancies saved!")
    print()


if __name__ == "__main__":
    version     = int(input("Input version of the run: "))
    signal_type = input("Input signal_type ('binary' or 'analog'): ")
    resume_from = int(input("Input resume_from (0 if starting fresh): "))
    target_k    = int(input("Input target_k (e.g. 5000, 10000, 14566): "))

    main(signal_type=signal_type, version=version,
         resume_from=resume_from, target_k=target_k)