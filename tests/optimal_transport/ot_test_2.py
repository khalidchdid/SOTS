import numpy as np
import pandas as pd
from itertools import combinations
from SOTS_IMTS.modules.optimal_transport.sliced_wasserstein import SlicedWasserstein


def load_signature_table(path: str) -> tuple:
    """
    Load a signature table from CSV.

    Args:
        path : path to CSV file with columns
               discrepancy_id; dim_0; dim_1; ...; dim_26
    Returns:
        ids : np.array of shape (n,), discrepancy_id column
        X   : np.array of shape (n, d), signature points
    """
    df = pd.read_csv(path, sep=';')
    ids = df['discrepancy_id'].values
    X = df.drop(columns=['discrepancy_id']).values
    return ids, X


def main():
    # --- Step 1 : Load full dataset -------------------------------------
    ids_full, X_full = load_signature_table(r"C:\Users\kakhmouch\SOTOS_IMTS\tests\optimal_transport\signature_tables\dataset.csv")
    n_total = len(X_full)

    k = 10          # coreset size
    n_trials = 10000  # number of random subsets to test

    print(f"Full dataset size : {n_total}")
    print(f"Coreset size (k)  : {k}")
    print(f"Number of trials  : {n_trials}")

    # --- Step 2 : Sample n_trials random subsets of size k --------------
    rng = np.random.default_rng(0)
    sw = SlicedWasserstein(X_full, p=2)

    results = []  # list of (distance, indices)

    for trial in range(n_trials):
        idx = rng.choice(n_total, size=k, replace=False)
        C = X_full[idx]
        distance = sw.compute(C, L=200, method="exact")
        results.append((distance, idx))

    # --- Step 3 : Sort and select top 3 best and worst -------------------
    results.sort(key=lambda r: r[0])
    top_3_best = results[:3]
    top_3_worst = results[-3:][::-1]  # reversed so largest distance first

    # --- Step 4 : Display results ----------------------------------------
    print("\n--- Top 3 best coresets (lowest Sliced Wasserstein distance) ---")
    for rank, (distance, idx) in enumerate(top_3_best, start=1):
        ids_selected = sorted(ids_full[idx].tolist())
        print(f"\nRank {rank} — SW = {distance:.6f}")
        print(f"  Discrepancy ids : {ids_selected}")

    print("\n--- Top 3 worst coresets (highest Sliced Wasserstein distance) ---")
    for rank, (distance, idx) in enumerate(top_3_worst, start=1):
        ids_selected = sorted(ids_full[idx].tolist())
        print(f"\nRank {rank} — SW = {distance:.6f}")
        print(f"  Discrepancy ids : {ids_selected}")


if __name__ == "__main__":
    main()