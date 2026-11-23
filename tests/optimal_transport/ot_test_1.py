import numpy as np
import pandas as pd
from SOTS_IMTS.modules.optimal_transport.sliced_wasserstein import SlicedWasserstein


def load_signature_table(path: str) -> tuple:
    """
    Load a signature table from CSV.

    Args:
        path : path to CSV file with columns
               discrepancy_id; dim_0; dim_1; ...; dim_26
    Returns:
        ids : np.array of shape (n,), discrepancy_id column
        X   : np.array of shape (n, 27), signature points
    """
    df = pd.read_csv(path, sep=';')
    ids = df['discrepancy_id'].values
    X = df.drop(columns=['discrepancy_id']).values
    return ids, X


def random_coreset(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    """
    Sample a random coreset of size k from the full dataset.

    Args:
        X    : np.array of shape (n, d), full dataset
        k    : size of the coreset to sample
        seed : random seed for reproducibility
    Returns:
        np.array of shape (k, d)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx]


def main():
    # --- Step 1 : Load tables -------------------------------------------
    ids_full, X_full = load_signature_table(r"C:\Users\kakhmouch\SOTOS_IMTS\tests\optimal_transport\signature_tables\dataset.csv")     # full dataset
    _, C_given = load_signature_table(r"C:\Users\kakhmouch\SOTOS_IMTS\tests\optimal_transport\signature_tables\coreset.csv")            # given coreset

    print(f"Full dataset size      : {len(X_full)}")
    print(f"Given coreset size     : {len(C_given)}")

    # --- Step 2 : Sample two random coresets of different sizes --------
    k_random_1 = len(C_given)
    k_random_2 = len(C_given)

    rng_1 = np.random.default_rng(42)
    idx_random_1 = rng_1.choice(len(X_full), size=k_random_1, replace=False)
    C_random_1 = X_full[idx_random_1]
    ids_random_1 = ids_full[idx_random_1]

    rng_2 = np.random.default_rng(123)
    idx_random_2 = rng_2.choice(len(X_full), size=k_random_2, replace=False)
    C_random_2 = X_full[idx_random_2]
    ids_random_2 = ids_full[idx_random_2]

    print(f"Random coreset 1 size  : {len(C_random_1)}")
    print(f"Random coreset 1 ids   : {sorted(ids_random_1.tolist())}")
    print(f"Random coreset 2 size  : {len(C_random_2)}")
    print(f"Random coreset 2 ids   : {sorted(ids_random_2.tolist())}")

    # --- Step 3 : Compute Sliced Wasserstein distances ------------------
    sw = SlicedWasserstein(X_full, p=2)

    dist_given    = sw.compute(C_given,    L=500, method="exact")
    dist_random_1 = sw.compute(C_random_1, L=500, method="exact")
    dist_random_2 = sw.compute(C_random_2, L=500, method="exact")

    # --- Step 4 : Display results --------------------------------------
    print("\n--- Sliced Wasserstein distances to full dataset ---")
    print(f"Given coreset (t2)      [n={len(C_given):4d}] : SW = {dist_given:.6f}")
    print(f"Random coreset 1        [n={len(C_random_1):4d}] : SW = {dist_random_1:.6f}")
    print(f"Random coreset 2        [n={len(C_random_2):4d}] : SW = {dist_random_2:.6f}")


if __name__ == "__main__":
    main()