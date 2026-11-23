import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pickle

class HerdingCoresetSelector:
    """
    Coreset selection via Feature Herding (Chen et al. 2010, DeepCore).

    Iteratively selects the point that minimizes the distance between
    the dataset mean and the current coreset mean in the signature
    space R^27.

    At each step:
        x* = argmin_{x not in C} || mu_X - (sum_{C} + x) / (|C| + 1) ||

    where mu_X = (1/n) sum_i Phi(d_i) is the full dataset mean.

    Args:
        X      : np.array or torch.Tensor of shape (n, d)
        device : "cuda" or "cpu" (default: cuda if available)

    Example:
        selector = HerdingCoresetSelector(X)
        selected_idx, history = selector.select(k=200)
    """

    def __init__(self, X, device: str = None):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X.to(self.device)
        self.n, self.d = self.X.shape

        # Full dataset mean — fixed throughout selection
        self.mu_X = self.X.mean(dim=0)  # (d,)

    def select(
        self,
        k: int,
        verbose: bool = True,
        log_path: str = None,
        initial_selected_idx: list = None,
        initial_remaining_idx: list = None,
        checkpoint_path: str = None,
    ) -> tuple:

        selected_idx  = list(initial_selected_idx) if initial_selected_idx else []
        remaining_idx = list(initial_remaining_idx) if initial_remaining_idx \
                        else list(range(self.n))

        # Rebuild coreset_sum from selected_idx if resuming
        if selected_idx:
            coreset_sum = self.X[selected_idx].sum(dim=0)
        else:
            coreset_sum = torch.zeros(self.d, device=self.device)

        history = []

        log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
            log_file = open(log_path, "a" if initial_selected_idx else "w")
            if not initial_selected_idx:
                log_file.write("step,candidate_idx,distance\n")

        for step in range(k):
            global_step = len(selected_idx) + 1

            n_cand = len(remaining_idx)
            candidates_sum  = coreset_sum.unsqueeze(0) + self.X[remaining_idx]
            candidates_mean = candidates_sum / global_step
            dists = torch.norm(self.mu_X.unsqueeze(0) - candidates_mean, dim=1)

            best_local     = torch.argmin(dists).item()
            best_candidate = remaining_idx[best_local]
            best_distance  = dists[best_local].item()

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)
            coreset_sum += self.X[best_candidate]
            history.append(best_distance)

            if verbose and (global_step % 10 == 0 or global_step == 1):
                print(f"[Herding] Step {global_step:5d} — "
                    f"added {best_candidate:5d} — "
                    f"dist = {best_distance:.6f}")

            if log_file and (global_step % 10 == 0 or global_step == 1):
                log_file.write(f"{global_step},{best_candidate},{best_distance:.6f}\n")
                log_file.flush()

            if checkpoint_path and (global_step % 10 == 0):
                os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "selected_idx" : selected_idx,
                        "remaining_idx": remaining_idx,
                        "extra_state"  : {},
                    }, f)

        if log_file:
            log_file.close()

        return np.array(selected_idx), history