import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from modules.optimal_transport.sliced_wasserstein import SlicedWasserstein
from modules.optimal_transport.torch_sliced_wasserstein import SlicedWassersteinTorch
from modules.optimal_transport.torch_wasserstein import WassersteinTorch


class GreedyCoresetSelector:
    """
    Greedy coreset selection minimizing either Sliced Wasserstein (fast,
    recommended) or Sinkhorn-regularized Wasserstein (slower, more exact).

    Args:
        X         : np.array of shape (n, d)
        p         : order of the Wasserstein distance (default: 2)
        transport : "sliced" (default) or "sinkhorn"
        eps       : Sinkhorn regularization (only if transport="sinkhorn")
        n_iter    : Sinkhorn iterations (only if transport="sinkhorn")
    """

    def __init__(
        self,
        X: np.ndarray,
        p: int = 2,
        transport: str = "sliced",
        eps: float = 0.1,
        n_iter: int = 100,
    ):
        self.use_gpu = torch.cuda.is_available()
        self.n, self.d = X.shape
        self.p = p
        self.transport = transport

        if transport == "sliced":
            if self.use_gpu:
                self.sw = SlicedWassersteinTorch(X, p=p, device="cuda")
            else:
                self.sw = SlicedWasserstein(X, p=p)
        elif transport == "sinkhorn":
            device = "cuda" if self.use_gpu else "cpu"
            self.sw = WassersteinTorch(X, p=p, eps=eps, n_iter=n_iter, device=device)
        else:
            raise ValueError(f"Unknown transport: {transport}. Choose 'sliced' or 'sinkhorn'.")
        
    def select(
        self,
        k: int,
        L: int = 50,
        method: str = "exact",
        verbose: bool = True,
        log_path: str = None,
        candidate_batch_size: int = 200,
        sinkhorn_chunk_size: int = 50,
        projection_method: str = "uniform",
        kappa: float = 10.0,
        initial_selected_idx: list = None,
        initial_remaining_idx: list = None,
        checkpoint_path: str = None,
    ) -> tuple:
        """
        Greedily select a coreset of size k.

        Supports resuming from a checkpoint: pass initial_selected_idx and
        initial_remaining_idx loaded from a previous run, and set k to the
        number of additional steps to perform (target_k - resume_from).

        Args:
            k                    : number of steps to perform in this run
            L                    : number of SW projection directions
            method               : "exact" or "grid" (CPU path only)
            verbose              : print progress every 10 steps
            log_path             : path to .txt log file
            candidate_batch_size : GPU chunk size
            sinkhorn_chunk_size  : Sinkhorn chunk size
            projection_method    : "uniform", "ebsw", "rpsw", "iwrpsw"
            kappa                : concentration for rpsw/iwrpsw
            initial_selected_idx : list of already selected indices (resume)
            initial_remaining_idx: list of remaining indices (resume)
            checkpoint_path      : path to save/load checkpoint .pkl file
        Returns:
            selected_idx : np.array of all selected indices (resume + new)
            history      : list of SW distances for the new steps only
        """

        # Initialize or resume
        selected_idx  = list(initial_selected_idx) if initial_selected_idx else []
        remaining_idx = list(initial_remaining_idx) if initial_remaining_idx \
                        else list(range(self.n))

        history = []

        # Log file — append mode if resuming
        log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
            log_file = open(log_path, "a" if initial_selected_idx else "w")
            if not initial_selected_idx:
                log_file.write("step,candidate_idx,sw_distance\n")

        for step in range(k):
            global_step = len(selected_idx) + 1  # absolute step number

            if self.use_gpu:
                best_candidate, best_distance = self._step_gpu(
                    selected_idx, remaining_idx, L,
                    candidate_batch_size, projection_method, kappa,
                    sinkhorn_chunk_size
                )
            else:
                best_candidate, best_distance = self._step_cpu(
                    selected_idx, remaining_idx, L, method
                )

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)
            history.append(best_distance)

            if verbose and (global_step % 10 == 0 or global_step == 1):
                print(f"[Greedy] Step {global_step:5d} — "
                    f"added {best_candidate:5d} — "
                    f"SW = {best_distance:.6f}")

            if log_file and (global_step % 10 == 0 or global_step == 1):
                log_file.write(f"{global_step},{best_candidate},{best_distance:.6f}\n")
                log_file.flush()

            # Save checkpoint every 10 steps
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


    def _step_gpu(self, selected_idx, remaining_idx, L,
                candidate_batch_size, projection_method, kappa,
                sinkhorn_chunk_size):
        """Evaluate all remaining candidates in parallel on GPU and return best."""
        n_cand = len(remaining_idx)
        best_candidate = None
        best_distance  = float("inf")

        for start in range(0, n_cand, candidate_batch_size):
            chunk_idx  = remaining_idx[start:start + candidate_batch_size]
            chunk_size = len(chunk_idx)

            current = self.sw.X[selected_idx].unsqueeze(0).expand(chunk_size, -1, -1) \
                if selected_idx else \
                torch.empty(chunk_size, 0, self.d, device=self.sw.device)
            candidates = self.sw.X[chunk_idx].unsqueeze(1)
            trial_batch = torch.cat([current, candidates], dim=1)

            if self.transport == "sliced":
                distances = self.sw.compute_batch(
                    trial_batch, L=L, method=projection_method, kappa=kappa
                )
            elif self.transport == "sinkhorn":
                distances = self.sw.compute_batch(
                    trial_batch, chunk_size=sinkhorn_chunk_size
                )

            local_best = torch.argmin(distances).item()
            local_best_distance = distances[local_best].item()

            if local_best_distance < best_distance:
                best_distance  = local_best_distance
                best_candidate = chunk_idx[local_best]

            del trial_batch, distances, current, candidates
            torch.cuda.empty_cache()

        return best_candidate, best_distance


    def _step_cpu(self, selected_idx, remaining_idx, L, method):
        """Evaluate all remaining candidates sequentially on CPU."""
        best_candidate = None
        best_distance  = float("inf")

        for candidate in remaining_idx:
            trial_idx = selected_idx + [candidate]
            C = self.sw.X[trial_idx]
            distance = self.sw.compute(C, L=L, method=method)

            if distance < best_distance:
                best_distance  = distance
                best_candidate = candidate

        return best_candidate, best_distance
