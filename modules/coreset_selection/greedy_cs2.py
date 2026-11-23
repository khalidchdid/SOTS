import numpy as np
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
        projection_method: str = "uniform"
    ) -> tuple:
        if self.use_gpu:
            return self._select_gpu(k, L, method, verbose, log_path,
                                    candidate_batch_size, sinkhorn_chunk_size, projection_method)
        else:
            return self._select_cpu(k, L, method, verbose)

    def _select_gpu(self, k, L, method, verbose, log_path,
                    candidate_batch_size, sinkhorn_chunk_size, projection_method):
        remaining_idx = list(range(self.n))
        selected_idx = []
        history = []

        log_file = None
        if log_path:
            import os
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_file = open(log_path, 'w')
            log_file.write("step,candidate_idx,sw_distance\n")

        for step in range(k):
            n_cand = len(remaining_idx)
            k_cur = len(selected_idx)

            best_candidate = None
            best_distance = float('inf')

            for start in range(0, n_cand, candidate_batch_size):
                chunk_idx = remaining_idx[start:start + candidate_batch_size]
                chunk_size = len(chunk_idx)

                current = self.sw.X[selected_idx].unsqueeze(0).expand(chunk_size, -1, -1) \
                    if selected_idx else \
                    torch.empty(chunk_size, 0, self.d, device=self.sw.device)
                candidates = self.sw.X[chunk_idx].unsqueeze(1)
                trial_batch = torch.cat([current, candidates], dim=1)

                if self.transport == "sliced":
                    distances = self.sw.compute_batch(trial_batch, L=L, method=projection_method)
                elif self.transport == "sinkhorn":
                    distances = self.sw.compute_batch(
                        trial_batch, chunk_size=sinkhorn_chunk_size, method=projection_method
                    )

                local_best = torch.argmin(distances).item()
                local_best_distance = distances[local_best].item()

                if local_best_distance < best_distance:
                    best_distance = local_best_distance
                    best_candidate = chunk_idx[local_best]

                del trial_batch, distances, current, candidates
                torch.cuda.empty_cache()

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)
            history.append(best_distance)

            if verbose and ((step + 1) % 10 == 0 or step == 0):
                tag = "GPU-Sliced" if self.transport == "sliced" else "GPU-Sinkhorn"
                print(f"[{tag}] Step {step + 1:3d}/{k} — added {best_candidate:5d} "
                      f"— W = {best_distance:.6f}")

            if log_file and ((step + 1) % 10 == 0 or step == 0):
                log_file.write(f"{step + 1},{best_candidate},{best_distance:.6f}\n")
                log_file.flush()

        if log_file:
            log_file.close()

        return np.array(selected_idx), history

    # ------------------------------------------------------------------
    # CPU path — sequential evaluation of remaining candidates
    # ------------------------------------------------------------------

    def _select_cpu(self, k: int, L: int, method: str, verbose: bool) -> tuple:
        remaining_idx = list(range(self.n))
        selected_idx = []
        history = []

        for step in tqdm(range(k)):
            best_candidate = None
            best_distance = np.inf

            for candidate in remaining_idx:
                trial_idx = selected_idx + [candidate]
                C = self.sw.X[trial_idx]
                distance = self.sw.compute(C, L=L, method=method)

                if distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)
            history.append(best_distance)

            if verbose:
                print(f"[CPU] Step {step + 1:3d}/{k} — added {best_candidate:5d} "
                      f"— SW = {best_distance:.6f}")

        return np.array(selected_idx), history

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_history(self, history: list, save_path: str = None) -> None:
        """
        Plot the evolution of the Sliced/Sinkhorn Wasserstein distance as the
        coreset grows, step by step.

        Args:
            history   : list of float, SW distance after each greedy addition
            save_path : if provided, save the figure instead of displaying it
        """

        k_values = np.arange(1, len(history) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, history, marker='o', markersize=3, linewidth=1)
        plt.xlabel("Coreset size (k)")
        plt.ylabel("Sliced Wasserstein distance")
        plt.title("Evolution of SW distance during greedy coreset selection")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()