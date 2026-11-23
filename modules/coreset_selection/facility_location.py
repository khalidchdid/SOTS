import numpy as np
import torch
import os



class FacilityLocationCoresetSelector:
    """
    Coreset selection via Facility Location with RBF similarity.

    Greedily maximizes the submodular facility location function:

        f(C) = sum_{i=1}^{n} max_{j in C} s(Phi(d_i), Phi(d_j))

    where s(x, y) = exp(-||x - y||^2 / (2 * sigma^2)) is the RBF
    similarity and sigma is set automatically via the median heuristic:

        sigma = median({||x_i - x_j|| : i, j sampled from X})

    The greedy algorithm adds at each step the point x* that maximizes
    the marginal gain:

        x* = argmax_{x not in C} sum_i max(current_max_sim[i], s(x_i, x))

    Args:
        X           : np.array or torch.Tensor of shape (n, d)
        sigma       : RBF bandwidth. If None, computed via median
                      heuristic on a subsample of X (default: None)
        n_subsample : number of points used to estimate sigma via median
                      heuristic (default: 500)
        device      : "cuda" or "cpu" (default: cuda if available)

    Example:
        selector = FacilityLocationCoresetSelector(X)
        selected_idx, history = selector.select(k=200)
    """

    def __init__(
        self,
        X,
        sigma: float = None,
        n_subsample: int = 500,
        device: str = None,
    ):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X.to(self.device)
        self.n, self.d = self.X.shape

        # Compute sigma via median heuristic if not provided
        if sigma is None:
            self.sigma = self._median_heuristic(n_subsample)
            print(f"[FacilityLocation] sigma (median heuristic) = {self.sigma:.4f}")
        else:
            self.sigma = sigma

        self.S_full = self._rbf_similarity(self.X, self.X)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _median_heuristic(self, n_subsample: int) -> float:
        """
        Estimate sigma as the median pairwise Euclidean distance on a
        random subsample of X.

        Args:
            n_subsample : number of points to subsample
        Returns:
            float, estimated sigma
        """
        n_sub = min(n_subsample, self.n)
        idx = torch.randperm(self.n, device=self.device)[:n_sub]
        X_sub = self.X[idx]  # (n_sub, d)

        # Pairwise squared distances
        sq = (X_sub ** 2).sum(dim=1, keepdim=True)   # (n_sub, 1)
        dists_sq = sq + sq.T - 2 * X_sub @ X_sub.T   # (n_sub, n_sub)
        dists_sq = dists_sq.clamp(min=0)

        # Upper triangle only (no diagonal)
        mask = torch.triu(torch.ones(n_sub, n_sub, device=self.device), diagonal=1).bool()
        dists = dists_sq[mask].sqrt()

        return dists.median().item()

    def _rbf_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF similarity matrix between rows of A and rows of B.

        s(a, b) = exp(-||a - b||^2 / (2 * sigma^2))

        Args:
            A : torch.Tensor of shape (m, d)
            B : torch.Tensor of shape (p, d)
        Returns:
            torch.Tensor of shape (m, p)
        """
        sq_A = (A ** 2).sum(dim=1, keepdim=True)   # (m, 1)
        sq_B = (B ** 2).sum(dim=1, keepdim=True)   # (p, 1)
        dists_sq = sq_A + sq_B.T - 2 * A @ B.T     # (m, p)
        dists_sq = dists_sq.clamp(min=0)
        return torch.exp(-dists_sq / (2 * self.sigma ** 2))

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def select(
        self,
        k: int,
        verbose: bool = True,
        log_path: str = None,
        initial_selected_idx: list = None,
        initial_remaining_idx: list = None,
        extra_state: dict = None,
        checkpoint_path: str = None,
    ) -> tuple:
        import pickle

        selected_idx  = list(initial_selected_idx) if initial_selected_idx else []
        remaining_idx = list(initial_remaining_idx) if initial_remaining_idx \
                        else list(range(self.n))

        # Restore current_max_sim from extra_state if resuming
        if extra_state and "current_max_sim" in extra_state:
            current_max_sim = torch.tensor(
                extra_state["current_max_sim"], device=self.device
            )
            print("Restored current_max_sim from checkpoint")
        else:
            current_max_sim = torch.zeros(self.n, device=self.device)

        history = []

        log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
            log_file = open(log_path, "a" if initial_selected_idx else "w")
            if not initial_selected_idx:
                log_file.write("step,candidate_idx,facility_location_value\n")

        remaining_to_select = k - len(selected_idx)

        for step in range(remaining_to_select):
            global_step = len(selected_idx) + 1

            S = self.S_full[:, remaining_idx] # (n, n_cand)

            gains = torch.clamp(
                S - current_max_sim.unsqueeze(1), min=0
            ).sum(dim=0)  # (n_cand,)

            best_local     = torch.argmax(gains).item()
            best_candidate = remaining_idx[best_local]
            current_max_sim = torch.maximum(current_max_sim, S[:, best_local])
            f_value = current_max_sim.sum().item()

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)
            history.append(f_value)

            if verbose and (global_step % 10 == 0 or global_step == 1):
                print(f"[FacilityLocation] Step {global_step:5d} — "
                    f"added {best_candidate:5d} — "
                    f"f(C) = {f_value:.4f}")

            if log_file and (global_step % 10 == 0 or global_step == 1):
                log_file.write(f"{global_step},{best_candidate},{f_value:.6f}\n")
                log_file.flush()

            if checkpoint_path and (global_step % 10 == 0):
                os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "selected_idx" : selected_idx,
                        "remaining_idx": remaining_idx,
                        "extra_state"  : {
                            "current_max_sim": current_max_sim.cpu().numpy().tolist()
                        },
                    }, f)

        if log_file:
            log_file.close()

        return np.array(selected_idx), history