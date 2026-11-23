import torch
import torch.nn.functional as F


class SlicedWassersteinTorch:
    """
    GPU-capable Sliced Wasserstein distance computations in the signature
    representation space, used by both greedy and weighted coreset
    selection.

    Two distinct use cases are supported, each with its own method:
      - compute_batch : exact SW_p between several unweighted candidate
                         coresets and the full dataset, evaluated in
                         parallel (used by GreedyCoresetSelector).
      - compute        : SW_p between a continuously weighted measure
                         (w on the simplex) and the full dataset, using
                         a differentiable grid quantile approximation
                         (used by WeightedCoresetSelector).

    These two cases are mathematically different (empirical CDF of a
    fixed-size unweighted sample vs. weighted CDF over the full dataset)
    and are therefore not unified into a single formula, even though
    both rely on sorting and quantile comparisons.

    Args:
        X      : np.array or torch.Tensor of shape (n, d), the full
                 dataset of signature points
        p      : order of the Wasserstein distance (default: 2)
        device : "cuda" or "cpu" (default: cuda if available)

    Example:
        sw = SlicedWassersteinTorch(X, p=2)

        # Greedy use case
        distances = sw.compute_batch(candidates, L=50)

        # Weighted use case
        loss = sw.compute(w, L=50)
    """

    def __init__(self, X, p: int = 2, device: str = None):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X.to(self.device)
        self.n, self.d = self.X.shape
        self.p = p

    # ------------------------------------------------------------------
    # Shared private methods
    # ------------------------------------------------------------------

    def _random_directions(self, L: int) -> torch.Tensor:
        """
        Sample L random directions uniformly on the unit sphere S^{d-1}.

        Args:
            L : number of directions
        Returns:
            torch.Tensor of shape (L, d)
        """
        directions = torch.randn(L, self.d, device=self.device)
        norms = directions.norm(dim=1, keepdim=True)
        return directions / norms


    @staticmethod
    def _wasserstein_1d_exact_batched(
        u: torch.Tensor,
        v: torch.Tensor,
        p: int,
    ) -> torch.Tensor:
        """
        Exact 1-D Wasserstein distance between two empirical measures of
        possibly different sizes, via breakpoints in the quantile domain
        (0,1). Supports arbitrary leading batch dimensions, as long as
        they match between u and v.

        Args:
            u : torch.Tensor of shape (..., k), projected samples
            v : torch.Tensor of shape (..., n), projected reference samples,
                same leading batch dimensions as u
            p : order of the Wasserstein distance
        Returns:
            torch.Tensor of shape (...), batch of W_p^p values
        """
        k = u.shape[-1]
        n = v.shape[-1]
        device = u.device

        u_sorted, _ = torch.sort(u, dim=-1)  # (..., k)
        v_sorted, _ = torch.sort(v, dim=-1)  # (..., n)

        # Rank positions (breakpoints), identical across the batch
        ranks_u = torch.arange(1, k + 1, device=device, dtype=torch.float32) / k
        ranks_v = torch.arange(1, n + 1, device=device, dtype=torch.float32) / n

        # Merge breakpoints: sorted union, batch-independent
        tau = torch.cat([ranks_u, ranks_v])
        tau, _ = torch.sort(tau)
        tau = torch.cat([torch.zeros(1, device=device), tau])  # (k+n+1,)

        # Indices into each sorted sequence at every breakpoint
        idx_u = torch.searchsorted(ranks_u, tau[1:])
        idx_v = torch.searchsorted(ranks_v, tau[1:])
        idx_u = torch.clamp(idx_u, max=k - 1)  # (k+n,)
        idx_v = torch.clamp(idx_v, max=n - 1)

        widths = tau[1:] - tau[:-1]  # (k+n,)

        # Gather along the last dimension, broadcasting over batch dims
        q_u = u_sorted[..., idx_u]  # (..., k+n)
        q_v = v_sorted[..., idx_v]  # (..., k+n)

        return torch.sum(torch.abs(q_u - q_v) ** p * widths, dim=-1)

    def _weighted_quantiles(
        self,
        sorted_proj: torch.Tensor,
        sorted_w: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the weighted quantile function at points t, given
        already-sorted projected values and their corresponding weights.

        Args:
            sorted_proj : torch.Tensor of shape (n,), sorted projections
            sorted_w    : torch.Tensor of shape (n,), weights reordered
                          to match sorted_proj
            t           : torch.Tensor of shape (M,), quantile levels in (0,1)
        Returns:
            torch.Tensor of shape (M,), weighted quantile values at t
        """
        cum_w = torch.cumsum(sorted_w, dim=0)
        idx = torch.searchsorted(cum_w, t)
        idx = torch.clamp(idx, max=len(sorted_proj) - 1)
        return sorted_proj[idx]

    def _wasserstein_1d_grid_weighted(self, proj, w, M):
        n = len(proj)
        sorted_proj, sort_idx = torch.sort(proj)
        sorted_w = w[sort_idx]

        uniform_ranks = (torch.arange(n, device=self.device, dtype=torch.float32) + 0.5) / n
        t = (torch.arange(M, device=self.device, dtype=torch.float32) + 0.5) / M

        cum_w = torch.cumsum(sorted_w, dim=0)           # (n,)
        cum_w_prev = torch.cat([torch.zeros(1, device=self.device), cum_w[:-1]])  # (n,)

        # Soft indicator via sigmoid - differentiable approximation of the hard indicator
        # sharpness controls how close to hard indicator (higher = sharper)
        sharpness = 1000.0
        indicator = torch.sigmoid(sharpness * (cum_w.unsqueeze(1) - t.unsqueeze(0))) - \
                    torch.sigmoid(sharpness * (cum_w_prev.unsqueeze(1) - t.unsqueeze(0)))  # (n, M)

        q_weighted = (sorted_proj.unsqueeze(1) * indicator).sum(dim=0)  # (M,)

        idx_uniform = torch.clamp(torch.searchsorted(uniform_ranks, t), max=n - 1)
        q_uniform = sorted_proj[idx_uniform].detach()   # reference — no grad needed

        return torch.mean(torch.abs(q_weighted - q_uniform) ** self.p)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        candidates: torch.Tensor,
        L: int = 50,
        method: str = "uniform",
        kappa: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute SW distance between each candidate coreset and the full
        dataset X, using one of four projection strategies.

        Args:
            candidates : torch.Tensor of shape (n_cand, k_cur, d)
            L          : number of projection directions
            method     : "uniform"  — standard SW, fully batched GPU
                        "ebsw"    — uniform directions + importance
                                    weighting, fully batched GPU
                        "rpsw"    — random-path directions + uniform
                                    average, loop over candidates
                        "iwrpsw"  — random-path directions + importance
                                    weighting, loop over candidates
            kappa      : concentration of Power Spherical distribution
                        (used by rpsw and iwrpsw only)
        Returns:
            torch.Tensor of shape (n_cand,), SW distance per candidate
        """
        candidates = candidates.to(self.device)
        n_cand, k_cur, d = candidates.shape

        # ------------------------------------------------------------------
        # 1. Uniform SW — fully batched
        # ------------------------------------------------------------------
        if method == "uniform":
            directions = self._random_directions(L)      # (L, d)

            proj_C = torch.einsum('cid,ld->cli', candidates, directions)    # (n_cand, L, k_cur)
            proj_X = (self.X @ directions.T).T           # (L, n)

            sw_per_dir = self._wasserstein_1d_exact_batched(
                proj_C, proj_X, p=self.p
            )  # (n_cand, L)

            return sw_per_dir.mean(dim=1) ** (1.0 / self.p)

        # ------------------------------------------------------------------
        # 2. EBSW — uniform directions + importance weighting, fully batched
        #    Directions are independent of the candidate — same batching
        #    strategy as uniform.
        #    (Nguyen & Ho, NeurIPS 2023)
        # ------------------------------------------------------------------
        elif method == "ebsw":
            directions = self._random_directions(L)      # (L, d)

            proj_C = torch.einsum('cid,ld->cli', candidates, directions)        # (n_cand, L, k_cur)
            proj_X = (self.X @ directions.T).T           # (L, n)

            w1d = self._wasserstein_1d_exact_batched(
                proj_C, proj_X, p=self.p
            )  # (n_cand, L)

            # Importance weights f(W_p^p) per candidate, normalized over L
            weights = torch.exp(w1d)                     # (n_cand, L)
            weights_norm = weights / weights.sum(dim=1, keepdim=True)

            return (w1d * weights_norm).sum(dim=1) ** (1.0 / self.p)

        # ------------------------------------------------------------------
        # 3. RPSW — random-path directions + uniform average
        #    Directions depend on the candidate — cannot batch over n_cand.
        #    (Nguyen et al., ICML 2024, Algorithm 1)
        # ------------------------------------------------------------------
        elif method == "rpsw":
            distances = torch.zeros(n_cand, device=self.device)

            for i in range(n_cand):
                C_i = candidates[i]                      # (k_cur, d)
                directions = self._random_path_directions(C_i, L, kappa)
                # (L, d)

                proj_C = C_i @ directions.T              # (k_cur, L)
                proj_X = (self.X @ directions.T).T       # (L, n)

                w1d = self._wasserstein_1d_exact_batched(
                    proj_C.T.unsqueeze(0),               # (1, L, k_cur)
                    proj_X.unsqueeze(0),                 # (1, L, n)
                    p=self.p
                ).squeeze(0)                             # (L,)

                distances[i] = w1d.mean() ** (1.0 / self.p)

            return distances

        # ------------------------------------------------------------------
        # 4. IWRPSW — random-path directions + importance weighting
        #    Directions depend on the candidate — cannot batch over n_cand.
        #    (Nguyen et al., ICML 2024, Algorithm 2)
        # ------------------------------------------------------------------
        elif method == "iwrpsw":
            distances = torch.zeros(n_cand, device=self.device)

            for i in range(n_cand):
                C_i = candidates[i]                      # (k_cur, d)
                directions = self._random_path_directions(C_i, L, kappa)
                # (L, d)

                proj_C = C_i @ directions.T              # (k_cur, L)
                proj_X = (self.X @ directions.T).T       # (L, n)

                w1d = self._wasserstein_1d_exact_batched(
                    proj_C.T.unsqueeze(0),               # (1, L, k_cur)
                    proj_X.unsqueeze(0),                 # (1, L, n)
                    p=self.p
                ).squeeze(0)                             # (L,)

                weights = torch.exp(w1d)
                weights_norm = weights / weights.sum()
                distances[i] = (w1d * weights_norm).sum() ** (1.0 / self.p)

            return distances

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: 'uniform', 'ebsw', 'rpsw', 'iwrpsw'."
            )


    def compute(
        self,
        w: torch.Tensor,
        L: int = 50,
        M: int = None,
    ) -> torch.Tensor:
        """
        Compute SW_p^p(w-weighted measure, uniform measure on X),
        differentiable with respect to w. Used as the optimization loss
        in WeightedCoresetSelector.

        Args:
            w : torch.Tensor of shape (n,), weights on the simplex
            L : number of random projection directions
            M : number of grid points for the quantile approximation
                (default: n)
        Returns:
            torch.Tensor scalar, SW_p^p (not raised to 1/p; used directly
            as the optimization loss)
        """
        if M is None:
            M = 200

        w = w.to(self.device)
        directions = self._random_directions(L)
        proj_X = self.X @ directions.T  # (n, L)

        losses = []
        for l in range(L):
            losses.append(self._wasserstein_1d_grid_weighted(proj_X[:, l], w, M))
        
        return torch.stack(losses).mean()