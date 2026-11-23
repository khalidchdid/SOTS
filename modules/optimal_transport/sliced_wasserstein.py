import numpy as np

class SlicedWasserstein:
    """
    Computes the Sliced Wasserstein distance between the full dataset
    distribution and a candidate coreset distribution, in the signature
    representation space R^27.

    The full dataset X is fixed at initialization. The coreset C is
    passed at each call, enabling repeated evaluation during greedy
    coreset selection.

    Args:
        X : np.array of shape (n, d), the full dataset of signature points
        p : order of the Wasserstein distance (default: 2)

    Example:
        sw = SlicedWasserstein(X, p=2)
        distance = sw.compute(C, L=500)
    """

    def __init__(self, X: np.ndarray, p: int = 2):
        self.X = X          # shape (n, d)
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.p = p

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _random_directions(d: int, L: int) -> np.ndarray:
        """
        Sample L random directions uniformly on the unit sphere S^{d-1}.

        Args:
            d : ambient dimension
            L : number of directions
        Returns:
            np.array of shape (L, d), each row a unit vector
        """
        directions = np.random.randn(L, d)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        return directions / norms

    @staticmethod
    def _project(points: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Project points onto a direction theta.

        Args:
            points : np.array of shape (m, d)
            theta  : np.array of shape (d,), unit vector
        Returns:
            np.array of shape (m,), scalar projections <theta, point>
        """
        return points @ theta

    @staticmethod
    def _wasserstein_1d_exact(u: np.ndarray, v: np.ndarray, p: int) -> float:
        """
        Compute the exact 1-D Wasserstein distance between two empirical
        measures of possibly different sizes, via breakpoints in the
        quantile domain (0,1).

        The empirical CDF of k sorted points has jumps at rank positions
        {1/k, 2/k, ..., 1}. Merging the rank positions of both samples
        gives the breakpoints where min(F_u, F_v) changes value; on each
        resulting interval the quantile functions F_u^{-1} and F_v^{-1}
        are constant, so the integral reduces to a finite sum.

        Args:
            u : np.array of shape (k,), projected coreset samples
            v : np.array of shape (n,), projected dataset samples
            p : order of the Wasserstein distance
        Returns:
            float, W_p^p(theta_# mu_C, theta_# mu_X) for this direction
        """
        k, n = len(u), len(v)

        u_sorted = np.sort(u)
        v_sorted = np.sort(v)

        # Rank positions (breakpoints) of each empirical CDF in (0,1]
        ranks_u = np.arange(1, k + 1) / k
        ranks_v = np.arange(1, n + 1) / n

        # Merge breakpoints: sorted union of both rank sequences
        tau = np.sort(np.concatenate([ranks_u, ranks_v]))
        tau = np.concatenate([[0.0], tau])  # prepend 0 for interval widths

        # For each tau value, find which sample point is "active"
        # i.e. the smallest sorted value whose rank >= tau
        idx_u = np.searchsorted(ranks_u, tau[1:], side='left')
        idx_v = np.searchsorted(ranks_v, tau[1:], side='left')
        idx_u = np.clip(idx_u, 0, k - 1)
        idx_v = np.clip(idx_v, 0, n - 1)

        q_u = u_sorted[idx_u]
        q_v = v_sorted[idx_v]

        # Interval widths between consecutive breakpoints
        widths = np.diff(tau)

        return np.sum(np.abs(q_u - q_v) ** p * widths)

    @staticmethod
    def _wasserstein_1d_grid(u: np.ndarray, v: np.ndarray, p: int, M: int = None) -> float:
        """
        Compute the 1-D Wasserstein distance via a uniform grid
        approximation of the quantile functions (linear interpolation
        between sorted samples).

        Args:
            u : np.array of shape (k,), projected coreset samples
            v : np.array of shape (n,), projected dataset samples
            p : order of the Wasserstein distance
            M : number of grid points (default: max(k, n))
        Returns:
            float, approximate W_p^p(theta_# mu_C, theta_# mu_X)
        """
        k, n = len(u), len(v)
        if M is None:
            M = max(k, n)

        t = (np.arange(M) + 0.5) / M  # midpoints, avoids t=0 and t=1

        q_u = np.quantile(u, t)
        q_v = np.quantile(v, t)

        return np.mean(np.abs(q_u - q_v) ** p)

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def compute(
        self,
        C: np.ndarray,
        L: int = 500,
        method: str = "exact",
    ) -> float:
        """
        Compute the Sliced Wasserstein distance SW_p(mu_C, mu_X) via
        Monte Carlo averaging over L random directions.

        Args:
            C      : np.array of shape (k, d), the candidate coreset
            L      : number of random projection directions
            method : "exact" (breakpoints, default) or "grid" (approximation)
        Returns:
            float, the estimated SW_p(mu_C, mu_X)
        """
        directions = self._random_directions(self.d, L)

        total = 0.0
        for theta in directions:
            u = self._project(C, theta)
            v = self._project(self.X, theta)

            if method == "exact":
                total += self._wasserstein_1d_exact(u, v, self.p)
            elif method == "grid":
                total += self._wasserstein_1d_grid(u, v, self.p)
            else:
                raise ValueError(f"Unknown method: {method}")

        sw_p_p = total / L
        return sw_p_p ** (1.0 / self.p)