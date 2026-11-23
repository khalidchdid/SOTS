import torch


class WassersteinTorch:
   """
   Sinkhorn approximation of the Wasserstein distance between the full
   dataset empirical measure and candidate coresets, using entropic
   regularization.

   Significantly more expensive than Sliced Wasserstein, recommended
   only for one-shot evaluation or greedy selection on reduced datasets.

   Args:
       X       : np.array or torch.Tensor of shape (n, d)
       p       : order of the Wasserstein distance (default: 2)
       eps     : entropic regularization parameter. Smaller = closer to
                 true Wasserstein but slower and less stable (default: 0.1)
       n_iter  : number of Sinkhorn iterations (default: 100)
       device  : "cuda" or "cpu" (default: cuda if available)

   Example:
       w = WassersteinTorch(X, p=2, eps=0.05, n_iter=200)
       distances = w.compute_batch(candidates, batch_size=50)
   """

   def __init__(self, X, p: int = 2, eps: float = 0.1,
                n_iter: int = 100, device: str = None):
       if not torch.is_tensor(X):
           X = torch.tensor(X, dtype=torch.float32)
       self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
       self.X = X.to(self.device)
       self.n, self.d = self.X.shape
       self.p = p
       self.eps = eps
       self.n_iter = n_iter

       # Uniform weights over the full dataset
       self.b = torch.full((self.n,), 1.0 / self.n, device=self.device)

   # ------------------------------------------------------------------
   # Private methods
   # ------------------------------------------------------------------

   def _cost_matrix(self, C: torch.Tensor) -> torch.Tensor:
       """
       Compute the pairwise cost matrix between coreset points and
       full dataset points.

       Args:
           C : torch.Tensor of shape (k, d), one candidate coreset
       Returns:
           torch.Tensor of shape (k, n), entry (i,j) = ||C_i - X_j||^p
       """
       # Squared Euclidean distances via expansion
       # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
       C_sq = (C ** 2).sum(dim=1, keepdim=True)          # (k, 1)
       X_sq = (self.X ** 2).sum(dim=1, keepdim=True).T   # (1, n)
       cross = C @ self.X.T                               # (k, n)
       dist_sq = C_sq + X_sq - 2 * cross
       dist_sq = dist_sq.clamp(min=0)

       if self.p == 2:
           return dist_sq
       elif self.p == 1:
           return dist_sq.sqrt()
       else:
           return dist_sq ** (self.p / 2)

   def _sinkhorn(self, M: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
       """
       Sinkhorn algorithm for entropic regularized optimal transport.

       Minimizes:
           <P, M> - eps * H(P)
       subject to P 1_n = a, P^T 1_k = b (= uniform over X)

       where H(P) = -sum_ij P_ij log P_ij is the entropy of P.

       Args:
           M : torch.Tensor of shape (k, n), cost matrix
           a : torch.Tensor of shape (k,), source weights (uniform over C)
       Returns:
           torch.Tensor scalar, Sinkhorn divergence W_eps(a, b)
       """
       k = M.shape[0]

       # Log-domain Sinkhorn for numerical stability
       log_K = -M / self.eps           # (k, n)
       log_a = a.log()                 # (k,)
       log_b = self.b.log()            # (n,)

       log_u = torch.zeros(k, device=self.device)
       log_v = torch.zeros(self.n, device=self.device)

       for _ in range(self.n_iter):
           log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
           log_v = log_b - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)

       # Transport plan in log domain
       log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)  # (k, n)
       P = log_P.exp()

       return (P * M).sum()

   # ------------------------------------------------------------------
   # Public methods
   # ------------------------------------------------------------------

   def compute(self, C: torch.Tensor) -> torch.Tensor:
       """
       Compute the Sinkhorn-regularized Wasserstein distance between
       a single coreset C and the full dataset X.

       Args:
           C : torch.Tensor of shape (k, d)
       Returns:
           torch.Tensor scalar
       """
       k = C.shape[0]
       a = torch.full((k,), 1.0 / k, device=self.device)
       M = self._cost_matrix(C)
       return self._sinkhorn(M, a)

   def compute_batch(
       self,
       candidates: torch.Tensor,
       chunk_size: int = 50,
   ) -> torch.Tensor:
       """
       Compute the Sinkhorn distance for a batch of candidate coresets,
       processed in chunks to avoid OOM.

       Args:
           candidates : torch.Tensor of shape (n_cand, k_cur, d)
           chunk_size : number of candidates evaluated simultaneously
       Returns:
           torch.Tensor of shape (n_cand,)
       """
       n_cand = candidates.shape[0]
       distances = torch.zeros(n_cand, device=self.device)

       for start in range(0, n_cand, chunk_size):
           end = min(start + chunk_size, n_cand)
           for i in range(start, end):
               distances[i] = self.compute(candidates[i])

       return distances