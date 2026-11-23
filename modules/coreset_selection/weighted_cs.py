import numpy as np
import os
import torch

from modules.optimal_transport.torch_sliced_wasserstein import SlicedWassersteinTorch


class WeightedCoresetSelector:
    """
    Coreset selection via continuous weight optimization on the simplex,
    minimizing the Sliced Wasserstein distance between the weighted
    measure and the full dataset's empirical measure.

    Always uses SlicedWassersteinTorch, since gradient-based optimization
    requires torch regardless of device - automatically runs on GPU if
    available, otherwise on CPU.

    Args:
        X : np.array of shape (n, d), the full dataset of signature points
        p : order of the Wasserstein distance (default: 2)

    Example:
        selector = WeightedCoresetSelector(X, p=2)
        idx_selected, weights, history = selector.select(
            k=200, sparsity=True, n_iter=500, L=50
        )
    """

    def __init__(self, X: np.ndarray, p: int = 2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sw = SlicedWassersteinTorch(X, p=p, device=device)
        self.n = X.shape[0]
        self.device = device

    def select(
        self,
        k: int,
        sparsity: bool = True,
        n_iter: int = 500,
        lr: float = 0.05,
        L: int = 50,
        verbose: bool = True,
        log_path: str = None,
        initial_selected_idx: list = None,
        initial_remaining_idx: list = None,
        checkpoint_path: str = None,
    ) -> tuple:
        import pickle

        history = []

        # Restore logits from checkpoint if resuming
        if checkpoint_path and os.path.exists(checkpoint_path) and initial_selected_idx:
            print(f"Loading weighted checkpoint from {checkpoint_path} ...")
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            logits_init = torch.tensor(
                ckpt["extra_state"]["logits"],
                device=self.device,
                dtype=torch.float32
            )
            start_iter = ckpt["extra_state"].get("iter", 0)
            print(f"Resuming Adam from iteration {start_iter}")
        else:
            logits_init = torch.zeros(self.n, device=self.device)
            start_iter  = 0

        logits    = logits_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=lr)
        remaining_iters = n_iter - start_iter

        log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
            log_file = open(log_path, "a" if start_iter > 0 else "w")

        for it in range(remaining_iters):
            global_it = start_iter + it + 1

            optimizer.zero_grad()
            w    = torch.softmax(logits, dim=0)
            loss = self.sw.compute(w, L=L)
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            if verbose and global_it % 50 == 0:
                device_tag = "GPU" if self.device == "cuda" else "CPU"
                print(f"[{device_tag}] Iter {global_it:4d}/{n_iter} — SW^p = {loss.item():.6f}")

            if log_file and global_it % 50 == 0:
                log_file.write(f"iter_{global_it}\n")
                log_file.flush()

            # Save checkpoint every 50 iterations
            if checkpoint_path and global_it % 50 == 0:
                os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "selected_idx" : [],
                        "remaining_idx": [],
                        "extra_state"  : {
                            "logits": logits.detach().cpu().numpy().tolist(),
                            "iter"  : global_it,
                        },
                    }, f)

        if log_file:
            log_file.close()

        w_final = torch.softmax(logits, dim=0).detach().cpu().numpy()

        if sparsity:
            selected_idx = np.argsort(w_final)[-k:][::-1]
            weights      = w_final[selected_idx]
            return selected_idx, weights, history
        else:
            return None, w_final, history
