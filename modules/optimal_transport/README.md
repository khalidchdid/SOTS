# Module — `optimal_transport`

> Computes the Sliced Wasserstein distance between empirical signature distributions. Provides both a CPU (NumPy) and a GPU (PyTorch) implementation.

---

## Files

| File | Description |
|---|---|
| `sliced_wasserstein.py` | CPU implementation — `SlicedWasserstein` (NumPy, exact breakpoints) |
| `torch_sliced_wasserstein.py` | GPU implementation — `SlicedWassersteinTorch` (PyTorch, exact + differentiable) |
| `main.py` | Standalone testing and benchmarking entry point |

---

## Classes

### `SlicedWasserstein` — CPU

```python
from optimal_transport.sliced_wasserstein import SlicedWasserstein

sw = SlicedWasserstein(X, p=2)
distance = sw.compute(C, L=100, method="exact")
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | `np.array (n, d)` | — | Full dataset of signature points |
| `p` | `int` | `2` | Wasserstein order |

| Method | Description |
|---|---|
| `compute(C, L, method)` | SW distance between coreset `C` and full dataset `X`. `method="exact"` uses breakpoints, `method="grid"` uses quantile grid approximation |

---

### `SlicedWassersteinTorch` — GPU

```python
from optimal_transport.torch_sliced_wasserstein import SlicedWassersteinTorch

sw = SlicedWassersteinTorch(X, p=2)
distances = sw.compute_batch(candidates, L=100)  # Greedy use case
loss = sw.compute(w, L=100)                       # Weighted use case
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | `np.array / Tensor (n, d)` | — | Full dataset of signature points |
| `p` | `int` | `2` | Wasserstein order |
| `device` | `str` | `"cuda"` if available | Computation device |

| Method | Output | Description |
|---|---|---|
| `compute_batch(candidates, L)` | `Tensor (n_cand,)` | Exact SW for a batch of candidate coresets — used by `GreedyCoresetSelector` |
| `compute(w, L, M)` | `Tensor scalar` | Differentiable SW for a weighted measure — used by `WeightedCoresetSelector` |

---

## 1D Wasserstein — Exact Breakpoint Method

Both implementations use the exact 1D Wasserstein formula via breakpoints in the quantile domain, valid for any $p \geq 1$ and any sample sizes $k \neq n$ :

$$W_p^p\left(\theta_\# \hat{\mu}_{\mathcal{C}},\, \theta_\# \hat{\mu}_{\mathcal{X}}\right) = \sum_{\ell=1}^{k+n-1} \left| F_{\mathcal{C}}^{-1}(\tau_\ell) - F_{\mathcal{X}}^{-1}(\tau_\ell) \right|^p \cdot (\tau_{\ell+1} - \tau_\ell)$$

where $\tau_1 < \cdots < \tau_{k+n}$ is the sorted union of rank positions $\{i/k\}$ and $\{j/n\}$.

This is implemented as the static method `_wasserstein_1d_exact_batched` in `SlicedWassersteinTorch`, supporting arbitrary leading batch dimensions — enabling fully parallelized evaluation over all candidates and directions simultaneously.

---

## Design Notes

**Two separate classes** — `SlicedWasserstein` (NumPy) and `SlicedWassersteinTorch` (PyTorch) are kept separate rather than unified, because their use cases are mathematically distinct:
- `compute_batch` operates on unweighted empirical measures of fixed size — exact breakpoint method
- `compute` operates on a continuously weighted measure with differentiable weights — grid quantile approximation (required for gradient flow through Adam)

**Automatic device selection** — `SlicedWassersteinTorch` defaults to CUDA if available, CPU otherwise.

**Monte Carlo estimation** — the integral over $\mathbb{S}^{d-1}$ is approximated by averaging over $L$ directions drawn uniformly:

$$\widetilde{W}_p(\hat{\mu}_{\mathcal{C}}, \hat{\mu}_{\mathcal{X}}) \approx \left(\frac{1}{L} \sum_{\ell=1}^{L} W_p^p(\theta_{\ell\#}\hat{\mu}_{\mathcal{C}}, \theta_{\ell\#}\hat{\mu}_{\mathcal{X}})\right)^{1/p}$$

$L = 100$ is a reasonable default for coreset selection loops; use $L \geq 500$ for final one-shot evaluation.

---

## Usage

```python
# CPU — one-shot evaluation
from optimal_transport.sliced_wasserstein import SlicedWasserstein

sw = SlicedWasserstein(X, p=2)
distance = sw.compute(C, L=500, method="exact")

# GPU — greedy loop
from optimal_transport.torch_sliced_wasserstein import SlicedWassersteinTorch

sw = SlicedWassersteinTorch(X, p=2)
distances = sw.compute_batch(trial_batch, L=100)  # (n_cand,)

More details can be found in : https://overleaf.enst.fr/project/6a2fb190291f36167ffce0ee