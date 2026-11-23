# Module — `coreset_selection`

> Selects an optimal coreset of size $k$ by minimizing the Sliced Wasserstein distance between the coreset and the full dataset in the signature space.

---

## Files

| File | Description |
|---|---|
| `greedy_cs.py` | `GreedyCoresetSelector` — iterative greedy selection |
| `weighted_cs.py` | `WeightedCoresetSelector` — weight optimization via Adam *(upcoming)* |
| `main.py` | Runs greedy coreset selection on the full signature dataset |

---

## Classes

### `GreedyCoresetSelector`

Iteratively adds the point that most reduces $\widetilde{W}_p(\hat{\mu}_{\mathcal{C}}, \hat{\mu}_{\mathcal{X}})$. Automatically uses GPU (`SlicedWassersteinTorch`) if available, CPU (`SlicedWasserstein`) otherwise.

```python
from coreset_selection.greedy_cs import GreedyCoresetSelector

selector = GreedyCoresetSelector(X, p=2)
selected_idx, history = selector.select(k=200, L=100)
```

**Constructor args:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | `np.array (n, d)` | — | Full dataset of signature points |
| `p` | `int` | `2` | Wasserstein order |

**`select(k, L, method, verbose, log_path)`:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | — | Target coreset size |
| `L` | `int` | `50` | Number of SW projection directions |
| `method` | `str` | `"exact"` | CPU only — `"exact"` or `"grid"` |
| `verbose` | `bool` | `True` | Print progress every 10 steps |
| `log_path` | `str` | `None` | Path to `.txt` log file |

Returns `(selected_idx, history)` where `selected_idx` is a `np.array` of indices into `X` and `history` is the SW distance after each addition.

**`plot_history(history, save_path)`** — plots SW distance vs coreset size $k$.

---

### `WeightedCoresetSelector` *(upcoming)*

Optimizes a weight vector $w \in \Delta_n$ via Adam on softmax-logits, then hard-thresholds to the $k$ largest weights.

---

## GPU vs CPU

| | GPU path | CPU path |
|---|---|---|
| Backend | `SlicedWassersteinTorch` | `SlicedWasserstein` |
| Candidate evaluation | Batched in parallel | Sequential loop |
| SW method | Exact breakpoints | `"exact"` or `"grid"` |
| OOM handling | Chunked by `candidate_batch_size` | — |

---

## Usage

```python
from coreset_selection.greedy_cs import GreedyCoresetSelector

selector = GreedyCoresetSelector(X, p=2)
selected_idx, history = selector.select(
    k=200,
    L=100,
    log_path="coreset_selection/logs/sw_history.txt"
)

# Map back to original discrepancy IDs
original_ids_selected = [original_indices[i] for i in selected_idx]

# Plot SW evolution
selector.plot_history(history, save_path="experiments/results/sw_evolution.png")