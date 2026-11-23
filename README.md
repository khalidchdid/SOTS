# SOTS — Signature and Optimal Transport for data Selection

> Optimal training data selection for Irregular Multivariate Time Series (IMTS) classification via path signatures and Sliced Wasserstein optimal transport.

---

## Overview

SOTS is a research framework developed during a research internship at FRAMATOME I&C. It addresses the following problem:

> Given a large annotated dataset of irregular multivariate time series, select a minimal subset (coreset) that preserves the downstream classification performance of a Transformer model — reducing annotation cost while maintaining model quality.

The method encodes each time series as a pair of truncated path signatures (values path + validity path), then selects a coreset by minimizing the Sliced Wasserstein distance between the empirical signature distributions of the coreset and the full dataset.

---

## Method

Each discrepancy $d_i$ is represented by a 27-dimensional point:

$$\Phi(d_i) = \left(t_i^{\text{start}}\; \rho(S^2(X_i^{\text{val}}))\; \rho(S^2(X_i^{\text{vld}}))\right) \in \mathbb{R}^{27}$$

where:
- $t_i^{\text{start}}$ is the normalized start time of the discrepancy
- $S^2$ is the order-2 truncated path signature with time augmentation
- $\rho$ is a post-signature rescaling ($k!$ per order)
- Values and validity paths are kept separate to preserve the coupled structure

The optimal coreset is then defined as:

$$\mathcal{C}^\star = \arg\min_{\mathcal{C} \subset  \mathcal{X} \; |\mathcal{C}|=k} \widetilde{W}_p \left(\hat{\mathcal{C}} \, \hat{\mathcal{X}}\right)$$

Two algorithms are implemented to solve this:
- **Greedy** — iterative GPU-batched exact Sliced Wasserstein minimization
- **Weighted** — continuous weight optimization via Adam on softmax-logits

---

## Project Structure

```
SOTS_IMTS/
│
├── main.py                     # End-to-end pipeline entry point
├── pyproject.toml              # Dependencies
├── setup.cfg
│
├── modules/
│   ├── data_preprocessing/     # Signal alignment, interpolation, normalization
│   ├── signatures/             # Truncated path signature computation
│   ├── optimal_transport/      # Sliced Wasserstein (NumPy + PyTorch/GPU)
│   ├── coreset_selection/      # Greedy and Weighted coreset selectors
│   ├── model/                  # Transformer classifier
│   └── utilities/              # Shared utilities
│
├── experiments/                # Results, curves, logs
├── tests/                      # Unit tests per module
├── docs/                       # Research notes, papers, presentation
└── scripts/                    # Setup scripts
```

> **Note:** The `data/` folder is local only and not tracked in the repository. See the [Data](#data) section below.

---

## Data

The data folder lives outside the repository and has the following structure:

```
data/
├── database_analog_F1.db       # Main database — source of discrepancy IDs,
│                               # start/end times, and labels
├── analog/                     # Processed outputs per analog signal version
│   ├── 1/
│   ├── 2/
│   └── ...
├── binary/                     # Processed outputs per binary signal version
│   └── 1/
├── models/                     # Trained model version metadata
│   ├── analog/
│   └── binary/
└── TC_ES063/                   # Raw signal folders (one per test campaign)
    ├── SA01/                   # Each subfolder contains .h5 signal files
    ├── SI02/                   # from which discrepancies are extracted
    └── ...
```

**The pipeline expects:**
1. The `.db` file as the source of discrepancy metadata (IDs, timestamps, labels)
2. The `.h5` signal files as the source of raw time series values
3. The `analog/` and `binary/` folders to store pipeline outputs

---

## Quick Start

```python
import numpy as np
from modules.data_preprocessing.src.discrepancies_table import discrepancies_table
from modules.data_preprocessing.src.interpolator import DiscrepancyPreprocessor
from modules.signatures.src.compute import SignatureComputer
from modules.coreset_selection.src.greedy_cs import GreedyCoresetSelector

# 1. Load raw discrepancies from the database
table = discrepancies_table(data_path, database_name).discrepancies_table

# 2. Align signals onto a common timeline
preprocessor = DiscrepancyPreprocessor()
aligned = preprocessor.process(table)

# 3. Compute signature points
computer = SignatureComputer(order=2)
original_indices = [idx for idx, _, _ in aligned]
points = [
    computer.compute_point(values_path, validity_path)
    for _, values_path, validity_path in aligned
]
X = np.array(points)  # shape (n, 27)

# 4. Select coreset via greedy Sliced Wasserstein
selector = GreedyCoresetSelector(X, p=2)
selected_idx, history = selector.select(k=200, L=100)

# 5. Retrieve original discrepancy IDs
original_ids_selected = [original_indices[i] for i in selected_idx]

# 6. Visualize SW evolution
selector.plot_history(history, save_path="experiments/results/sw_evolution.png")
```

---

## Modules

| Module | Description | Key classes |
|---|---|---|
| `data_preprocessing` | Signal loading, alignment, interpolation | `discrepancies_table`, `DiscrepancyPreprocessor` |
| `signatures` | Truncated path signature computation | `SignatureComputer` |
| `optimal_transport` | Sliced Wasserstein distance (CPU + GPU) | `SlicedWasserstein`, `SlicedWassersteinTorch` |
| `coreset_selection` | Greedy and weighted coreset selection | `GreedyCoresetSelector`, `WeightedCoresetSelector` |
| `model` | Vanilla Transformer classifier | — |

---

## References

- Chevyrev & Kormilitzin (2016) — *A Primer on the Signature Method in Machine Learning*
- Morrill et al. (2020) — *A Generalised Signature Method for Multivariate Time Series Feature Extraction*
- Peyré & Cuturi (2019) — *Computational Optimal Transport*
- Bonneel et al. (2015) — *Sliced and Radon Wasserstein Barycenters of Measures*

---

## License

See `LICENSE`.