# Module — `signatures`

> Computes truncated path signatures for each discrepancy, producing the point representation used in optimal transport coreset selection.

---

## Files

| File | Description |
|---|---|
| `signature_computer.py` | `SignatureComputer` class — core signature computation |
| `main.py` | Runs signature computation on the full aligned dataset and exports to CSV |

---

## Class — `SignatureComputer`

```python
from signatures.signature_computer import SignatureComputer

computer = SignatureComputer(order=2)
sig_values, sig_validity = computer.compute(values_path, validity_path)
point = computer.compute_point(values_path, validity_path)  # shape (27,)
```

**Constructor args:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `order` | `int` | `2` | Truncation order of the signature |

**Key methods:**

| Method | Output shape | Description |
|---|---|---|
| `compute(values_path, validity_path)` | `((13,), (13,))` | Two signature vectors separately |
| `compute_point(values_path, validity_path)` | `(27,)` | Concatenated OT point — `(t_start, sig_values, sig_validity)` |
| `export(points, output_path)` | — | Exports signature points to CSV |

---

## Point Representation

Each discrepancy is mapped to a point in $\mathbb{R}^{27}$:

$$\Phi(d_i) = \left(t_i^{\text{start}},\; \rho(S^2(X_i^{\text{val}})),\; \rho(S^2(X_i^{\text{vld}}))\right)$$

| Component | Dim | Description |
|---|---|---|
| $t_i^{\text{start}}$ | 1 | Normalized start time |
| $\rho(S^2(X^{\text{val}}))$ | 13 | Rescaled signature of $(t, x_1, x_2)$ |
| $\rho(S^2(X^{\text{vld}}))$ | 13 | Rescaled signature of $(t, v_1, v_2)$ |

---

## Design Notes

**Time augmentation** is applied by default — time is included as the first component of each path, making the signature sensitive to when variations occur, not just how much.

**Rescaling** — order-$k$ terms are multiplied by $k!$ to compensate for their natural decay of $O(1/k!)$, ensuring all components are on a comparable scale for the Euclidean distance used in OT.

**Order** — order 2 is the default and recommended setting. It captures total variations (order 1) and pairwise interactions between components including the Lévy area (order 2). Order 3 would capture higher-order dynamics but increases dimension exponentially ($d^3 = 27$ additional terms for $d=3$).

**Separate paths for values and validity** — values $(t, x_1, x_2)$ and validity $(t, v_1, v_2)$ are encoded as separate signatures rather than a single path in $\mathbb{R}^4$. This preserves the physical structure of the data: the nature of an error is sometimes conditioned by the validity, and the two paths encode different types of information.

---

## Usage

```python
from signatures.signature_computer import SignatureComputer

computer = SignatureComputer(order=2)
points = [
    computer.compute_point(vp, vyp)
    for vp, vyp in zip(values_paths, validity_paths)
]
X = np.array(points)  # shape (n, 27)
computer.export(points, "signatures/logs/signatures.csv")

More details can be found in : https://overleaf.enst.fr/read/zfhgmnxjvhfn#0886ce
