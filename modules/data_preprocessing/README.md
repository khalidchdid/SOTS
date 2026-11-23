# Module — `data_preprocessing`

> Loads raw discrepancies from the database and signal files, aligns the two signals of each discrepancy onto a common timeline, and prepares them for signature computation.

---

## Responsibilities

This module is the **first step** of the SOTS pipeline. It handles:

1. **Data loading** — querying the `.db` database for discrepancy metadata (IDs, timestamps, labels) and extracting the corresponding raw time series from `.h5` signal files
2. **Alignment** — building a common timeline for each pair of signals via step-function interpolation
3. **Filtering** — discarding discrepancies where one of the two signals is empty

---

## Files

| File | Description |
|---|---|
| `discrepancies_table.py` | Loads discrepancy metadata from the `.db` file and extracts raw signal values from `.h5` files |
| `interpolation.py` | Aligns two irregularly sampled signals onto a common timeline via step-function interpolation |
| `main.py` | Entry point for running the preprocessing module standalone |

---

## Classes

### `discrepancies_table`

Loads all discrepancies from the database and corresponding `.h5` signal files.

```python
from data_preprocessing.discrepancies_table import discrepancies_table

table = discrepancies_table(data_path, database_name).discrepancies_table
```

**Constructor args:**
| Argument | Type | Description |
|---|---|---|
| `data_path` | `str` | Path to the root data folder |
| `database_name` | `str` | Name of the `.db` file |

**Output — `.discrepancies_table`:**

A list of tuples, one per discrepancy:

```python
(t1, ve1, vy1, t2, ve2, vy2)
```

| Field | Type | Description |
|---|---|---|
| `t1` | `list[float]` | Timestamps of signal 1 |
| `ve1` | `list[float]` | Values of signal 1 |
| `vy1` | `list[int]` | Validity of signal 1 (`0` or `1`) |
| `t2` | `list[float]` | Timestamps of signal 2 |
| `ve2` | `list[float]` | Values of signal 2 |
| `vy2` | `list[int]` | Validity of signal 2 (`0` or `1`) |

---

### `DiscrepancyPreprocessor`

Aligns each discrepancy onto a common timeline and returns structured path tuples ready for signature computation.

```python
from data_preprocessing.interpolation import DiscrepancyPreprocessor

preprocessor = DiscrepancyPreprocessor()
aligned = preprocessor.process(table)
```

**Methods:**

#### `process(discrepancies)`

```python
aligned = preprocessor.process(table)
```

| Argument | Type | Description |
|---|---|---|
| `discrepancies` | `list[tuple]` | Output of `discrepancies_table.discrepancies_table` |

Returns a list of tuples `(original_index, values_path, validity_path)` where:
- `original_index` — index of the discrepancy in the original table (preserved through filtering)
- `values_path` — `list[tuple(float, float, float)]` — `(t, ve1, ve2)` on the common timeline
- `validity_path` — `list[tuple(float, float, float)]` — `(t, vy1, vy2)` on the common timeline

Discrepancies where either signal is empty are silently discarded.

#### `export(aligned, output_path)`

Exports the aligned discrepancies to a CSV file for inspection.

```python
preprocessor.export(aligned, "data_preprocessing/logs/aligned.csv")
```

Output columns: `discrepancy_id | t | ve1 | ve2 | vy1 | vy2`

---

## Preprocessing Pipeline

```
Raw discrepancy (t1, ve1, vy1, t2, ve2, vy2)
            │
            ▼
Filter empty signals
            │
            ▼
t_common = union(t1, t2), sorted
            │
            ▼
Step-function interpolation onto t_common
(value at t_k held constant until t_{k+1})
            │
            ▼
(original_index, values_path, validity_path)
```

---

## Design Notes

**Step-function interpolation:** for each timestamp in the common timeline, the interpolated value is the last observed value at or before that timestamp. This is the natural choice for industrial signals that hold their value between measurements.

**Common timeline:** built as the sorted union of both signals' timestamps — no information is lost, both signals are defined at every point of the common timeline.

**Original index tracking:** the `original_index` field maps each aligned discrepancy back to its position in the raw database table, allowing downstream modules to recover the original discrepancy IDs after filtering.

**Normalization:** timestamps and values are normalized using the min/max of the parent signal before signature computation. This is handled at the `discrepancies_table` level so that all values passed to `DiscrepancyPreprocessor` are already in `[0, 1]`.

---

## Usage in the Full Pipeline

```python
from data_preprocessing.discrepancies_table import discrepancies_table
from data_preprocessing.interpolation import DiscrepancyPreprocessor

# Load
table = discrepancies_table(data_path, database_name).discrepancies_table
print(f"Loaded {len(table)} raw discrepancies")

# Preprocess
preprocessor = DiscrepancyPreprocessor()
aligned = preprocessor.process(table)
print(f"Aligned {len(aligned)} discrepancies after filtering")

# Export for inspection
preprocessor.export(aligned, "data_preprocessing/logs/aligned.csv")

# Extract indices and paths for downstream use
original_indices = [idx for idx, _, _ in aligned]
values_paths     = [vp  for _, vp, _  in aligned]
validity_paths   = [vyp for _, _,  vyp in aligned]