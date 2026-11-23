# Module — `model`

> Vanilla Transformer classifier for IMTS fault classification. Handles data loading, training, and evaluation.

---

## Files

| File | Description |
|---|---|
| `imts_transformer_dense.py` | Transformer architecture definition |
| `transformer.py` | Training loop |
| `unit_loader.py` | DataLoader — loads discrepancies and labels for training |
| `warmup_module.py` | Learning rate warmup scheduler |
| `evaluation_metrics.py` | Evaluation metrics *(F1 macro, F1 per class, confusion matrix, AUROC, AUPRC — upcoming)* |
| `main.py` | Entry point — trains and evaluates the model |

---

## Architecture

Vanilla Transformer classifier operating on irregular multivariate time series. Takes a sequence of $(t, x_1, x_2, v_1, v_2)$ values as input and outputs a probability distribution over 10 fault classes.

---

## Training

```python
# Handled via main.py
python -m modules.model.main \
    --data_path /path/to/data \
    --database_name database_analog_F1.db
```

The training loop in `transformer.py` supports :
- Learning rate warmup via `warmup_module.py`
- Validation loss and accuracy tracking per epoch
- Model checkpointing to `data/models/`

---

## Evaluation *(upcoming)*

`evaluation_metrics.py` will compute the following metrics for the four configurations (full dataset, coreset greedy, coreset weighted, random sampling) :

| Metric | Description |
|---|---|
| Accuracy | Global prediction accuracy |
| F1 macro | Unweighted average F1 — sensitive to rare classes |
| F1 per class | Per-class F1 — key metric for coreset evaluation |
| Confusion matrix | Full error breakdown across 10 classes |
| AUROC | Area under ROC curve |
| AUPRC | Area under Precision-Recall curve — most relevant for class imbalance |

---

## Usage

```python
from model.unit_loader import UnitLoader
from model.imts_transformer_dense import IMTSTransformerDense
from model.transformer import train

# Load data
loader = UnitLoader(data_path, database_name, selected_ids=original_ids_selected)

# Train on coreset
model = IMTSTransformerDense(...)
train(model, loader)