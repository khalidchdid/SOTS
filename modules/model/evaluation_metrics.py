"""Evaluation metrics for IMTS fault classification."""
import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate(model, loader, device, n_classes=10, criterion=None):
    """
    Validation loop returning all relevant metrics for imbalanced
    multi-class fault classification.

    Metrics returned:
    - val_loss      : mean cross-entropy loss over the validation set
    - acc           : accuracy
    - f1_macro      : F1 macro
    - f1_per_class  : F1 per class (np.array of shape (n_classes,))
    - auprc         : AUPRC macro
    - conf_matrix   : confusion matrix (np.array of shape (n_classes, n_classes))
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for feat, pad_mask, static_5, static_6, y in loader:
            feat     = feat.to(device)
            pad_mask = pad_mask.to(device)
            static_5 = static_5.to(device)
            static_6 = static_6.to(device)
            y        = y.to(device)

            logits = model(feat, pad_mask, static_5, static_6)
            loss   = criterion(logits, y)
            running_loss += loss.item() * y.size(0)

            probs = logits.softmax(dim=1)   # [batch, n_classes]
            preds = logits.argmax(dim=1)    # [batch]

            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)   # [N, n_classes]

    val_loss = running_loss / max(1, len(loader.dataset))

    # ------------------------------------------------------------------
    # 1. Accuracy
    # ------------------------------------------------------------------
    # Proportion of correctly predicted examples over all examples.
    # Simple to interpret but misleading with class imbalance: a model
    # predicting only the majority class can achieve high accuracy while
    # completely ignoring rare fault classes. Kept as a secondary metric.
    acc = accuracy_score(y_true, y_pred)

    # ------------------------------------------------------------------
    # 2. F1 Macro
    # ------------------------------------------------------------------
    # For each class c, F1_c = 2 * precision_c * recall_c /
    #                              (precision_c + recall_c)
    # where precision_c = TP_c / (TP_c + FP_c)
    #       recall_c    = TP_c / (TP_c + FN_c)
    #
    # F1 macro = unweighted mean of F1_c over all classes.
    # Because it is unweighted, each class contributes equally regardless
    # of its frequency : rare fault classes have the same weight as
    # frequent ones. This is the PRIMARY metric for comparing coreset
    # configurations in this study.
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ------------------------------------------------------------------
    # 3. F1 per class
    # ------------------------------------------------------------------
    # Same formula as above, computed separately for each class.
    # Allows identifying which specific fault classes are well or poorly
    # learned, critical for assessing whether the coreset preserves
    # rare classes. Returns an array of shape (n_classes,).
    f1_per_class = f1_score(
        y_true, y_pred, average=None,
        labels=list(range(n_classes)), zero_division=0
    )

    # ------------------------------------------------------------------
    # 4. AUPRC macro
    # ------------------------------------------------------------------
    # For each class c, the Precision-Recall curve plots precision vs
    # recall at all probability thresholds. AUPRC_c is the area under
    # this curve (computed as average precision, i.e. a weighted mean
    # of precisions at each threshold weighted by recall increases).
    # AUPRC macro = unweighted mean of AUPRC_c over present classes.
    #
    # More informative than AUROC for imbalanced datasets: the ROC curve
    # uses True Negative Rate which is trivially high when negatives
    # dominate, making AUROC look good even for bad models. The PR curve
    # focuses on positives only and better reflects performance on rare
    # classes. AUROC is therefore NOT reported in this study.
    present = np.unique(y_true)
    if present.size >= 2:
        y_bin_present   = label_binarize(y_true, classes=present)
        probs_present   = y_prob[:, present]
        auprc = average_precision_score(
            y_bin_present, probs_present, average='macro'
        )
    else:
        auprc = np.nan

    # ------------------------------------------------------------------
    # 5. Confusion matrix
    # ------------------------------------------------------------------
    # Entry (i, j) = number of examples of true class i predicted as
    # class j. Diagonal = correct predictions. Off-diagonal = errors.
    # Allows visualizing systematic confusions between fault classes
    # (e.g. class 3 always confused with class 7) and identifying which
    # classes benefit or suffer most from a given coreset configuration.
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(n_classes))
    )

    return val_loss, acc, f1_macro, f1_per_class, auprc, conf_matrix


def print_evaluation(val_loss, acc, f1_macro, f1_per_class, auprc, conf_matrix):
    """
    Print a structured summary of all evaluation metrics.
    """
    print("=" * 50)
    print(f"  Val Loss   : {val_loss:.4f}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  F1 Macro   : {f1_macro*100:.2f}%")
    print(f"  AUPRC      : {auprc:.4f}")
    print("-" * 50)
    print("  F1 per class :")
    for i, f1 in enumerate(f1_per_class):
        print(f"    Class {i:2d} : {f1*100:.2f}%")
    print("-" * 50)
    print("  Confusion matrix :")
    print(conf_matrix)
    print("=" * 50)