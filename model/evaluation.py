"""Shared evaluation helpers used by both train_models.py and the Streamlit app.

All six metrics required by the assignment are computed here so that the
numbers in the README and the numbers in the live app come from one code path.

The dataset is multi-class (10 digits), therefore:
  * Precision / Recall / F1 are MACRO averaged (every digit counts equally).
  * AUC is One-vs-Rest, MACRO averaged, computed from predict_proba.
  * MCC is the multi-class generalisation implemented in scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def score_predictions(y_true, y_pred, y_proba=None, class_labels=None):
    """Return a dict with Accuracy, AUC, Precision, Recall, F1 and MCC."""
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            scores["AUC"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=class_labels
            )
        except ValueError:
            # happens when an uploaded file does not contain every class
            scores["AUC"] = np.nan
    return scores


def confusion_frame(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    names = [str(c) for c in class_labels]
    return pd.DataFrame(cm, index=names, columns=names)


def report_frame(y_true, y_pred, class_labels):
    rep = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    return pd.DataFrame(rep).T.round(4)
