"""
Model Evaluation and Metrics Module.
Calculates classification and regression metrics with comprehensive edge-case handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes complete classification evaluation metrics with edge-case protection.
    
    Handles:
    - Single class present in test set
    - Zero division in precision / recall
    - Probability estimates missing for ROC-AUC
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC & Curve
    roc_auc = None
    fpr, tpr, thresholds = None, None, None
    if y_prob is not None:
        try:
            # Handle probability matrix (2 cols) or 1D array
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                prob_positive = y_prob[:, 1]
            else:
                prob_positive = y_prob.ravel()

            # Only calculate ROC-AUC if both classes exist in y_true
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, prob_positive))
                fpr, tpr, thresholds = roc_curve(y_true, prob_positive)
        except Exception:
            roc_auc = None

    # Confusion matrix
    # Ensure 2x2 shape even if a class was not predicted
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr) if fpr is not None else None
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, MSE, RMSE, and R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Handle edge case where all values are identical or NaN
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0}

    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
    mse = float(mean_squared_error(y_true_clean, y_pred_clean))
    rmse = float(np.sqrt(mse))

    # R2 can be negative if model performs worse than mean
    try:
        r2 = float(r2_score(y_true_clean, y_pred_clean))
    except Exception:
        r2 = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


def metrics_to_dataframe(metrics_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts nested metrics dictionary into a clean pandas DataFrame for display and export.
    """
    rows = []
    for model_name, m in metrics_dict.items():
        row = {"Model": model_name}
        for k, v in m.items():
            if k in ["confusion_matrix", "roc_curve"]:
                continue
            if v is not None:
                row[k.replace("_", " ").title()] = round(v, 4)
            else:
                row[k.replace("_", " ").title()] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)
