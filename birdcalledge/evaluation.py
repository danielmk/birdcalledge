# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:22:50 2026

@author: Daniel
"""

import numpy as np

CONFUSION_KEYS = [
    "tpr", "fnr",
    "tnr", "fpr",
    "precision", "fdr",
    "accuracy", "balanced_accuracy",
    "TP", "TN", "FP", "FN",
]

def confusion_rates(y_true, y_pred):
    """
    Compute all relevant binary classification rates
    for 0 = negative (noise), 1 = positive (signal).
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Confusion matrix counts
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Rates (safe division everywhere)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # recall / sensitivity
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0   # miss rate

    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # specificity
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0   # false alarm rate

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # PPV
    fdr = FP / (FP + TP) if (FP + TP) > 0 else 0.0        # false discovery rate

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        # core rates
        "tpr": tpr,
        "fnr": fnr,
        "tnr": tnr,
        "fpr": fpr,

        # decision quality
        "precision": precision,
        "fdr": fdr,

        # global
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,

        # optional counts (often useful for debugging)
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }

