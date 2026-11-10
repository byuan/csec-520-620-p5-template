"""Metrics used throughout Project 5.

PROVIDED: you do not need to change this file. Use these helpers so that every number in
results/metrics.json is computed the same way.
"""
from __future__ import annotations

import numpy as np

from .data import ATTACK_CATEGORIES


def binary_metrics(y_true, y_pred) -> dict:
    """precision, recall (= detection rate), f1 and fpr for labels 0 = normal, 1 = attack."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fpr)}


def per_category_detection(y_pred, categories) -> dict:
    """Fraction of rows flagged as attacks, for each of the nine attack categories."""
    y_pred = np.asarray(y_pred).astype(int)
    categories = np.asarray(categories)
    out = {}
    for c in ATTACK_CATEGORIES:
        mask = categories == c
        if mask.any():
            out[c] = float(y_pred[mask].mean())
    return out
