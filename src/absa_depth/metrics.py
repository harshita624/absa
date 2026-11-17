"""Regression metrics on the original 1–9 target scale."""
from __future__ import annotations
import numpy as np


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Calculate RMSEs and Pearson correlations, including joint VA RMSE."""
    p, y = np.asarray(predictions, dtype=float), np.asarray(targets, dtype=float)
    err = p-y
    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        if len(a)<2 or np.std(a)==0 or np.std(b)==0: return float("nan")
        return float(np.corrcoef(a,b)[0,1])
    return {"RMSE_V": float(np.sqrt(np.mean(err[:,0]**2))), "RMSE_A": float(np.sqrt(np.mean(err[:,1]**2))),
            "RMSE_VA": float(np.sqrt(np.mean(err[:,0]**2+err[:,1]**2))),
            "Pearson_V": pearson(p[:,0],y[:,0]), "Pearson_A": pearson(p[:,1],y[:,1])}


def unnormalize(values): return values * 4 + 5
