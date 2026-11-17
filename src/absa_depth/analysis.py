"""Subgroup definitions and summary utilities for RQ3."""
from __future__ import annotations
import re
import numpy as np

NEGATION_WORDS={"not","no","never","neither","nor","without","hardly","n't"}
CONTRAST_WORDS={"but","however","although","though","yet","while"}

def subgroup_masks(texts: list[str], aspects_per_sentence: list[int], targets: np.ndarray) -> dict[str,np.ndarray]:
    """Return aspect-level masks for documented text and target subgroups."""
    neg=[]; contrast=[]; multi=[]; extreme=[]
    for i,text in enumerate(texts):
        words=set(re.findall(r"\b[\w']+\b",text.casefold()))
        neg.append(bool(words & NEGATION_WORDS)); contrast.append(bool(words & CONTRAST_WORDS))
        multi.append(aspects_per_sentence[i]>=2)
    # Targets and predictions use flattened aspect order, so repeat sentence labels.
    aspect_neg=np.repeat(neg,aspects_per_sentence); aspect_contrast=np.repeat(contrast,aspects_per_sentence)
    aspect_multi=np.repeat(multi,aspects_per_sentence); aspect_extreme=np.abs(np.asarray(targets)[:,0]-5)>=3
    return {"negation":aspect_neg,"contrast":aspect_contrast,"multi_aspect":aspect_multi,"extreme_valence":aspect_extreme}

def subgroup_summary(mask, targets, last_predictions, exit_predictions, depths):
    """Return count, average depth and last-layer/exit RMSE_VA for a mask."""
    from .metrics import regression_metrics
    mask=np.asarray(mask,dtype=bool)
    if not mask.any(): return {"count":0,"mean_exit_layer":float("nan"),"last_RMSE_VA":float("nan"),"exit_RMSE_VA":float("nan")}
    return {"count":int(mask.sum()),"mean_exit_layer":float(np.mean(np.asarray(depths)[mask])),
            "last_RMSE_VA":regression_metrics(np.asarray(last_predictions)[mask],np.asarray(targets)[mask])["RMSE_VA"],
            "exit_RMSE_VA":regression_metrics(np.asarray(exit_predictions)[mask],np.asarray(targets)[mask])["RMSE_VA"]}
