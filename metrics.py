"""
metrics.py
----------
Loss function and evaluation metrics.

    DiceLoss        : soft multi-class Dice loss for training
    evaluate_model  : Dice and Jaccard for WT / TC / ET regions

BraTS evaluation regions (post label remapping):
    Whole Tumour (WT) = {1, 2, 3}
    Tumour Core  (TC) = {1, 3}
    Enhancing Tumour (ET) = {3}
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.
    Background (class 0) excluded from loss by default.
    """

    def __init__(self, num_classes: int = 4,
                 smooth: float = 1e-5,
                 ignore_bg: bool = True):
        super().__init__()
        self.C         = num_classes
        self.smooth    = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        probs   = F.softmax(logits, dim=1)
        oh      = F.one_hot(targets, self.C).permute(0,4,1,2,3).float()
        start   = 1 if self.ignore_bg else 0
        d       = 0.0
        for c in range(start, self.C):
            p = probs[:, c]; g = oh[:, c]
            d += (2*(p*g).sum() + self.smooth) / \
                 (p.sum() + g.sum() + self.smooth)
        return 1.0 - d / (self.C - start)


def _dice(pred: np.ndarray, gt: np.ndarray,
          smooth: float = 1e-5) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return float((2*tp + smooth) / (2*tp + fp + fn + smooth))


def _jaccard(pred: np.ndarray, gt: np.ndarray,
             smooth: float = 1e-5) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return float((tp + smooth) / (tp + fp + fn + smooth))


def build_brats_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "WT": (seg >= 1),
        "TC": ((seg == 1) | (seg == 3)),
        "ET": (seg == 3),
    }


def evaluate_model(model: nn.Module,
                   dataloader,
                   device: str = "cpu") -> Dict[str, float]:
    model.eval()
    rd = {"WT": [], "TC": [], "ET": []}
    rj = {"WT": [], "TC": [], "ET": []}
    with torch.no_grad():
        for imgs, lbls in dataloader:
            preds = torch.argmax(
                model(imgs.to(device)), dim=1).cpu().numpy()
            lbls  = lbls.numpy()
            for b in range(preds.shape[0]):
                pr = build_brats_regions(preds[b])
                gt = build_brats_regions(lbls[b])
                for r in ("WT", "TC", "ET"):
                    rd[r].append(_dice(pr[r],    gt[r]))
                    rj[r].append(_jaccard(pr[r], gt[r]))
    res = {}
    for r in ("WT", "TC", "ET"):
        res[f"dice_{r}"]    = float(np.mean(rd[r]))
        res[f"jaccard_{r}"] = float(np.mean(rj[r]))
    res["mean_dice"] = float(np.mean(
        [res[f"dice_{r}"] for r in ("WT","TC","ET")]))
    res["mean_jaccard"] = float(np.mean(
        [res[f"jaccard_{r}"] for r in ("WT","TC","ET")]))
    return res
