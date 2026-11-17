"""
Warm-up regularizers of Shi et al. (2021), ported from their regularization.py.

`pre` is the list recorded by PooledCNN7: the input box first, then (lower, upper) of the
input of every ReLU. With tolerance tau,

    L_tightness = 1/(tau m) sum_i ReLU(tau - E(Delta_0) / E(Delta_i))             (Eq. 7)
    L_relu      = 1/(tau m) sum_i [ReLU(tau - min(a_i, 1/a_i)) + ReLU(tau - min(b_i, 1/b_i))]   (Eq. 8)

where Delta_i is the radius at layer i, a_i is the ratio of the contributions of active and
inactive neurons to the layer mean and b_i the same ratio for the variance.
As in their code, a layer whose ratio is not finite contributes nothing to L_relu.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def warmup_regularizer(pre: List[Tuple[Tensor, Tensor]], tol: float = 0.5) -> Tensor:
    """L_tightness + L_relu for the recorded bounds."""
    lo0, hi0 = pre[0]
    tight0 = ((hi0 - lo0) / 2).mean()
    tight = relu = torch.zeros((), device=lo0.device)
    for lo, hi in pre[1:]:
        center, diff = (hi + lo) / 2, (hi - lo) / 2
        tight = tight + F.relu(tol - tight0 / diff.mean().clamp(min=1e-12)) / tol
        active, inactive = lo > 0, hi < 0
        delta = (center - center.mean()) ** 2
        mean_ratio = (center * active).mean() / -(center * inactive).mean()
        var_ratio = (delta * active).sum() / (delta * inactive).sum()
        mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
        var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
        term = (F.relu(tol - mean_ratio) + F.relu(tol - var_ratio)) / tol
        if torch.isfinite(term):
            relu = relu + term
    m = len(pre) - 1
    return (tight + relu) / m
