"""
interval_gnn/losses.py
======================
Loss functions for training CR-INN models.

composite_loss
--------------
Implements Equation (d) from the report:

    L_total = L_task
            + lambda1 * sum_l max(0, ||W_r^(l)||_inf - 1)   [Lipschitz penalty]
            + lambda2 * mean_l E[r^(l)]                      [mean-radius penalty]

The two regularisers attack interval explosion at different levels:
  - lambda1 penalises weight matrices whose inf-norm exceeds 1, which is the
    necessary and sufficient condition for guaranteed per-layer non-expansion
    (Corollary 3.5 of the report).
  - lambda2 directly penalises large radii in the hidden representations,
    providing a complementary signal that is agnostic to how W_bar is set.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .layers import IntervalConvLayer
from .cnn_layers import IntervalConv2d, kappa


# ---------------------------------------------------------------------------
# Task losses (convenience wrappers)
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Standard cross-entropy for classification."""
    return F.cross_entropy(logits, labels)


def mse_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """MSE for regression."""
    return F.mse_loss(preds, targets)


# ---------------------------------------------------------------------------
# Regularisers
# ---------------------------------------------------------------------------

def lipschitz_penalty(model: torch.nn.Module, threshold: float = 1.0) -> Tensor:
    """
    sum_ell  max(0, ||W_r^(ell)||_inf - threshold)

    Iterates over all IntervalConvLayer modules in the model and adds a
    hinge penalty for each whose radius-path weight norm exceeds `threshold`.

    Parameters
    ----------
    model     : any nn.Module containing IntervalConvLayer sub-modules
    threshold : the Lipschitz target (default 1.0 → non-expansion guarantee)
    """
    device = next(model.parameters()).device
    penalty = torch.zeros(1, device=device)
    for module in model.modules():
        if isinstance(module, IntervalConvLayer):
            norm = module.radius_weight_inf_norm()
            penalty = penalty + torch.clamp(norm - threshold, min=0.0)
        elif isinstance(module, IntervalConv2d):
            # the CNN radius goes through |K_c|, so penalise kappa(K_c)
            penalty = penalty + torch.clamp(kappa(module.weight) - threshold, min=0.0)
    return penalty


def mean_radius_penalty(hidden_radii: list[Tensor]) -> Tensor:
    """
    mean_ell  mean(r^(ell))

    Parameters
    ----------
    hidden_radii : list of radius tensors recorded during the forward pass
                   (populated by RadiusTracker or passed manually).
                   Each tensor has shape [N, H] or [batch, H].
    """
    if len(hidden_radii) == 0:
        return torch.zeros(1)
    stacked = torch.stack([r.mean() for r in hidden_radii])
    return stacked.mean()


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

def composite_loss(
    task_loss: Tensor,
    model: torch.nn.Module,
    hidden_radii: Optional[list[Tensor]] = None,
    lambda1: float = 1e-3,
    lambda2: float = 1e-4,
    lipschitz_threshold: float = 1.0,
) -> Tensor:
    """
    Full composite training objective (Eq. (d) in the report).

    Parameters
    ----------
    task_loss     : scalar tensor from cross_entropy_loss or mse_loss
    model         : the IntervalGCN / IntervalGIN / IntervalCNN instance
    hidden_radii  : list of radius tensors from RadiusTracker.history (optional)
    lambda1       : weight for the Lipschitz penalty
    lambda2       : weight for the mean-radius penalty
    lipschitz_threshold : norm ceiling (1.0 → non-expansion per Corollary 3.5)

    Returns
    -------
    total_loss : scalar Tensor (differentiable, backprop-safe)
    """
    total = task_loss

    # Lipschitz penalty on W_r norms
    lip = lipschitz_penalty(model, threshold=lipschitz_threshold)
    total = total + lambda1 * lip

    # Mean-radius penalty on hidden representations
    if hidden_radii is not None and len(hidden_radii) > 0:
        rad = mean_radius_penalty(hidden_radii)
        total = total + lambda2 * rad

    return total


# ---------------------------------------------------------------------------
# IBP loss (Gowal et al. 2018)
# ---------------------------------------------------------------------------

def worst_case_logits(lo: Tensor, hi: Tensor, y: Tensor) -> Tensor:
    """z_hat: lower bound for the true class, upper bound for every other class."""
    z = hi.clone()
    z.scatter_(1, y[:, None], lo.gather(1, y[:, None]))
    return z


def ibp_loss(clean_logits: Tensor, lo: Tensor, hi: Tensor, y: Tensor, kappa: float) -> Tensor:
    """L = kappa * CE(z(x), y) + (1 - kappa) * CE(z_hat, y)."""
    return (kappa * F.cross_entropy(clean_logits, y)
            + (1 - kappa) * F.cross_entropy(worst_case_logits(lo, hi, y), y))


def certified(lo: Tensor, hi: Tensor, y: Tensor) -> Tensor:
    """True where every wrong-class upper bound is below the true-class lower bound."""
    z = worst_case_logits(lo, hi, y)
    z_true = z.gather(1, y[:, None]).squeeze(1)
    others = z.scatter(1, y[:, None], float("-inf"))
    return others.max(dim=1).values < z_true


def margin_certified(lb: Tensor, y: Tensor) -> Tensor:
    """True where every margin lower bound f_y - f_j (j != y) is positive."""
    return lb.scatter(1, y[:, None], float("inf")).min(dim=1).values > 0


def smoothed_eps(step: int, warmup: int, ramp: int, eps_max: float,
                 mid: float = 0.25, beta: float = 4.0) -> float:
    """
    SmoothedScheduler of Gowal et al. (2018), as used by Shi et al. (2021): 0 for `warmup` steps,
    then proportional to t^beta for the first `mid` of the ramp, then linear up to eps_max.
    """
    if step <= warmup:
        return 0.0
    final = warmup + ramp
    if step >= final:
        return eps_max
    mid_step = int(ramp * mid) + warmup
    t = float(mid_step - warmup) ** (beta - 1.0)
    alpha = eps_max / ((final - mid_step) * beta * t + (mid_step - warmup) * t)
    mid_value = alpha * float(mid_step - warmup) ** beta
    if step >= mid_step:
        return alpha * beta * t * (step - mid_step) + mid_value
    return alpha * float(step - warmup) ** beta


def eps_kappa_schedule(step: int, warmup: int, ramp: int, eps_max: float,
                       kappa_end: float = 0.5) -> tuple[float, float]:
    """Linear ramp of eps from 0 and of kappa from 1 to kappa_end after `warmup` steps."""
    t = min(max(step - warmup, 0) / max(ramp, 1), 1.0)
    return t * eps_max, 1.0 - t * (1.0 - kappa_end)
