"""
interval_gnn/utils.py
=====================
Utilities for radius tracking, theoretical bound verification, and plotting.

Classes
-------
RadiusTracker
    Collects ‖r^(ℓ)‖_∞ at each layer during a forward pass.

Functions
---------
compute_rho
    Computes the theoretical contraction factor ρ = γ_agg * W_bar.
verify_contraction
    Checks whether the empirical radius evolution respects the bound
    ‖r^(L)‖_∞ ≤ ρ^L * ‖r^(0)‖_∞ (Theorem 3.4 of the report).
plot_radius_evol
    Plots the layer-by-layer ‖r^(ℓ)‖_∞ curve with the theoretical bound.
set_seed
    Sets all relevant random seeds for reproducibility.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Radius Tracker
# ---------------------------------------------------------------------------

class RadiusTracker:
    """
    Lightweight object that accumulates ‖r^(ℓ)‖_∞ values during a forward pass.

    Usage
    -----
    tracker = RadiusTracker()
    model.radius_tracker = tracker   # models support this attribute

    # inside forward loop:
    tracker.record(r, layer=0)

    # after forward:
    history = tracker.history  # list of (layer_idx, inf_norm) tuples
    """

    def __init__(self) -> None:
        self.history: list[tuple[int, float]] = []
        self._raw: list[Tensor] = []   # stores actual radius tensors

    def reset(self) -> None:
        self.history.clear()
        self._raw.clear()

    def record(self, r: Tensor, layer: int) -> None:
        """Record ‖r‖_∞ for a given layer."""
        with torch.no_grad():
            inf_norm = r.abs().max().item()
        self.history.append((layer, inf_norm))
        self._raw.append(r.detach())

    @property
    def norms(self) -> list[float]:
        """List of ‖r^(ℓ)‖_∞ values in layer order."""
        return [v for _, v in sorted(self.history)]

    @property
    def raw_tensors(self) -> list[Tensor]:
        """Actual radius tensors (detached) in order."""
        return self._raw


# ---------------------------------------------------------------------------
# Theoretical bound utilities
# ---------------------------------------------------------------------------

def compute_rho(W_bar: float, agg: str, d_max: Optional[int] = None) -> float:
    """
    Compute the theoretical contraction factor ρ from Theorem 3.4.

        ρ = γ_agg * W_bar

    where:
        γ_agg = 1             for agg='mean'  (degree-independent)
        γ_agg = d_max + 1     for agg='sum'   (grows with max degree)

    Parameters
    ----------
    W_bar : float   Infinity-norm bound on W_r (Assumption 3.2).
    agg   : str     'mean' or 'sum'.
    d_max : int     Maximum node degree (required for agg='sum').

    Returns
    -------
    rho : float
    """
    if agg == "mean":
        gamma = 1.0
    elif agg == "sum":
        if d_max is None:
            raise ValueError("d_max must be provided for agg='sum'")
        gamma = float(d_max + 1)
    else:
        raise ValueError(f"Unknown agg mode: {agg!r}")
    return gamma * W_bar


def verify_contraction(
    norms: list[float],
    rho: float,
    r_max: float = float("inf"),
    rtol: float = 1e-3,
) -> dict:
    """
    Check whether the empirical ‖r^(ℓ)‖_∞ sequence respects the bound
    from Theorem 3.4 and Proposition 3.3.

    Returns a dict with:
        'holds'          : bool  – True if bound holds at every layer
        'violations'     : list  – (layer, empirical, theoretical) for failures
        'saturation_ok'  : bool  – True if all norms ≤ r_max (Prop 3.3)
        'pre_sat_layers' : int   – number of layers in the pre-saturation regime
    """
    r0 = norms[0] if len(norms) > 0 else 0.0
    violations = []
    pre_sat = 0

    for ell, emp in enumerate(norms):
        theoretical = min(rho ** ell * r0, r_max)
        if emp > theoretical * (1 + rtol):
            violations.append((ell, emp, theoretical))
        if rho ** ell * r0 < r_max:
            pre_sat += 1

    return {
        "holds": len(violations) == 0,
        "violations": violations,
        "saturation_ok": all(v <= r_max + rtol for v in norms),
        "pre_sat_layers": pre_sat,
        "rho": rho,
        "r0": r0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_radius_evol(
    norms_dict: dict[str, list[float]],
    rho: Optional[float] = None,
    r0: Optional[float] = None,
    r_max: Optional[float] = None,
    title: str = "Radius Propagation: Empirical vs Theoretical Bound",
    figsize: tuple = (9, 5),
) -> None:
    """
    Plot ‖r^(ℓ)‖_∞ across layers for one or more model variants, and
    overlay the theoretical contraction bound ρ^ℓ * r^(0).

    Parameters
    ----------
    norms_dict : dict mapping label → list of per-layer inf-norms
    rho        : theoretical contraction factor (optional overlay)
    r0         : initial radius norm (optional; defaults to first entry)
    r_max      : saturation ceiling (draws a horizontal dashed line)
    title      : plot title
    figsize    : matplotlib figure size
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10.colors
    for idx, (label, norms) in enumerate(norms_dict.items()):
        layers = list(range(len(norms)))
        ax.plot(layers, norms, marker="o", linewidth=2,
                color=colors[idx % 10], label=label)

    # Theoretical bound overlay
    if rho is not None and len(norms_dict) > 0:
        first_norms = next(iter(norms_dict.values()))
        r0_val = r0 if r0 is not None else (first_norms[0] if first_norms else 1.0)
        L = max(len(v) for v in norms_dict.values())
        layers_th = list(range(L))
        bound = [min(rho ** ell * r0_val, r_max or float("inf")) for ell in layers_th]
        ax.plot(layers_th, bound, linestyle="--", linewidth=1.8,
                color="crimson", label=rf"Theorem bound  $\rho^{{\ell}}\|r^{{(0)}}\|_\infty$  ($\rho={rho:.3f}$)")

    # r_max saturation line
    if r_max is not None:
        ax.axhline(r_max, linestyle=":", linewidth=1.5, color="grey",
                   label=f"$r_{{\\max}} = {r_max}$")

    ax.set_xlabel("Layer ℓ", fontsize=13)
    ax.set_ylabel(r"$\|r^{(\ell)}\|_\infty$", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


def plot_train_curves(
    history: dict[str, list],
    title: str = "Training Curves",
    figsize: tuple = (12, 4),
) -> None:
    """
    Plot training/validation accuracy and loss over epochs.

    Parameters
    ----------
    history : dict with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val", linewidth=2, linestyle="--")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.35)

    # Accuracy
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train", linewidth=2)
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.35)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show()


def summarize_results(results: dict[str, dict]) -> None:
    """Pretty-print a results summary table."""
    try:
        import pandas as pd
        df = pd.DataFrame(results).T
        print(df.to_string())
    except ImportError:
        for name, metrics in results.items():
            print(f"{name:30s}: {metrics}")


# ---------------------------------------------------------------------------
# Width bound for interval CNNs (replaces verify_contraction for the CNN)
# ---------------------------------------------------------------------------

def verify_width_bound(model, r0: float, norms: list[float], rtol: float = 1e-4) -> dict:
    """
    Check ||r_ell||_inf <= (prod_{j<=ell} g_j) * ||r_0||_inf (width-growth theorem), where
    g_j = kappa(K_j) for a plain block and kappa(K_j) * max|gamma/sigma| with BatchNorm,
    computed from the *trained* weights (BatchNorm in eval mode). ReLU never enlarges a
    radius, so the bound holds layer by layer; `norms` are the recorded post-ReLU radii.
    """
    ks = [layer.gain() for layer in model.interval_layers]
    bound, bounds, violations = r0, [], []
    for ell, emp in enumerate(norms):
        bound *= ks[ell]
        bounds.append(bound)
        if emp > bound * (1 + rtol):
            violations.append((ell, emp, bound))
    return {"holds": not violations, "violations": violations, "kappas": ks, "bounds": bounds, "r0": r0}


def verify_graph_width_bound(model, edge_index, num_nodes: int, r0: float, norms: list[float],
                             rtol: float = 1e-4) -> dict:
    """
    Graph version of the width-growth bound for IntervalGCN:
        ||R_l||_max <= prod_{j<=l} ||Â||_inf * ||W_j||_inf * ||R_0||_max,
    with ||Â||_inf the max row sum of the propagation matrix and ||W_j||_inf the
    max row sum of |W_j| (trained weights). ReLU never enlarges a radius.
    """
    from .layers import propagation_inf_norm
    gammas, ws, bounds, violations = [], [], [], []
    bound = r0
    for ell, (conv, emp) in enumerate(zip(model.convs, norms)):
        g = propagation_inf_norm(edge_index, num_nodes, conv.agg_mode)
        w = conv.radius_weight_inf_norm().item()
        bound *= g * w
        gammas.append(g)
        ws.append(w)
        bounds.append(bound)
        if emp > bound * (1 + rtol):
            violations.append((ell, emp, bound))
    return {"holds": not violations, "violations": violations, "gamma_A": gammas,
            "W_inf": ws, "bounds": bounds, "r0": r0}
