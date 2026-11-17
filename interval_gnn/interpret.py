"""
interval_gnn/interpret.py
=========================
Interpretability tools that show what the interval adds to a CNN.

Every function works on an IntervalCNN in eval mode and an l_inf input box
[x - eps, x + eps] ∩ [lo, hi] (model space).

What each tool answers
----------------------
logit_intervals / plot_logit_intervals
    Which classes can the network output for *some* input in the box?
    Non-overlapping true-class interval = certified.
certified_radius / plot_certified_radius
    The largest eps the IBP bounds can certify, per image (bisection).
radius_maps / plot_radius_maps
    Where in the image, layer by layer, the uncertainty concentrates.
unstable_fraction
    Fraction of ReLUs whose input interval contains 0 (the source of looseness).
interval_saliency / plot_saliency
    Which input pixels' uncertainty hurts the certificate the most:
    |d(worst-case margin) / d r_0|.
width_uncertainty / plot_width_vs_error
    Is the output interval width a useful uncertainty score?
    (AUROC for detecting misclassified images.)
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .cnn_layers import input_box, interval_relu
from .losses import worst_case_logits


# ---------------------------------------------------------------------------
# Logit intervals and margins
# ---------------------------------------------------------------------------

@torch.no_grad()
def logit_intervals(model, x: Tensor, eps, lo: float = 0.0, hi: float = 1.0):
    """(z_lo, z_hi) for every class; eps may be a float or a per-sample tensor [N]."""
    model.eval()
    if torch.is_tensor(eps):
        eps = eps.view(-1, *([1] * (x.dim() - 1)))
    return model.bounds(x, eps, lo, hi)


def worst_margin(z_lo: Tensor, z_hi: Tensor, y: Tensor) -> Tensor:
    """lower(z_y) - max_{j != y} upper(z_j); > 0 means certified."""
    z = worst_case_logits(z_lo, z_hi, y)
    z_true = z.gather(1, y[:, None]).squeeze(1)
    return z_true - z.scatter(1, y[:, None], float("-inf")).max(dim=1).values


def plot_logit_intervals(z_lo: Tensor, z_hi: Tensor, y: int, classes: Sequence[str],
                         clean: Optional[Tensor] = None, ax=None, title: str = ""):
    """Horizontal bars [z_lo, z_hi] per class for one image; the true class is highlighted."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    z_lo, z_hi = z_lo.cpu(), z_hi.cpu()
    k = len(classes)
    for j in range(k):
        color = "tab:green" if j == y else "tab:gray"
        ax.barh(j, (z_hi[j] - z_lo[j]).item(), left=z_lo[j].item(), color=color, alpha=0.7, height=0.6)
    if clean is not None:
        ax.scatter(clean.cpu(), range(k), color="black", s=12, zorder=3, label="clean logit")
    other_hi = torch.cat([z_hi[:y], z_hi[y + 1:]]).max().item()
    ax.axvline(other_hi, color="tab:red", ls="--", lw=1, label="max upper (other classes)")
    ax.axvline(z_lo[y].item(), color="tab:green", ls=":", lw=1, label="lower (true class)")
    ax.set_yticks(range(k))
    ax.set_yticklabels(classes)
    cert = z_lo[y].item() > other_hi
    ax.set_title(title or ("certified" if cert else "not certified"))
    ax.legend(fontsize=7, loc="lower right")
    return cert


# ---------------------------------------------------------------------------
# Certified radius
# ---------------------------------------------------------------------------

@torch.no_grad()
def certified_radius(model, x: Tensor, y: Tensor, eps_max: float,
                     lo: float = 0.0, hi: float = 1.0, steps: int = 12) -> Tensor:
    """
    Largest eps in [0, eps_max] that IBP certifies, per image, by bisection.
    -1 marks a wrong clean prediction; 0 a correct one that is not certifiable at tiny eps.
    IBP bounds grow with eps (inclusion monotonicity), so bisection is valid.
    """
    a = torch.zeros(len(x), device=x.device)
    b = torch.full((len(x),), float(eps_max), device=x.device)
    ok_max = worst_margin(*logit_intervals(model, x, b, lo, hi), y) > 0
    for _ in range(steps):
        m = (a + b) / 2
        ok = worst_margin(*logit_intervals(model, x, m, lo, hi), y) > 0
        a = torch.where(ok, m, a)
        b = torch.where(ok, b, m)
    correct = model(x).argmax(1).eq(y)
    rad = torch.where(ok_max, torch.full_like(a, float(eps_max)), a)
    return torch.where(correct, rad, torch.full_like(rad, -1.0))


def plot_certified_radius(radii: Tensor, eps_ref: Optional[float] = None, scale: float = 1.0,
                          ax=None, label: str = ""):
    """
    Verified accuracy vs eps: fraction of images with certified radius >= eps.
    Wrong predictions have radius -1, so the curve starts at the clean accuracy for eps = 0.
    """
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    r = radii.cpu() * scale
    grid = torch.linspace(0, max(r.max().item(), 1e-12), 200)
    ax.plot(grid, [(r >= e).float().mean().item() for e in grid], label=label)
    if eps_ref is not None:
        ax.axvline(eps_ref * scale, color="gray", ls="--", lw=1)
    ax.set_xlabel("eps (pixel units)" if scale != 1.0 else "eps")
    ax.set_ylabel("verified accuracy")
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Inside the network: radius maps and unstable ReLUs
# ---------------------------------------------------------------------------

@torch.no_grad()
def radius_maps(model, x: Tensor, eps, lo: float = 0.0, hi: float = 1.0) -> list[Tensor]:
    """Channel-mean radius [N, H_l, W_l] after each interval conv block (index 0 = input box)."""
    model.eval()
    c, r = input_box(x, eps, lo, hi)
    maps = [r.mean(dim=1)]
    for layer in model.interval_layers:
        c, r = layer(c, r)
        maps.append(r.mean(dim=1))
    return maps


@torch.no_grad()
def unstable_fraction(model, x: Tensor, eps, lo: float = 0.0, hi: float = 1.0) -> list[float]:
    """Per conv layer: fraction of pre-ReLU intervals with lower < 0 < upper."""
    model.eval()
    c, r = input_box(x, eps, lo, hi)
    out = []
    for layer in model.interval_layers:
        m, s = layer.pre_activation(c, r)
        out.append(((m - s < 0) & (m + s > 0)).float().mean().item())
        c, r = interval_relu(m, s)
    return out


def plot_radius_maps(x: Tensor, maps: list[Tensor], idx: int = 0, img_scale=(0.5, 0.5)):
    """Input image next to the radius map of every layer, upsampled to the input size."""
    import matplotlib.pyplot as plt
    n = len(maps)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.2))
    img = (x[idx].cpu() * img_scale[1] + img_scale[0]).clamp(0, 1).permute(1, 2, 0).squeeze()
    axes[0].imshow(img, cmap=None if img.dim() == 3 else "gray")
    axes[0].set_title("input")
    H, W = x.shape[-2:]
    for l, (ax, m) in enumerate(zip(axes[1:], maps)):
        up = F.interpolate(m[idx:idx + 1, None], size=(H, W), mode="nearest")[0, 0].cpu()
        ax.imshow(img, cmap=None if img.dim() == 3 else "gray")
        im = ax.imshow(up, cmap="inferno", alpha=0.6)
        ax.set_title("input box" if l == 0 else f"layer {l}: mean $r$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interval saliency
# ---------------------------------------------------------------------------

def interval_saliency(model, x: Tensor, y: Tensor, eps, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    """
    |d(worst-case margin) / d r_0| summed over channels, shape [N, H, W].

    For a fixed ReLU pattern the output radius is linear in the input radius r_0,
    so this is how much one more unit of uncertainty at each pixel costs the
    certificate. Compare with the ordinary gradient saliency of the clean logit.
    """
    model.eval()
    c0, r0 = input_box(x, eps, lo, hi)
    r0 = r0.detach().requires_grad_(True)
    c, r = model.forward_interval(c0.detach(), r0)
    margin = worst_margin(c - r, c + r, y).sum()
    (g,) = torch.autograd.grad(margin, r0)
    return g.abs().sum(dim=1)


def gradient_saliency(model, x: Tensor, y: Tensor) -> Tensor:
    """Ordinary |d z_y / d x| summed over channels (point-network reference)."""
    model.eval()
    x = x.detach().requires_grad_(True)
    z = model(x).gather(1, y[:, None]).sum()
    (g,) = torch.autograd.grad(z, x)
    return g.abs().sum(dim=1)


def plot_saliency(x: Tensor, maps: dict[str, Tensor], idx: int = 0, img_scale=(0.5, 0.5)):
    """Input image next to several saliency maps for the same image."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.2 * (len(maps) + 1), 3.2))
    img = (x[idx].detach().cpu() * img_scale[1] + img_scale[0]).clamp(0, 1).permute(1, 2, 0).squeeze()
    axes[0].imshow(img, cmap=None if img.dim() == 3 else "gray")
    axes[0].set_title("input")
    for ax, (name, m) in zip(axes[1:], maps.items()):
        im = ax.imshow(m[idx].detach().cpu(), cmap="magma")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interval width as an uncertainty score
# ---------------------------------------------------------------------------

def auroc(scores: Tensor, labels: Tensor) -> float:
    """AUROC of `scores` for the positive class `labels == 1` (Mann-Whitney, ties averaged)."""
    scores, labels = scores.double().cpu(), labels.bool().cpu()
    n_pos, n_neg = labels.sum().item(), (~labels).sum().item()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    for v in scores.unique():  # average ranks of ties
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


@torch.no_grad()
def width_uncertainty(model, loader, eps, lo: float = 0.0, hi: float = 1.0, device=None) -> dict:
    """
    Output width of the predicted class and the clean softmax confidence, per test image,
    plus the AUROC of each score for detecting misclassifications.
    """
    model.eval()
    widths, conf, wrong = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        pred = z.argmax(1)
        z_lo, z_hi = model.bounds(x, eps, lo, hi)
        widths.append((z_hi - z_lo).gather(1, pred[:, None]).squeeze(1).cpu())
        conf.append(F.softmax(z, 1).max(1).values.cpu())
        wrong.append(pred.ne(y).cpu())
    widths, conf, wrong = torch.cat(widths), torch.cat(conf), torch.cat(wrong)
    return {
        "width": widths, "confidence": conf, "wrong": wrong,
        "auroc_width": auroc(widths, wrong),
        "auroc_1-confidence": auroc(1 - conf, wrong),
    }


def plot_width_vs_error(res: dict, n_bins: int = 10, ax=None):
    """Error rate per quantile bin of output width (should rise if width signals uncertainty)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    w, wrong = res["width"], res["wrong"].float()
    edges = torch.quantile(w, torch.linspace(0, 1, n_bins + 1))
    mids, err = [], []
    for i in range(n_bins):
        m = (w >= edges[i]) & ((w <= edges[i + 1]) if i == n_bins - 1 else (w < edges[i + 1]))
        if m.any():
            mids.append(w[m].median().item())
            err.append(wrong[m].mean().item())
    ax.plot(mids, err, "o-")
    ax.set_xlabel("output interval width (predicted class)")
    ax.set_ylabel("error rate")
    ax.set_title(f"AUROC(width) = {res['auroc_width']:.3f},  AUROC(1-conf) = {res['auroc_1-confidence']:.3f}")
    ax.grid(alpha=0.3)
    return ax
