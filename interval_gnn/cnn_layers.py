"""
interval_gnn/cnn_layers.py
==========================
Sound center-radius interval layers for CNNs (IBP, Gowal et al. 2018; Shi et al. 2021).

An interval tensor <c, r> means the box [c - r, c + r] with r >= 0.
Every layer maps a box to a box that encloses every possible output
(soundness theorem of the report). The radius is propagated through |K_c|,
the *same* kernel as the center, so it receives gradients from the IBP loss.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Interval = Tuple[Tensor, Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_cr(lo: Tensor, hi: Tensor) -> Interval:
    """[lo, hi] -> <c, r>."""
    return (hi + lo) / 2, (hi - lo) / 2


def to_bounds(c: Tensor, r: Tensor) -> Interval:
    """<c, r> -> [lo, hi]."""
    return c - r, c + r


def kappa(W: Tensor) -> Tensor:
    """kappa(K) = max_o sum_{c,a,b} |K[o,c,a,b]| = ||T_K||_inf (convolution-matrix theorem)."""
    return W.abs().flatten(1).sum(dim=1).max()


def input_box(x: Tensor, eps: float, lo: float = 0.0, hi: float = 1.0) -> Interval:
    """Clipped l_inf ball [x - eps, x + eps] ∩ [lo, hi] in center-radius form."""
    return to_cr((x - eps).clamp(lo, hi), (x + eps).clamp(lo, hi))


# ---------------------------------------------------------------------------
# Linear interval layers
# ---------------------------------------------------------------------------

class IntervalConv2d(nn.Module):
    """
    <K_c, K_r> * <c, r>:
        m = K_c * c + b_c
        s = |K_c| * r + K_r * (|c| + r) + b_r

    With interval_weights=False (default) K_r = 0 and this is exactly IBP;
    for a point kernel the result is the exact hull of the range.
    K_r = softplus(rho) >= 0 when interval_weights=True (INN alternative).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        interval_weights: bool = False,
        rho_init: float = -6.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.interval_weights = interval_weights
        if interval_weights:
            self.rho = nn.Parameter(torch.full_like(self.conv.weight, rho_init))
            self.rho_b = nn.Parameter(torch.full((out_channels,), rho_init))

    @property
    def weight(self) -> Tensor:
        return self.conv.weight

    def _conv(self, x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
        c = self.conv
        return F.conv2d(x, w, b, c.stride, c.padding, c.dilation, c.groups)

    def forward(self, c: Tensor, r: Optional[Tensor] = None) -> Interval:
        m = self.conv(c)
        if r is None:
            r = torch.zeros_like(c)
        s = self._conv(r, self.conv.weight.abs())
        if self.interval_weights:
            s = s + self._conv(c.abs() + r, F.softplus(self.rho), F.softplus(self.rho_b))
        return m, s


class IntervalLinear(nn.Module):
    """Same rule for a fully connected layer: m = W c + b, s = |W| r."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self) -> Tensor:
        return self.lin.weight

    def forward(self, c: Tensor, r: Tensor) -> Interval:
        return self.lin(c), F.linear(r, self.lin.weight.abs())


# ---------------------------------------------------------------------------
# Monotone / nonnegative-linear interval maps
# ---------------------------------------------------------------------------

def interval_relu(c: Tensor, r: Tensor) -> Interval:
    """Increasing activation applied to both endpoints."""
    return to_cr(F.relu(c - r), F.relu(c + r))


def interval_maxpool(c: Tensor, r: Tensor, k: int = 2) -> Interval:
    """Lattice max-pool: exact range of max over the box."""
    return to_cr(F.max_pool2d(c - r, k), F.max_pool2d(c + r, k))


def interval_avgpool(c: Tensor, r: Tensor, k: int = 2) -> Interval:
    """Average pooling: linear with nonnegative weights."""
    return F.avg_pool2d(c, k), F.avg_pool2d(r, k)


def interval_global_avgpool(c: Tensor, r: Tensor) -> Interval:
    """Global average pooling [N, C, H, W] -> [N, C]."""
    return c.mean(dim=(2, 3)), r.mean(dim=(2, 3))


def interval_dropout(c: Tensor, r: Tensor, p: float, training: bool) -> Interval:
    """Dropout with one shared mask: a nonnegative diagonal map, so it is sound."""
    if not training or p == 0.0:
        return c, r
    mask = F.dropout(torch.ones_like(c), p=p, training=True)
    return c * mask, r * mask


def interval_bn(bn: nn.modules.batchnorm._BatchNorm, c: Tensor, r: Optional[Tensor]) -> Interval:
    """
    BatchNorm (1d or 2d) as a channel-wise affine map; running statistics in eval mode.

    Training mode follows Shi et al. (2021): the statistics come from the *clean* batch.
    A clean pass (r == 0) computes them, updates the running statistics once and caches
    them on the module; the following bounds pass on the same batch (r > 0) reuses the
    cached statistics and does not touch the running ones. Using the box centers instead
    would normalize the bounds with different statistics than the clean prediction and
    pollute the running statistics, so eval-mode accuracy would collapse.
    The affine map is fixed for the batch either way, so the bounds stay sound.
    Call the clean forward before bounds() on each training batch.
    r = None is a clean pass without a radius tensor; the returned radius is then None.
    """
    dims = [0] + list(range(2, c.dim()))
    if bn.training:
        cached = getattr(bn, "_ibp_stats", None)
        if r is None or bool((r == 0).all()) or cached is None or cached[2] != c.shape:
            mean = c.mean(dim=dims)
            var = c.var(dim=dims, unbiased=False)
            with torch.no_grad():
                n = c.numel() / c.size(1)
                bn.running_mean.mul_(1 - bn.momentum).add_(bn.momentum * mean)
                bn.running_var.mul_(1 - bn.momentum).add_(bn.momentum * var * n / max(n - 1, 1))
            bn._ibp_stats = (mean, var, c.shape)
        else:
            mean, var = cached[0], cached[1]
    else:
        mean, var = bn.running_mean, bn.running_var
    g = bn.weight / torch.sqrt(var + bn.eps)
    b = bn.bias - g * mean
    shape = [1, -1] + [1] * (c.dim() - 2)
    g, b = g.view(shape), b.view(shape)
    return g * c + b, None if r is None else g.abs() * r


# ---------------------------------------------------------------------------
# Backward-compatible block used by the notebooks
# ---------------------------------------------------------------------------

class IntervalConv2dLayer(nn.Module):
    """
    One interval block: IntervalConv2d -> [interval BatchNorm] -> interval ReLU.

    Replaces the old layer whose separate W_r path, tanh squash and ReLU on the
    radius broke enclosure and left r = 0 forever. r_max, W_bar and hard_clip
    are accepted for compatibility and ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        r_max: float = 1.0,
        W_bar: float = 0.9,
        hard_clip: bool = False,
        bias: bool = True,
        interval_weights: bool = False,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = IntervalConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, bias=bias, interval_weights=interval_weights)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None

    @property
    def W_c(self) -> nn.Conv2d:
        return self.conv.conv

    def pre_activation(self, c: Tensor, r: Optional[Tensor] = None) -> Interval:
        """Interval right before the ReLU."""
        c, r = self.conv(c, r)
        return interval_bn(self.bn, c, r) if self.bn is not None else (c, r)

    def forward(self, c: Tensor, r: Optional[Tensor] = None) -> Interval:
        return interval_relu(*self.pre_activation(c, r))

    def clean(self, x: Tensor) -> Tensor:
        """Ordinary forward (r = 0) without computing the zero radius."""
        x = self.conv.conv(x)
        if self.bn is not None:
            x = interval_bn(self.bn, x, None)[0]
        return F.relu(x)

    def gain(self) -> float:
        """
        Factor by which the block can enlarge ||r||_inf: kappa(K) times max |gamma / sigma|
        of the BatchNorm (eval mode). ReLU never enlarges a radius.
        """
        g = kappa(self.conv.weight).item()
        if self.bn is not None:
            g *= (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)).abs().max().item()
        return g


def ibp_init_(model: nn.Module, as_in_paper_code: bool = False) -> None:
    """
    IBP initialization (Shi et al. 2021): w ~ N(0, sigma^2), sigma = sqrt(2*pi) / fan_in, bias 0.
    Then E||row of W||_1 = 2, which with ReLU halving keeps the radius roughly constant
    across layers at initialization instead of exploding.
    as_in_paper_code=True follows their manual_init.py exactly: the last layer and all
    biases keep their default initialization.
    """
    layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in layers[:-1] if as_in_paper_code else layers:
        fan_in = m.weight[0].numel()
        nn.init.normal_(m.weight, 0.0, math.sqrt(2 * math.pi) / fan_in)
        if m.bias is not None and not as_in_paper_code:
            nn.init.zeros_(m.bias)


class Transition2dLayer(nn.Module):
    """
    Global average pooling of an interval feature map, kept as an interval.

    The old version concatenated r to the features, which gave the classifier a
    constant zero input. The interval is now passed on to an interval head.
    """

    def __init__(self, in_channels: int, dropout: float = 0.0, **_ignored) -> None:
        super().__init__()
        self.out_dim = in_channels
        self.dropout = dropout

    def forward(self, c: Tensor, r: Tensor) -> Interval:
        c, r = interval_global_avgpool(c, r)
        return interval_dropout(c, r, self.dropout, self.training)
