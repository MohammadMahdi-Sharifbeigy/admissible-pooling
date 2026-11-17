"""
Interval max-pooling operators for draft A (admissible orders).

Every operator takes a feature map in center-radius form <c, r> of shape [N, C, H, W],
splits it into non-overlapping k x k windows (odd borders are cropped, like F.max_pool2d)
and returns one interval per window, again as <c, r>.

    lattice  : [max lo, max hi]                      exact range of the maximum (sound)
    lex2     : admissible max for the order "upper endpoint first, then lower"   (sound)
    lex1     : admissible max for the order "lower endpoint first, then upper"   (unsound)
    xuyager  : admissible max for the Xu-Yager order (midpoint, ties by upper)   (unsound)

With r = 0 all four equal F.max_pool2d, so the clean prediction does not depend on the
operator. `tau` is the tolerance for floating-point ties on the first key.
Gradients flow through the gathered endpoints only (the selection itself is piecewise
constant), as described in the paper.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Interval = Tuple[Tensor, Tensor]

POOL_OPS = ("lattice", "lex2", "lex1", "xuyager")
SOUND_OPS = ("lattice", "lex2")


def _windows(x: Tensor, k: int) -> Tensor:
    """[N, C, H, W] -> [N, C, H//k, W//k, k*k] with non-overlapping windows."""
    n, ch, h, w = x.shape
    h2, w2 = h // k, w // k
    x = x[:, :, : h2 * k, : w2 * k]
    return x.reshape(n, ch, h2, k, w2, k).permute(0, 1, 2, 4, 3, 5).reshape(n, ch, h2, w2, k * k)


def _select(first: Tensor, second: Tensor, lo: Tensor, hi: Tensor, tau: float) -> Interval:
    """Lexicographic argmax: maximize `first`, break ties (within tau) by `second`; gather [lo, hi]."""
    top = first.max(dim=-1, keepdim=True).values
    cand = first >= top - tau
    key = torch.where(cand, second, torch.full_like(second, float("-inf")))
    j = key.argmax(dim=-1, keepdim=True)
    return lo.gather(-1, j).squeeze(-1), hi.gather(-1, j).squeeze(-1)


def interval_pool(c: Tensor, r: Tensor, op: str = "lattice", k: int = 2, tau: float = 1e-6) -> Interval:
    """Pool <c, r> over k x k windows with one of POOL_OPS; returns <c, r>."""
    if op == "lattice":
        lo, hi = F.max_pool2d(c - r, k), F.max_pool2d(c + r, k)
        return (lo + hi) / 2, (hi - lo) / 2
    lo, hi = _windows(c - r, k), _windows(c + r, k)
    if op == "lex2":
        plo, phi = _select(hi, lo, lo, hi, tau)
    elif op == "lex1":
        plo, phi = _select(lo, hi, lo, hi, tau)
    elif op == "xuyager":
        plo, phi = _select(lo + hi, hi, lo, hi, tau)
    else:
        raise ValueError(f"Unknown pooling operator {op!r}; choose from {POOL_OPS}")
    return (plo + phi) / 2, (phi - plo) / 2


class IntervalPool2d(nn.Module):
    """Module wrapper; `op` may be changed after training to evaluate other operators on the same weights."""

    def __init__(self, op: str = "lattice", k: int = 2, tau: float = 1e-6) -> None:
        super().__init__()
        if op not in POOL_OPS:
            raise ValueError(f"Unknown pooling operator {op!r}; choose from {POOL_OPS}")
        self.op, self.k, self.tau = op, k, tau

    def forward(self, c: Tensor, r: Tensor) -> Interval:
        return interval_pool(c, r, self.op, self.k, self.tau)

    def extra_repr(self) -> str:
        return f"op={self.op}, k={self.k}, tau={self.tau}"


def excess_width(c: Tensor, r: Tensor, op: str, k: int = 2) -> Tensor:
    """Width of `op` minus width of the lattice maximum, per window (>= 0 for lex2)."""
    _, r_op = interval_pool(c, r, op, k)
    _, r_lat = interval_pool(c, r, "lattice", k)
    return 2 * (r_op - r_lat)
