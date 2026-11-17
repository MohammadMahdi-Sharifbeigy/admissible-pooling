"""
interval_gnn/layers.py
======================
Sound center-radius interval layers for GNNs (graph version of IBP).

Node features are an interval matrix <C, R> = [C - R, C + R] with R >= 0.
A message-passing layer with a nonnegative propagation matrix Â (self-loops
included) and a weight W maps it to

    M = Â C W^T + b,        S = Â R |W|^T,

which encloses Â X W^T + b for every X in the box (Â >= 0, so the radius
only goes through |W|). The radius uses the *same* W as the center.

Â choices (agg):
    'gcn'   D^{-1/2} (A + I) D^{-1/2}   exactly as in GCNConv (Kipf & Welling)
    'mean'  D^{-1} (A + I)
    'sum'   A + I

The old layer used a separate W_r, a tanh squash and no input radius, so its
radius stayed 0 and did not enclose anything; it has been replaced.

IntervalGINConv implements GIN:  h_i' = MLP((1 + e) h_i + sum_{j in N(i)} h_j),
with the interval rule |1 + e| R for the self term.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree

from .cnn_layers import IntervalLinear, interval_bn, interval_relu

Interval = Tuple[Tensor, Tensor]


# ---------------------------------------------------------------------------
# Propagation matrix Â
# ---------------------------------------------------------------------------

def propagation_matrix(edge_index: Tensor, num_nodes: int, agg: str = "gcn",
                       dtype=torch.float32) -> Tuple[Tensor, Tensor]:
    """Return (edge_index, edge_weight) of Â with self-loops; all weights are >= 0."""
    if agg == "gcn":
        ei, w = gcn_norm(edge_index, None, num_nodes, improved=False, add_self_loops=True, dtype=dtype)
        return ei, w
    ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    if agg == "mean":
        deg = degree(ei[1], num_nodes=num_nodes, dtype=dtype).clamp(min=1)
        return ei, 1.0 / deg[ei[1]]
    if agg == "sum":
        return ei, torch.ones(ei.size(1), dtype=dtype, device=ei.device)
    raise ValueError(f"Unknown agg: {agg!r}")


def propagate(h: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
    """(Â h)_i = sum_j Â_ij h_j with messages flowing source (row 0) -> target (row 1)."""
    src, dst = edge_index
    out = torch.zeros_like(h)
    return out.index_add_(0, dst, edge_weight[:, None] * h[src])


def propagation_inf_norm(edge_index: Tensor, num_nodes: int, agg: str = "gcn") -> float:
    """||Â||_inf = max row sum (the graph factor in the width-growth bound)."""
    ei, w = propagation_matrix(edge_index, num_nodes, agg)
    return torch.zeros(num_nodes, device=w.device).index_add_(0, ei[1], w).max().item()


# ---------------------------------------------------------------------------
# IntervalConvLayer  (GCN / mean / sum message passing)
# ---------------------------------------------------------------------------

class IntervalConvLayer(nn.Module):
    """
    Interval message-passing layer: M = Â C W^T + b, S = Â R |W|^T,
    followed by the interval ReLU when activation=True.

    With agg='gcn' the center path is exactly GCNConv(in, out), so a model built
    from these layers equals BaselineGCN when R = 0.
    r_max, W_bar and hard_clip are accepted for compatibility and ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        agg: Literal["gcn", "mean", "sum"] = "gcn",
        bias: bool = True,
        activation: bool = True,
        r_max: float = 1.0,
        W_bar: float = 0.9,
        hard_clip: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.agg_mode = agg
        self.activation = activation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.lin.weight)  # same init as GCNConv

    @property
    def weight(self) -> Tensor:
        return self.lin.weight

    def forward(self, c: Tensor, edge_index: Tensor, r: Optional[Tensor] = None) -> Interval:
        if r is None:
            r = torch.zeros_like(c)
        ei, w = propagation_matrix(edge_index, c.size(0), self.agg_mode, c.dtype)
        m = propagate(self.lin(c), ei, w)
        if self.bias is not None:
            m = m + self.bias
        s = propagate(r @ self.lin.weight.abs().t(), ei, w)
        return interval_relu(m, s) if self.activation else (m, s)

    def radius_weight_inf_norm(self) -> Tensor:
        """||W||_inf (max row sum of |W|); the per-layer feature factor of the width bound."""
        return self.lin.weight.abs().sum(dim=1).max()

    def extra_repr(self) -> str:
        return f"in={self.in_channels}, out={self.out_channels}, agg={self.agg_mode}"


# ---------------------------------------------------------------------------
# IntervalGINConv
# ---------------------------------------------------------------------------

class IntervalGINConv(nn.Module):
    """
    Interval GIN layer, same parameters as
        GINConv(Sequential(Linear, BatchNorm1d, ReLU, Linear), train_eps=True).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.fc1 = IntervalLinear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.fc2 = IntervalLinear(out_channels, out_channels)

    def forward(self, c: Tensor, edge_index: Tensor, r: Tensor) -> Interval:
        src, dst = edge_index
        a = 1 + self.eps
        mc = a * c + torch.zeros_like(c).index_add_(0, dst, c[src])
        mr = a.abs() * r + torch.zeros_like(r).index_add_(0, dst, r[src])
        c, r = self.fc1(mc, mr)
        c, r = interval_bn(self.bn, c, r)
        c, r = interval_relu(c, r)
        return self.fc2(c, r)
