"""
interval_gnn/models.py
======================
Interval GNNs trained with IBP and their point-valued baselines.

Every interval model has the same architecture as its baseline, so with R = 0
it computes the same function; differences in accuracy come from interval
training, not from the architecture.

    IntervalGCN  <->  BaselineGCN   (node or graph classification)
    IntervalGIN  <->  BaselineGIN   (graph classification, MUTAG)

Interval models share this interface:
    forward(x, edge_index, batch=None, r=None)          -> center logits
    forward_interval(x, edge_index, batch=None, r=None) -> (C, R) of the logits
    bounds(x, edge_index, eps, batch=None, lo=0, hi=1)  -> (lower, upper) logits
        for every feature matrix in [x - eps, x + eps] ∩ [lo, hi].

The old IntervalGAT has been removed: attention weights depend on the features
through a softmax, so the simple radius rule does not bound them.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool

from .cnn_layers import IntervalLinear, input_box, interval_bn, interval_dropout, interval_relu
from .layers import IntervalConvLayer, IntervalGINConv
from .utils import RadiusTracker


# ---------------------------------------------------------------------------
# Shared prediction head
# ---------------------------------------------------------------------------

class _Head(nn.Module):
    """Two-layer MLP prediction head with dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class _IntervalHead(nn.Module):
    """Interval version of _Head (same parameter layout: fc1, fc2)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.fc1 = IntervalLinear(in_dim, hidden_dim)
        self.fc2 = IntervalLinear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, c: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]:
        c, r = interval_relu(*self.fc1(c, r))
        c, r = interval_dropout(c, r, self.dropout, self.training)
        return self.fc2(c, r)

    def load_from(self, head: _Head) -> None:
        self.fc1.lin.load_state_dict(head.mlp[0].state_dict())
        self.fc2.lin.load_state_dict(head.mlp[3].state_dict())


class _IntervalGNN(nn.Module):
    """Shared center-logits / bounds interface."""

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None,
                r: Optional[Tensor] = None) -> Tensor:
        return self.forward_interval(x, edge_index, batch, r)[0]

    def bounds(self, x: Tensor, edge_index: Tensor, eps, batch: Optional[Tensor] = None,
               lo: float = 0.0, hi: float = 1.0) -> Tuple[Tensor, Tensor]:
        c0, r0 = input_box(x, eps, lo, hi)
        c, r = self.forward_interval(c0, edge_index, batch, r0)
        return c - r, c + r

    def _record(self, r: Tensor, idx: int) -> None:
        if self.radius_tracker is not None:
            self.radius_tracker.record(r, layer=idx)


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------

class IntervalGCN(_IntervalGNN):
    """
    n_layers interval GCN layers (Â = D^{-1/2}(A+I)D^{-1/2}) -> [global mean pool] -> interval head.
    Same architecture and dropout as BaselineGCN.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        dropout: float = 0.5,
        agg: Literal["gcn", "mean", "sum"] = "gcn",
        track_radius: bool = False,
        **_deprecated,  # r_max, W_bar, transition, proj_dim: no longer used
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            IntervalConvLayer(in_channels if i == 0 else hidden_channels, hidden_channels, agg=agg)
            for i in range(n_layers)
        )
        self.head = _IntervalHead(hidden_channels, hidden_channels, out_channels, dropout)
        self.dropout = dropout
        self.radius_tracker: Optional[RadiusTracker] = RadiusTracker() if track_radius else None

    def forward_interval(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None,
                         r: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        c = x
        r = torch.zeros_like(x) if r is None else r
        if self.radius_tracker is not None:
            self.radius_tracker.reset()
        for idx, conv in enumerate(self.convs):
            c, r = conv(c, edge_index, r)
            self._record(r, idx)
            c, r = interval_dropout(c, r, self.dropout, self.training)
        if batch is not None:
            c, r = global_mean_pool(c, batch), global_mean_pool(r, batch)
        return self.head(c, r)

    @classmethod
    def from_baseline(cls, base: "BaselineGCN") -> "IntervalGCN":
        """IntervalGCN with the baseline's weights (to bound a normally trained model)."""
        conv0 = base.convs[0]
        net = cls(conv0.in_channels, conv0.out_channels, base.head.mlp[-1].out_features,
                  n_layers=len(base.convs), dropout=base.dropout)
        for iv, conv in zip(net.convs, base.convs):
            iv.lin.weight.data.copy_(conv.lin.weight.data)
            iv.bias.data.copy_(conv.bias.data)
        net.head.load_from(base.head)
        return net.to(conv0.lin.weight.device).eval()


class BaselineGCN(nn.Module):
    """
    Plain multi-layer GCN — no interval mechanism.
    Used as the ablation control against IntervalGCN.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                GCNConv(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                )
            )
        self.head = _Head(hidden_channels, hidden_channels, out_channels, dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self.head(x)


# ---------------------------------------------------------------------------
# GIN
# ---------------------------------------------------------------------------

class IntervalGIN(_IntervalGNN):
    """
    n_layers interval GIN layers (each: GIN MLP -> BatchNorm -> ReLU -> dropout)
    -> global add pool -> interval head. Same architecture as BaselineGIN.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 3,
        dropout: float = 0.5,
        track_radius: bool = False,
        **_deprecated,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            IntervalGINConv(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(n_layers)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for _ in range(n_layers))
        self.head = _IntervalHead(hidden_channels, hidden_channels, out_channels, dropout)
        self.dropout = dropout
        self.radius_tracker: Optional[RadiusTracker] = RadiusTracker() if track_radius else None

    def forward_interval(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None,
                         r: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        c = x
        r = torch.zeros_like(x) if r is None else r
        if self.radius_tracker is not None:
            self.radius_tracker.reset()
        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            c, r = conv(c, edge_index, r)
            c, r = interval_relu(*interval_bn(bn, c, r))
            self._record(r, idx)
            c, r = interval_dropout(c, r, self.dropout, self.training)
        if batch is None:
            batch = torch.zeros(c.size(0), dtype=torch.long, device=c.device)
        c, r = global_add_pool(c, batch), global_add_pool(r, batch)
        return self.head(c, r)

    @classmethod
    def from_baseline(cls, base: "BaselineGIN") -> "IntervalGIN":
        """IntervalGIN with the baseline's weights and BatchNorm statistics."""
        lin0 = base.convs[0].nn[0]
        net = cls(lin0.in_features, lin0.out_features, base.head.mlp[-1].out_features,
                  n_layers=len(base.convs), dropout=base.dropout)
        for iv, conv, ivbn, bn in zip(net.convs, base.convs, net.bns, base.bns):
            iv.eps.data.copy_(conv.eps.data)
            iv.fc1.lin.load_state_dict(conv.nn[0].state_dict())
            iv.bn.load_state_dict(conv.nn[1].state_dict())
            iv.fc2.lin.load_state_dict(conv.nn[3].state_dict())
            ivbn.load_state_dict(bn.state_dict())
        net.head.load_from(base.head)
        return net.to(lin0.weight.device).eval()


class BaselineGIN(nn.Module):
    """Plain GIN with the same architecture as IntervalGIN."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.head = _Head(hidden_channels, hidden_channels, out_channels, dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.head(global_add_pool(x, batch))
