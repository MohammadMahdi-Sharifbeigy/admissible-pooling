import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
from torch import Tensor

from interval_gnn.cnn_layers import (
    IntervalConv2dLayer, IntervalLinear, Transition2dLayer,
    input_box, interval_bn, interval_dropout, interval_relu,
)
from interval_gnn.models import _Head
from interval_gnn.utils import RadiusTracker

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
# 'small': n_layers conv3x3/stride 2 (hidden channels) -> GAP -> Linear-ReLU-Dropout-Linear
#          (the original model of this project).
# 'cnn7' : the CNN-7 of Shi et al. (2021), standard for IBP:
#          conv64 - conv64 - conv128/s2 - conv128 - conv128 - fc512 - fc(out),
#          every conv/fc followed by BatchNorm and ReLU, flatten instead of GAP, no dropout.

CNN7_CONVS = [(64, 1), (64, 1), (128, 2), (128, 1), (128, 1)]   # (channels, stride)
CNN7_FC = 512


def _conv_specs(arch, in_channels, hidden_channels, n_layers):
    if arch == "small":
        return [(in_channels if i == 0 else hidden_channels, hidden_channels, 2) for i in range(n_layers)]
    if arch == "cnn7":
        specs, c_in = [], in_channels
        for c_out, s in CNN7_CONVS:
            specs.append((c_in, c_out, s))
            c_in = c_out
        return specs
    raise ValueError(f"Unknown arch: {arch!r}")


def _flat_dim(arch, specs, input_size):
    if arch == "small":
        return specs[-1][1]
    size = input_size
    for _, _, s in specs:
        size = (size + 2 - 3) // s + 1
    return specs[-1][1] * size * size


# ---------------------------------------------------------------------------
# IntervalCNN
# ---------------------------------------------------------------------------

class IntervalCNN(nn.Module):
    """
    Interval CNN for image classification, trained with IBP.

    With r = 0 it computes exactly the same function as BaselineCNN with the same
    `arch` (same layers, BatchNorm, pooling, head and dropout), so any accuracy gap
    comes from the interval training, not from the architecture.

    forward(x)            -> center logits (clean prediction)
    forward(x, r_init=r)  -> center logits of the box <x, r>
    bounds(x, eps)        -> (lo, hi) logit bounds over the clipped eps-ball
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 3,
        dropout: float = 0.5,
        arch: Literal["small", "cnn7"] = "small",
        input_size: int = 32,
        track_radius: bool = False,
        interval_weights: bool = False,
        **_deprecated,  # r_max, W_bar, transition: no longer used
    ) -> None:
        super().__init__()
        self.arch = arch
        self.dropout = dropout if arch == "small" else 0.0
        bn = arch == "cnn7"
        specs = _conv_specs(arch, in_channels, hidden_channels, n_layers)

        self.interval_layers = nn.ModuleList(
            IntervalConv2dLayer(ci, co, kernel_size=3, stride=s, padding=1,
                                interval_weights=interval_weights, batch_norm=bn)
            for ci, co, s in specs
        )
        self.transition = Transition2dLayer(specs[-1][1]) if arch == "small" else None

        flat = _flat_dim(arch, specs, input_size)
        fc = hidden_channels if arch == "small" else CNN7_FC
        self.fc1 = IntervalLinear(flat, fc)
        self.bn_fc = nn.BatchNorm1d(fc) if bn else None
        self.fc2 = IntervalLinear(fc, out_channels)

        self.radius_tracker: Optional[RadiusTracker] = RadiusTracker() if track_radius else None

    def forward_interval(self, c: Tensor, r: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Propagate <c, r> and return logit intervals in center-radius form."""
        if r is None:
            r = torch.zeros_like(c)
        if self.radius_tracker is not None:
            self.radius_tracker.reset()

        for idx, layer in enumerate(self.interval_layers):
            c, r = layer(c, r)
            if self.radius_tracker is not None:
                self.radius_tracker.record(r, layer=idx)

        if self.transition is not None:
            c, r = self.transition(c, r)
        else:
            c, r = c.flatten(1), r.flatten(1)
        c, r = self.fc1(c, r)
        if self.bn_fc is not None:
            c, r = interval_bn(self.bn_fc, c, r)
        c, r = interval_relu(c, r)
        c, r = interval_dropout(c, r, self.dropout, self.training)
        return self.fc2(c, r)

    def forward(self, x: Tensor, r_init: Optional[Tensor] = None) -> Tensor:
        return self.forward_interval(x, r_init)[0]

    def bounds(self, x: Tensor, eps, lo: float = 0.0, hi: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Sound lower/upper logit bounds for every input in [x-eps, x+eps] ∩ [lo, hi]."""
        c, r = self.forward_interval(*input_box(x, eps, lo, hi))
        return c - r, c + r

    @classmethod
    def from_baseline(cls, base: "BaselineCNN") -> "IntervalCNN":
        """IntervalCNN with the baseline's weights and BatchNorm statistics (eval mode)."""
        net = cls(**base.config).to(next(base.parameters()).device)
        for layer, conv, bn in zip(net.interval_layers, base.convs, base.bns or [None] * len(base.convs)):
            layer.conv.conv.load_state_dict(conv.state_dict())
            if bn is not None:
                layer.bn.load_state_dict(bn.state_dict())
        if base.arch == "small":
            net.fc1.lin.load_state_dict(base.head.mlp[0].state_dict())
            net.fc2.lin.load_state_dict(base.head.mlp[3].state_dict())
        else:
            net.fc1.lin.load_state_dict(base.head[0].state_dict())
            net.bn_fc.load_state_dict(base.head[1].state_dict())
            net.fc2.lin.load_state_dict(base.head[3].state_dict())
        return net.eval()


# ---------------------------------------------------------------------------
# BaselineCNN
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    Standard CNN ablation control (same `arch` options as IntervalCNN).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 3,
        dropout: float = 0.5,
        arch: Literal["small", "cnn7"] = "small",
        input_size: int = 32,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.config = dict(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                           n_layers=n_layers, dropout=dropout, arch=arch, input_size=input_size)
        specs = _conv_specs(arch, in_channels, hidden_channels, n_layers)

        self.convs = nn.ModuleList(nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1) for ci, co, s in specs)
        self.bns = nn.ModuleList(nn.BatchNorm2d(co) for _, co, _ in specs) if arch == "cnn7" else None

        if arch == "small":
            self.head = _Head(hidden_channels, hidden_channels, out_channels, dropout)
            self.dropout = dropout
        else:
            flat = _flat_dim(arch, specs, input_size)
            self.head = nn.Sequential(nn.Linear(flat, CNN7_FC), nn.BatchNorm1d(CNN7_FC), nn.ReLU(),
                                      nn.Linear(CNN7_FC, out_channels))
            self.dropout = 0.0

    def forward(self, x: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if self.bns is not None:
                x = self.bns[i](x)
            x = F.relu(x)

        if self.arch == "small":
            x = x.mean(dim=(2, 3))  # Global Avg Pooling
        else:
            x = x.flatten(1)

        return self.head(x)
