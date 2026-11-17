"""
CNN-7 with interval max-pooling (draft A).

    cnn7 : conv64 - conv64 - conv128/s2 - conv128 - conv128 - fc512 - fc     (Shi et al. 2021, reference)
    pool1: conv64 - conv64 - conv128 - POOL - conv128 - conv128 - fc512 - fc
           (CNN-7 with its stride-2 convolution replaced by stride 1 + 2x2 pooling)
    pool2: conv64 - conv64 - POOL - conv128 - POOL - conv128 - conv128 - fc512 - fc

Every conv/fc is followed by BatchNorm and ReLU. The same module serves as the normally
trained model (train without IBP) and as the IBP model: with r = 0 every pooling operator
equals F.max_pool2d, so `forward(x)` is the ordinary network. `set_pool(op)` swaps the
operator without touching the weights.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from interval_gnn.cnn_layers import IntervalConv2dLayer, IntervalLinear, input_box, interval_bn, interval_relu
from interval_gnn.pooling import IntervalPool2d

# (out_channels, stride, pool after this conv?)
LAYOUTS = {
    "cnn7": [(64, 1, False), (64, 1, False), (128, 2, False), (128, 1, False), (128, 1, False)],
    "pool1": [(64, 1, False), (64, 1, False), (128, 1, True), (128, 1, False), (128, 1, False)],
    "pool2": [(64, 1, False), (64, 1, True), (128, 1, True), (128, 1, False), (128, 1, False)],
}
FC = 512


class PooledCNN7(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, input_size: int = 32,
                 layout: Literal["cnn7", "pool1", "pool2"] = "pool1", pool: str = "lattice",
                 k: int = 2, tau: float = 1e-6) -> None:
        super().__init__()
        self.layout, self.k = layout, k
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleDict()
        c_in, size = in_channels, input_size
        for i, (c_out, stride, has_pool) in enumerate(LAYOUTS[layout]):
            self.convs.append(IntervalConv2dLayer(c_in, c_out, 3, stride=stride, padding=1, batch_norm=True))
            size = (size + 2 - 3) // stride + 1
            if has_pool:
                self.pools[str(i)] = IntervalPool2d(pool, k, tau)
                size //= k
            c_in = c_out
        self.fc1 = IntervalLinear(c_in * size * size, FC)
        self.bn_fc = nn.BatchNorm1d(FC)
        self.fc2 = IntervalLinear(FC, out_channels)

    @property
    def pool(self) -> Optional[str]:
        return next((p.op for p in self.pools.values()), None)

    def set_pool(self, op: str) -> "PooledCNN7":
        for p in self.pools.values():
            p.op = op
        return self

    def _features(self, c: Tensor, r: Tensor, record: Optional[list] = None,
                  pre: Optional[list] = None) -> Tuple[Tensor, Tensor]:
        """Interval of the input of the last layer. `pre` collects the input box and each ReLU's input bounds."""
        if pre is not None:
            pre.append((c - r, c + r))
        for i, layer in enumerate(self.convs):
            c, r = layer.pre_activation(c, r)
            if pre is not None:
                pre.append((c - r, c + r))
            c, r = interval_relu(c, r)
            if str(i) in self.pools:
                c, r = self.pools[str(i)](c, r)
            if record is not None:
                record.append(r.detach().abs().amax().item())
        c, r = self.fc1(c.flatten(1), r.flatten(1))
        c, r = interval_bn(self.bn_fc, c, r)
        if pre is not None:
            pre.append((c - r, c + r))
        return interval_relu(c, r)

    def forward_interval(self, c: Tensor, r: Optional[Tensor] = None,
                         record: Optional[list] = None) -> Tuple[Tensor, Tensor]:
        if r is None:
            r = torch.zeros_like(c)
        return self.fc2(*self._features(c, r, record))

    def forward(self, x: Tensor) -> Tensor:
        """Clean pass: equals forward_interval(x)[0] but skips the all-zero radius branch."""
        for i, layer in enumerate(self.convs):
            x = layer.clean(x)
            if str(i) in self.pools:
                x = F.max_pool2d(x, self.k)
        x = self.fc1.lin(x.flatten(1))
        x = F.relu(interval_bn(self.bn_fc, x, None)[0])
        return self.fc2.lin(x)

    def bounds(self, x: Tensor, eps, lo=0.0, hi=1.0) -> Tuple[Tensor, Tensor]:
        c, r = self.forward_interval(*input_box(x, eps, lo, hi))
        return c - r, c + r

    def margin_bounds(self, x: Tensor, eps, y: Tensor, lo=0.0, hi=1.0, pre: Optional[list] = None) -> Tensor:
        """
        Lower bounds of f_y(x') - f_j(x') over the box, shape [N, K], with 0 in column y.

        The last linear layer is applied to the margin directly (w_y - w_j), which is exact for
        that layer and tighter than subtracting the bounds of the two logits. This is what
        specification-matrix IBP computes (Gowal et al. 2018; Shi et al. 2021).
        """
        c, r = self._features(*input_box(x, eps, lo, hi), pre=pre)
        w, b = self.fc2.lin.weight, self.fc2.lin.bias
        dw = w[y][:, None, :] - w[None, :, :]               # [N, K, D]
        db = b[y][:, None] - b[None, :]
        return (dw @ c[:, :, None]).squeeze(-1) + db - (dw.abs() @ r[:, :, None]).squeeze(-1)
