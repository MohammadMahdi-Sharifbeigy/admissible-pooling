"""
interval_gnn/data.py
====================
GPU-resident image loaders.

The image datasets here are small enough (CIFAR-10 train is ~150 MB as uint8) to
keep entirely on the GPU. Batches are then sliced from one tensor, and random
crop / horizontal flip run on the GPU, so there are no DataLoader worker processes.
On Windows those workers are re-spawned every epoch for every loader, which made
each epoch slow and sometimes hung at the start of validation.

Behaviour matches the torchvision pipeline used before:
    RandomHorizontalFlip() -> RandomCrop(size, padding=pad, fill=0) -> ToTensor() -> Normalize(mean, std)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def dataset_to_tensors(ds) -> tuple[Tensor, Tensor]:
    """Raw images [N, C, H, W] uint8 and labels [N] from a torchvision CIFAR / MNIST-style dataset."""
    x = ds.data
    x = torch.from_numpy(np.asarray(x)) if not torch.is_tensor(x) else x
    if x.dim() == 3:                 # MNIST-style [N, H, W]
        x = x[:, None]
    elif x.shape[-1] in (1, 3):      # CIFAR-style [N, H, W, C]
        x = x.permute(0, 3, 1, 2)
    y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    return x.contiguous(), y


class GPUImageLoader:
    """
    Iterable of (x, y) batches already on `device`, normalized as (x/255 - mean) / std.

    augment=True applies, per image, a random horizontal flip and a random crop
    from the image zero-padded by `pad` pixels (zero = black before normalization).
    """

    def __init__(self, x_uint8: Tensor, y: Tensor, batch_size: int, shuffle: bool = False,
                 augment: bool = False, pad: int = 4, flip: bool = True, pad_mode: str = "constant",
                 mean: Sequence[float] = (0.5,), std: Sequence[float] = (0.5,),
                 device="cuda", seed: Optional[int] = None) -> None:
        self.device = torch.device(device)
        self.x = x_uint8.to(self.device)
        self.y = y.to(self.device)
        self.batch_size = batch_size
        self.shuffle, self.augment, self.pad, self.flip = shuffle, augment, pad, flip
        self.pad_mode = pad_mode
        c = self.x.size(1)
        self.mean = torch.tensor(mean, device=self.device).view(1, -1, 1, 1).expand(1, c, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, -1, 1, 1).expand(1, c, 1, 1)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.dataset = torch.utils.data.TensorDataset(self.x, self.y)  # for len(loader.dataset)

    def __len__(self) -> int:
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def _augment(self, x: Tensor) -> Tensor:
        n, _, h, w = x.shape
        if self.flip:
            m = torch.rand(n, device=self.device, generator=self.gen) < 0.5
            x = torch.where(m[:, None, None, None], x.flip(3), x)
        if self.pad > 0:
            p = self.pad
            xp = F.pad(x, (p, p, p, p), mode=self.pad_mode)
            i = torch.randint(0, 2 * p + 1, (n,), device=self.device, generator=self.gen)
            j = torch.randint(0, 2 * p + 1, (n,), device=self.device, generator=self.gen)
            rows = (i[:, None] + torch.arange(h, device=self.device))[:, None, :, None]
            cols = (j[:, None] + torch.arange(w, device=self.device))[:, None, None, :]
            ch = torch.arange(x.size(1), device=self.device)[None, :, None, None]
            nn_ = torch.arange(n, device=self.device)[:, None, None, None]
            x = xp[nn_, ch, rows, cols]
        return x

    def __iter__(self):
        n = len(self.y)
        order = (torch.randperm(n, device=self.device, generator=self.gen) if self.shuffle
                 else torch.arange(n, device=self.device))
        for k in range(0, n, self.batch_size):
            idx = order[k:k + self.batch_size]
            x = self.x[idx].float().div_(255)
            if self.augment:
                x = self._augment(x)
            yield (x - self.mean) / self.std, self.y[idx]
