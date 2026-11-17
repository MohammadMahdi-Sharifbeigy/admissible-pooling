"""
Shared training loop for all drafts.

A run lives in  <root>/<draft>/<dataset>/<name>/seed<k>/ :

    config.json            full config + metadata (torch version, device, n_params, start time)
    metrics.csv            one row per epoch, rewritten after every epoch
    checkpoints/last.pt    model, optimizer, scheduler, epoch, step, best score, RNG states
    checkpoints/best.pt    the same for the selected epoch
    test_results.csv       final evaluation over the eps list (written once, after training)
    done.flag              written last; sweeps skip runs that have it

A Task adapts a data set to the loop: it yields batches and knows how to get clean logits
and IBP bounds from a model, the clipping range of the inputs and the scale between the
user's eps (pixel / feature units) and the model's input space.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from interval_gnn.losses import (certified, eps_kappa_schedule, ibp_loss, margin_certified, smoothed_eps,
                                 worst_case_logits)
from interval_gnn.regularizers import warmup_regularizer

Batch = Tuple[tuple, torch.Tensor, Optional[torch.Tensor]]   # (model inputs, labels, mask)


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

class RunDir:
    def __init__(self, root: str, draft: str, dataset: str, name: str, seed: int) -> None:
        self.path = os.path.join(root, draft, dataset, name, f"seed{seed}")
        self.ckpt_dir = os.path.join(self.path, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    @property
    def last(self) -> str:
        return os.path.join(self.ckpt_dir, "last.pt")

    @property
    def best(self) -> str:
        return os.path.join(self.ckpt_dir, "best.pt")

    @property
    def done(self) -> bool:
        return os.path.exists(self.file("done.flag"))

    def write_config(self, cfg: dict, **meta) -> None:
        with open(self.file("config.json"), "w", encoding="utf-8") as f:
            json.dump(dict(cfg, _meta=dict(torch=torch.__version__, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                           **meta)), f, indent=2)

    def mark_done(self) -> None:
        with open(self.file("done.flag"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    s = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: dict) -> None:
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"].cpu())
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class Task:
    """Interface between a data set and the training loop."""
    lo: float = 0.0
    hi: float = 1.0
    eps_scale: float = 1.0            # model-space radius = eps_scale * user eps

    def batches(self, split: str) -> Iterator[Batch]:
        raise NotImplementedError

    def steps_per_epoch(self) -> int:
        raise NotImplementedError

    def logits(self, model: nn.Module, inp: tuple) -> torch.Tensor:
        return model(*inp)

    def bounds(self, model: nn.Module, inp: tuple, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return model.bounds(*inp, eps * self.eps_scale, lo=self.lo, hi=self.hi)

    def interval_model(self, model: nn.Module) -> nn.Module:
        """Model that has .bounds (normally trained baselines are copied into their interval twin)."""
        return model

    def margins(self, model: nn.Module, inp: tuple, y: torch.Tensor, eps: float, pre: Optional[list] = None):
        """Lower bounds of f_y - f_j, [N, K] with 0 in column y (tight last layer)."""
        raise NotImplementedError


class ImageTask(Task):
    """Image classification with GPUImageLoader-style loaders {'train', 'val', 'test'}."""

    def __init__(self, loaders: dict, std=(0.5,), mean=(0.5,), device=None) -> None:
        self.loaders = loaders
        s = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        m = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.eps_scale = 1.0 / s
        self.lo, self.hi = (0.0 - m) / s, (1.0 - m) / s

    def margins(self, model, inp, y, eps, pre=None):
        return model.margin_bounds(inp[0], eps * self.eps_scale, y, self.lo, self.hi, pre=pre)

    def batches(self, split):
        for x, y in self.loaders[split]:
            yield (x,), y, None

    def steps_per_epoch(self):
        return len(self.loaders["train"])

    def bounds(self, model, inp, eps):
        return model.bounds(inp[0], eps * self.eps_scale, self.lo, self.hi)

    def interval_model(self, model):
        from interval_gnn.cnn_models import BaselineCNN, IntervalCNN
        return IntervalCNN.from_baseline(model) if isinstance(model, BaselineCNN) else model


class GraphTask(Task):
    """Node classification on one graph (data has train/val/test masks) or graph classification (loaders)."""

    def __init__(self, data=None, loaders: Optional[dict] = None, device=None) -> None:
        assert (data is None) != (loaders is None)
        self.data, self.loaders, self.device = data, loaders, device

    @property
    def node_level(self) -> bool:
        return self.data is not None

    def batches(self, split):
        if self.node_level:
            d = self.data
            yield (d.x, d.edge_index, None), d.y, getattr(d, f"{split}_mask")
        else:
            for b in self.loaders[split]:
                b = b.to(self.device)
                yield (b.x.float(), b.edge_index, b.batch), b.y, None

    def steps_per_epoch(self):
        return 1 if self.node_level else len(self.loaders["train"])

    def bounds(self, model, inp, eps):
        x, ei, batch = inp
        return model.bounds(x, ei, eps * self.eps_scale, batch, self.lo, self.hi)

    def interval_model(self, model):
        from interval_gnn.models import BaselineGCN, BaselineGIN, IntervalGCN, IntervalGIN
        if isinstance(model, BaselineGCN):
            return IntervalGCN.from_baseline(model)
        if isinstance(model, BaselineGIN):
            return IntervalGIN.from_baseline(model)
        return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, task: Task, split: str, eps: float, desc: Optional[str] = None,
             margin: bool = False) -> dict:
    """Clean loss/accuracy and IBP-verified loss/accuracy at radius eps (user units).
    margin=True certifies with the margin bounds (tight last layer) instead of logit intervals."""
    model.eval()
    net = task.interval_model(model)
    loss = rob = acc = ver = tot = 0.0
    it = task.batches(split)
    if desc:
        it = tqdm(it, desc=desc, leave=False)
    for inp, y, mask in it:
        out = task.logits(net, inp)
        if margin:
            lb = task.margins(net, inp, y, eps)
            rob += F.cross_entropy(-lb, y, reduction="sum").item()
            ver += margin_certified(lb, y).sum().item()
        else:
            lo, hi = task.bounds(net, inp, eps)
            if mask is not None:
                out, lo, hi, y = out[mask], lo[mask], hi[mask], y[mask]
            rob += F.cross_entropy(worst_case_logits(lo, hi, y), y, reduction="sum").item()
            ver += certified(lo, hi, y).sum().item()
        loss += F.cross_entropy(out, y, reduction="sum").item()
        acc += out.argmax(1).eq(y).sum().item()
        tot += y.numel()
    return dict(loss=loss / tot, robust_loss=rob / tot, acc=acc / tot, verified=ver / tot)


def test_table(model: nn.Module, task: Task, eps_list: List[float], name: str = "",
               margin: bool = False) -> pd.DataFrame:
    rows = []
    for e in eps_list:
        r = evaluate(model, task, "test", e, desc=f"test eps={e:g}", margin=margin)
        rows.append(dict(model=name, eps=e, clean_acc=r["acc"], verified_acc=r["verified"],
                         clean_loss=r["loss"], robust_loss=r["robust_loss"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    epochs=100, lr=5e-4, weight_decay=0.0,
    scheduler="multistep", milestones=[0.75, 0.875], gamma=0.2,   # milestones as fractions of epochs
    grad_clip=10.0,
    ibp=True, eps=0.1, warmup_epochs=1, ramp_epochs=40, kappa_end=0.5,
    eps_schedule="linear",      # 'smoothed': Shi et al.; kappa then stays at kappa_end
    margin=False,               # use margin bounds (tight last layer) in loss and certification
    reg_lambda=0.0, reg_tol=0.5, reg_min_eps=1e-6,   # warm-up regularizers of Shi et al.
    select="best",              # 'last': keep the final epoch instead of selecting on validation
)


class Trainer:
    """
    IBP or standard training with resumable checkpoints and per-epoch CSV logging.

    cfg keys (see DEFAULTS): epochs, lr, weight_decay, scheduler ('multistep' | 'cosine' | 'none'),
    milestones, gamma, grad_clip (None to disable), ibp, eps (user units), warmup_epochs,
    ramp_epochs, kappa_end, eps_schedule, margin, reg_lambda, reg_tol, select. Model selection:
    best validation verified accuracy at eps once the ramp is over (IBP), or best clean
    validation accuracy (standard training); select='last' keeps the final epoch.
    `extra_loss(model, task, inp, y, mask, eps_model, epoch)` may add a regularizer to the loss.
    """

    def __init__(self, model: nn.Module, task: Task, cfg: dict, run: RunDir,
                 extra_loss: Optional[Callable] = None, show_batches: Optional[bool] = None) -> None:
        self.model, self.task, self.run = model, task, run
        self.cfg = dict(DEFAULTS, **cfg)
        self.extra_loss = extra_loss
        c = self.cfg
        self.opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        if c["scheduler"] == "multistep":
            ms = [int(round(f * c["epochs"])) for f in c["milestones"]]
            self.sched = torch.optim.lr_scheduler.MultiStepLR(self.opt, ms, gamma=c["gamma"])
        elif c["scheduler"] == "cosine":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=c["epochs"])
        else:
            self.sched = None
        spe = task.steps_per_epoch()
        self.show_batches = spe >= 5 if show_batches is None else show_batches
        self.warmup_steps, self.ramp_steps = c["warmup_epochs"] * spe, c["ramp_epochs"] * spe

    # -- one epoch ---------------------------------------------------------
    def _train_epoch(self, step: int, epoch: int) -> Tuple[dict, int]:
        c, model, task = self.cfg, self.model, self.task
        model.train()
        tl = tcl = ta = tot = 0.0
        gn, gmax = [], 0.0
        eps = 0.0
        kappa = 1.0
        it = task.batches("train")
        bar = tqdm(it, total=task.steps_per_epoch(), desc="  train", leave=False) if self.show_batches else it
        for inp, y, mask in bar:
            self.opt.zero_grad(set_to_none=True)
            out = task.logits(model, inp)                     # clean pass first (BatchNorm statistics)
            if c["ibp"] and c["eps_schedule"] == "smoothed":
                eps, kappa = smoothed_eps(step, self.warmup_steps, self.ramp_steps, c["eps"]), c["kappa_end"]
            elif c["ibp"]:
                eps, kappa = eps_kappa_schedule(step, self.warmup_steps, self.ramp_steps, c["eps"], c["kappa_end"])
            reg_on = c["ibp"] and c["reg_lambda"] > 0 and step < self.warmup_steps + self.ramp_steps
            pre = [] if reg_on else None
            lo = hi = lb = None
            if c["ibp"] and c["margin"] and (eps > 0 or reg_on):
                lb = task.margins(model, inp, y, max(eps, c["reg_min_eps"]) if reg_on else eps, pre=pre)
            elif c["ibp"] and eps > 0:
                lo, hi = task.bounds(model, inp, eps)
            if mask is not None:
                out, y = out[mask], y[mask]
                if lo is not None:
                    lo, hi = lo[mask], hi[mask]
            clean = F.cross_entropy(out, y)
            if lb is not None and eps > 0:
                loss = kappa * clean + (1 - kappa) * F.cross_entropy(-lb, y)
            else:
                loss = ibp_loss(out, lo, hi, y, kappa) if lo is not None else clean
            if reg_on:
                loss = loss + c["reg_lambda"] * (1 - eps / c["eps"]) * warmup_regularizer(pre, c["reg_tol"])
            if self.extra_loss is not None:
                loss = loss + self.extra_loss(model, task, inp, y, mask, eps * task.eps_scale, epoch)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  c["grad_clip"] if c["grad_clip"] else float("inf"))
            norm = float(norm)
            gn.append(norm)
            gmax = max(gmax, norm) if math.isfinite(norm) else gmax
            self.opt.step()
            step += 1
            n = y.numel()
            tl += loss.item() * n
            tcl += clean.item() * n
            ta += out.argmax(1).eq(y).sum().item()
            tot += n
            if self.show_batches:
                bar.set_postfix(loss=f"{tl / tot:.4f}", acc=f"{ta / tot:.4f}", eps=f"{eps:.4g}")
        return dict(eps=eps, kappa=kappa, train_loss=tl / tot, train_clean_loss=tcl / tot, train_acc=ta / tot,
                    grad_norm=float(np.mean(gn)), grad_norm_max=gmax), step

    # -- full run ----------------------------------------------------------
    def fit(self, val_split: str = "val", stop_after: Optional[int] = None) -> pd.DataFrame:
        """Train to cfg['epochs'] (resuming from last.pt). stop_after=k stops after epoch k without
        loading best.pt, as an interruption would; calling fit() again continues from there."""
        c, run, model = self.cfg, self.run, self.model
        csv = run.file("metrics.csv")
        best, start, step, rows = -1.0, 0, 0, []
        if os.path.exists(run.last):
            ck = torch.load(run.last, map_location=next(model.parameters()).device, weights_only=False)
            model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            if self.sched is not None and ck.get("sched") is not None:
                self.sched.load_state_dict(ck["sched"])
            start, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
            _set_rng_state(ck["rng"])
            if os.path.exists(csv):
                rows = pd.read_csv(csv).iloc[:start].to_dict("records")
            print(f"Resumed {run.path} at epoch {start}")

        desc = "/".join(os.path.normpath(run.path).split(os.sep)[-4:])   # draft/dataset/name/seed
        end = c["epochs"] if stop_after is None else min(stop_after, c["epochs"])
        bar = tqdm(range(start, end), desc=desc, initial=start, total=c["epochs"])
        for ep in bar:
            t0 = time.time()
            tr, step = self._train_epoch(step, ep)
            va = evaluate(model, self.task, val_split, c["eps"] if c["ibp"] else c.get("eps_select", c["eps"]),
                          margin=c["margin"])
            lr_now = self.opt.param_groups[0]["lr"]
            if self.sched is not None:
                self.sched.step()
            score = va["verified"] if c["ibp"] else va["acc"]
            # best_score / is_best are always logged; with select='last' best.pt is not written and the
            # final epoch is kept, since with full_train the monitor images are part of the training set
            eligible = (not c["ibp"]) or ep + 1 >= c["warmup_epochs"] + c["ramp_epochs"]
            is_best = eligible and score > best
            if is_best:
                best = score
            rows.append(dict(epoch=ep + 1, step=step, **tr, val_loss=va["loss"], val_robust_loss=va["robust_loss"],
                             val_acc=va["acc"], val_verified=va["verified"], best_score=best, is_best=is_best,
                             lr=lr_now, epoch_time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(csv, index=False)
            state = dict(epoch=ep, step=step, best=best, model=model.state_dict(), opt=self.opt.state_dict(),
                         sched=self.sched.state_dict() if self.sched is not None else None, rng=_rng_state())
            torch.save(state, run.last)
            if is_best and c["select"] != "last":
                torch.save(state, run.best)
            bar.set_postfix(loss=f"{tr['train_loss']:.4f}", val_loss=f"{va['loss']:.4f}",
                            acc=f"{tr['train_acc']:.4f}", val_acc=f"{va['acc']:.4f}",
                            val_ver=f"{va['verified']:.4f}", best=f"{best:.4f}")

        if stop_after is not None and end < c["epochs"]:
            return pd.DataFrame(rows)
        if os.path.exists(run.best):
            model.load_state_dict(torch.load(run.best, map_location=next(model.parameters()).device,
                                             weights_only=False)["model"])
        model.eval()
        return pd.DataFrame(rows)


def load_best(model: nn.Module, run_path: str, device=None) -> nn.Module:
    """Load checkpoints/best.pt (or last.pt) of a finished run into `model`."""
    ck_dir = os.path.join(run_path, "checkpoints")
    path = os.path.join(ck_dir, "best.pt")
    if not os.path.exists(path):
        path = os.path.join(ck_dir, "last.pt")
    model.load_state_dict(torch.load(path, map_location=device or "cpu", weights_only=False)["model"])
    return model.eval()
