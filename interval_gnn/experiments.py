"""
Build tasks and models from a JSON config and run one experiment end to end.

Config (configs/<draft>/*.json):

    {
      "draft": "A", "dataset": "fmnist", "name": "lex2_pool1",
      "model": {"type": "pooled_cnn7", "layout": "pool1", "pool": "lex2"},
      "init": "ibp",                                   # optional: IBP initialization (Shi et al. 2021)
      "data": {"batch_size": 256, "augment": false},    # images (also: norm, pad, pad_mode, full_train)
                                                        # graphs: {"split": {...}}
      "train": {"epochs": 100, "eps": 0.1, ...},        # Trainer cfg (interval_gnn.train.DEFAULTS)
      "eps_eval": [0.0, 0.025, 0.05, 0.1]
    }

Datasets: mnist, fmnist, kmnist, cifar10 (pixel eps; data.norm picks the input normalization);
cora, citeseer, pubmed (Planetoid splits); mutag (stratified split or k-fold, see _graph_split).
Model types: cnn7, baseline_cnn7, pooled_cnn7, gcn (IntervalGCN), baseline_gcn, gin (IntervalGIN), baseline_gin.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from interval_gnn.train import GraphTask, ImageTask, RunDir, Trainer, set_all_seeds, test_table

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

IMAGE_SETS = {
    "mnist": ("MNIST", "MNIST", 1, 28),
    "fmnist": ("FashionMNIST", "FashionMNIST", 1, 28),
    "kmnist": ("KMNIST", "KMNIST", 1, 28),
    "cifar10": ("CIFAR10", "CIFAR10", 3, 32),
}
NORMS = {                                     # (mean, std) per channel
    "half": ((0.5,), (0.5,)),                 # the notebooks and the FMNIST / KMNIST runs
    "none": ((0.0,), (1.0,)),                 # MNIST (Shi et al. 2021)
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # CIFAR-10 (Shi et al. 2021)
}
NODE_SETS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
_CACHE: dict = {}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _image_tensors(name: str):
    if name not in _CACHE:
        import torchvision
        from interval_gnn.data import dataset_to_tensors
        cls_name, folder, _, _ = IMAGE_SETS[name]
        cls = getattr(torchvision.datasets, cls_name)
        tr = cls(root=os.path.join(DATA, folder), train=True, download=True)
        te = cls(root=os.path.join(DATA, folder), train=False, download=True)
        _CACHE[name] = (*dataset_to_tensors(tr), *dataset_to_tensors(te))
    return _CACHE[name]


def image_task(name: str, data_cfg: dict, seed: int, device) -> ImageTask:
    """
    10% of the training set (fixed permutation, seed 0, as in the notebooks) is held out for validation.
    data.full_train = true trains on the whole original split, as the published benchmarks do; the same
    10% then only monitors training, so such runs must not select on it (train.select = 'last').
    """
    from interval_gnn.data import GPUImageLoader
    x_all, y_all, x_te, y_te = _image_tensors(name)
    n_sub = data_cfg.get("subset")                       # for smoke tests
    n_val = len(y_all) // 10
    perm = torch.randperm(len(y_all), generator=torch.Generator().manual_seed(0))
    val_idx, tr_idx = perm[:n_val], perm if data_cfg.get("full_train") else perm[n_val:]
    if n_sub:
        tr_idx, val_idx = tr_idx[:n_sub], val_idx[: max(n_sub // 4, 64)]
        x_te, y_te = x_te[: max(n_sub // 4, 64)], y_te[: max(n_sub // 4, 64)]
    c = x_all.size(1)
    mean, std = NORMS[data_cfg.get("norm", "half")]
    mean, std = mean * (c // len(mean)), std * (c // len(std))
    norm = dict(mean=mean, std=std, device=device)
    bs = data_cfg.get("batch_size", 256)
    loaders = dict(
        train=GPUImageLoader(x_all[tr_idx], y_all[tr_idx], bs, shuffle=True, augment=data_cfg.get("augment", False),
                             pad=data_cfg.get("pad", 4), flip=data_cfg.get("flip", True),
                             pad_mode=data_cfg.get("pad_mode", "constant"), seed=seed, **norm),
        val=GPUImageLoader(x_all[val_idx], y_all[val_idx], 512, **norm),
        test=GPUImageLoader(x_te, y_te, 512, **norm),
    )
    return ImageTask(loaders, std=std, mean=mean, device=device)


def _stratified(labels: np.ndarray, fracs, seed: int):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_tr, n_va = int(round(fracs[0] * len(idx))), int(round(fracs[1] * len(idx)))
        for p, chunk in zip(parts, np.split(idx, [n_tr, n_tr + n_va])):
            p.extend(chunk.tolist())
    return [sorted(p) for p in parts]


def _kfold(labels: np.ndarray, k: int, fold: int, seed: int):
    """Stratified k-fold: fold `fold` is the test set, the next fold the validation set, the rest training."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk.tolist())
    te, va = folds[fold % k], folds[(fold + 1) % k]
    tr = [i for j, f in enumerate(folds) if j not in (fold % k, (fold + 1) % k) for i in f]
    return sorted(tr), sorted(va), sorted(te)


def graph_task(name: str, data_cfg: dict, device) -> GraphTask:
    if name in NODE_SETS:
        key = ("node", name)
        if key not in _CACHE:
            from torch_geometric.datasets import Planetoid
            _CACHE[key] = Planetoid(root=DATA, name=NODE_SETS[name])[0]
        return GraphTask(data=_CACHE[key].to(device))
    if name == "mutag":
        from torch_geometric.datasets import TUDataset
        from torch_geometric.loader import DataLoader
        if "mutag" not in _CACHE:
            _CACHE["mutag"] = TUDataset(root=DATA, name="MUTAG")
        ds = _CACHE["mutag"]
        labels = np.array([d.y.item() for d in ds])
        sp = data_cfg.get("split", {"type": "stratified", "fracs": [0.7, 0.15, 0.15], "seed": 42})
        if sp["type"] == "kfold":
            tr, va, te = _kfold(labels, sp.get("k", 10), sp["fold"], sp.get("seed", 0))
        else:
            tr, va, te = _stratified(labels, sp["fracs"], sp.get("seed", 42))
        bs = data_cfg.get("batch_size", 32)
        loaders = dict(train=DataLoader([ds[i] for i in tr], batch_size=bs, shuffle=True),
                       val=DataLoader([ds[i] for i in va], batch_size=64),
                       test=DataLoader([ds[i] for i in te], batch_size=64))
        return GraphTask(loaders=loaders, device=device)
    raise ValueError(f"Unknown graph dataset {name!r}")


def build_task(cfg: dict, seed: int, device):
    name = cfg["dataset"]
    if name in IMAGE_SETS:
        return image_task(name, cfg.get("data", {}), seed, device)
    return graph_task(name, cfg.get("data", {}), device)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _dims(cfg: dict, task) -> tuple:
    name = cfg["dataset"]
    if name in IMAGE_SETS:
        _, _, ch, size = IMAGE_SETS[name]
        return ch, 10, size
    if isinstance(task, GraphTask) and task.node_level:
        d = task.data
        return d.num_node_features, int(d.y.max()) + 1, None
    ds = _CACHE["mutag"]
    return ds.num_node_features, ds.num_classes, None


def build_model(cfg: dict, task) -> torch.nn.Module:
    from interval_gnn.cnn_layers import ibp_init_
    from interval_gnn.cnn_models import BaselineCNN, IntervalCNN
    from interval_gnn.models import BaselineGCN, BaselineGIN, IntervalGCN, IntervalGIN
    from interval_gnn.pooled_cnn import PooledCNN7
    m = dict(cfg["model"])
    t = m.pop("type")
    c_in, n_out, size = _dims(cfg, task)
    if t == "pooled_cnn7":
        model = PooledCNN7(c_in, n_out, input_size=size, **m)
    elif t == "cnn7":
        model = IntervalCNN(c_in, 64, n_out, arch="cnn7", input_size=size)
    elif t == "baseline_cnn7":
        model = BaselineCNN(c_in, 64, n_out, arch="cnn7", input_size=size)
    elif t == "gcn":
        model = IntervalGCN(c_in, m.get("hidden", 64), n_out, n_layers=m.get("n_layers", 2),
                            dropout=m.get("dropout", 0.5), agg=m.get("agg", "gcn"))
    elif t == "baseline_gcn":
        model = BaselineGCN(c_in, m.get("hidden", 64), n_out, n_layers=m.get("n_layers", 2),
                            dropout=m.get("dropout", 0.5))
    elif t == "gin":
        model = IntervalGIN(c_in, m.get("hidden", 32), n_out, n_layers=m.get("n_layers", 3),
                            dropout=m.get("dropout", 0.5))
    elif t == "baseline_gin":
        model = BaselineGIN(c_in, m.get("hidden", 32), n_out, n_layers=m.get("n_layers", 3),
                            dropout=m.get("dropout", 0.5))
    else:
        raise ValueError(f"Unknown model type {t!r}")
    if cfg.get("init") in ("ibp", "ibp_shi"):
        ibp_init_(model, as_in_paper_code=cfg["init"] == "ibp_shi")
    return model


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------

def run_dir(cfg: dict, seed: int, root: str) -> RunDir:
    return RunDir(root, cfg["draft"], cfg["dataset"], cfg["name"], seed)


def run_experiment(cfg: dict, seed: int, root: str = os.path.join(ROOT, "runs"), device=None,
                   force: bool = False, extra_loss=None) -> dict:
    """Train (or resume), evaluate on the test split and write test_results.csv + done.flag."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = run_dir(cfg, seed, root)
    if run.done and not force:
        print(f"skip (done): {run.path}")
        return dict(path=run.path, test=pd.read_csv(run.file("test_results.csv")),
                    history=pd.read_csv(run.file("metrics.csv")), model=None, task=None)
    set_all_seeds(seed)
    torch.backends.cudnn.benchmark = True          # input sizes are fixed
    task = build_task(cfg, seed, device)
    model = build_model(cfg, task).to(device)
    run.write_config(dict(cfg, seed=seed), device=str(device),
                     n_params=sum(p.numel() for p in model.parameters()))
    history = Trainer(model, task, cfg["train"], run, extra_loss=extra_loss).fit()
    test = test_table(model, task, cfg["eps_eval"], name=cfg["name"], margin=cfg["train"].get("margin", False))
    test.insert(0, "seed", seed)
    test.insert(0, "dataset", cfg["dataset"])
    test.to_csv(run.file("test_results.csv"), index=False)
    run.mark_done()
    return dict(path=run.path, test=test, history=history, model=model, task=task)


def load_run(path: str, device=None):
    """Rebuild task and model of a finished run from its config.json and load the best checkpoint."""
    from interval_gnn.train import load_best
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(os.path.join(path, "config.json"))
    cfg.pop("_meta", None)
    seed = cfg.get("seed", 0)
    task = build_task(cfg, seed, device)
    model = load_best(build_model(cfg, task).to(device), path, device)
    return cfg, task, model
