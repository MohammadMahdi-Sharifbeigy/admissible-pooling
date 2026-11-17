"""
Run one experiment from a JSON config.

    python scripts/run.py --config configs/A/fmnist_lex2_pool1.json --seed 0
    python scripts/run.py --config configs/C/cora_ibp.json --seed 0 --device cpu
    python scripts/run.py --config configs/A/fmnist_lex2_pool1.json --seed 0 --set train.epochs=2 data.subset=512

Resumes automatically from checkpoints/last.pt; skips runs that already have done.flag
(use --force to rerun the final evaluation).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interval_gnn.experiments import ROOT, deep_update, load_config, run_experiment  # noqa: E402


def parse_set(items):
    """['train.epochs=2', 'data.subset=512'] -> nested dict with JSON-parsed values."""
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        d = out
        *path, last = key.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", help="override config entries, e.g. train.epochs=2")
    a = ap.parse_args(argv)
    cfg = deep_update(load_config(a.config), parse_set(a.set))
    res = run_experiment(cfg, a.seed, root=a.root, device=a.device, force=a.force)
    print(res["test"].to_string(index=False))


if __name__ == "__main__":
    main()
