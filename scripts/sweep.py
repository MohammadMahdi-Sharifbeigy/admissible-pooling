"""
Run several configs x seeds one after the other (finished runs are skipped).

    python scripts/sweep.py --configs configs/A/fmnist_*_pool1.json --seeds 0 1 2
    python scripts/sweep.py --configs configs/C/*.json --seeds 0 1 2 3 4 --device cpu

A failing run is reported and the sweep continues with the next one.
"""
import argparse
import glob
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interval_gnn.experiments import ROOT, deep_update, load_config, run_experiment  # noqa: E402
from run import parse_set  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(argv)
    paths = sorted({p for pat in a.configs for p in glob.glob(pat)})
    jobs = [(p, s) for p in paths for s in a.seeds]
    print(f"{len(jobs)} runs: {len(paths)} configs x {len(a.seeds)} seeds")
    failed = []
    for i, (p, s) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {os.path.basename(p)} seed {s}  ({time.strftime('%H:%M:%S')})")
        try:
            cfg = deep_update(load_config(p), parse_set(a.set))
            run_experiment(cfg, s, root=a.root, device=a.device)
        except Exception:
            traceback.print_exc()
            failed.append((p, s))
    print(f"\nfinished; {len(failed)} failed" + "".join(f"\n  {p} seed {s}" for p, s in failed))


if __name__ == "__main__":
    main()
