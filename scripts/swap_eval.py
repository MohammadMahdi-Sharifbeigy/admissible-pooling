"""
Draft A, experiment A-swap: evaluate every finished pooled-CNN run with all four pooling
operators on the same weights (hypotheses H1-H4).

For each operator:
  violation_rate   fraction of (input, logit) pairs whose sampled outputs leave the computed
                   bounds by more than 1e-5 (n_inputs inputs, n_samples uniform points of the
                   box plus n_corners random corners each)
  clean_acc        identical for all operators (r = 0 gives ordinary max-pooling)
  verified_acc     at the training eps, meaningful for lattice and lex2 only
  mean_width       mean width of the predicted logit's interval
  width_ratio      mean over inputs of width(op) / width(lattice)

    python scripts/A_swap_eval.py                       # every run under runs/A without swap_results.csv
    python scripts/A_swap_eval.py --run runs/A/fmnist/lex2_pool1/seed0 --force

Writes swap_results.csv into each run directory.
"""
import argparse
import glob
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interval_gnn.cnn_layers import input_box, interval_relu  # noqa: E402
from interval_gnn.experiments import ROOT, load_run  # noqa: E402
from interval_gnn.losses import certified, margin_certified  # noqa: E402
from interval_gnn.pooling import POOL_OPS  # noqa: E402


def _first_pool(model, x, box_c, box_r):
    """Interval output of the first pooling layer, and the real pre-pool activations at x."""
    c, r = box_c, box_r
    for i, layer in enumerate(model.convs):
        c, r = interval_relu(*layer.pre_activation(c, r))
        x = layer.clean(x)
        if str(i) in model.pools:
            pc, pr = model.pools[str(i)](c, r)
            return pc - pr, pc + pr, x
    raise ValueError("model has no pooling layer")


@torch.no_grad()
def swap_eval(model, task, eps: float, n_inputs: int = 500, n_samples: int = 100, n_corners: int = 20,
              batch: int = 25, margin: bool = False, window_tol: float = 1e-4) -> pd.DataFrame:
    """
    Besides the logit-level violation rate, `window_violation_rate` is the fraction of windows of
    the first pooling layer (over inputs, channels and positions) whose true maximum at some sampled
    point leaves the pooled interval by more than window_tol; this is the per-window quantity of
    Prop. prop:violation. In float32 the sound operators exceed their interval by up to about 5e-5
    (rounding; zero in float64 apart from Lex2's tie tolerance tau), while unsound selections exceed
    it by up to about 1, hence the default 1e-4. max_window_excess is the largest excess seen.
    """
    model.eval()
    e = eps * task.eps_scale
    xs, ys = [], []
    for (x,), y, _ in task.batches("test"):
        xs.append(x)
        ys.append(y)
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    rows, widths = [], {}
    for op in POOL_OPS:
        model.set_pool(op)
        viol = wviol = n_win = acc = ver = wmax = 0.0
        w_list = []
        for k in tqdm(range(0, len(y_all), 512), desc=f"  {op}: bounds", leave=False):
            x, y = x_all[k:k + 512], y_all[k:k + 512]
            out = model(x)
            fc, fr = model._features(*input_box(x, e, task.lo, task.hi))     # one pass for both bounds
            lc, lr = model.fc2(fc, fr)
            lo, hi = lc - lr, lc + lr
            acc += out.argmax(1).eq(y).sum().item()
            if margin:
                w, b = model.fc2.lin.weight, model.fc2.lin.bias
                dw, db = w[y][:, None, :] - w[None, :, :], b[y][:, None] - b[None, :]
                m = (dw @ fc[:, :, None]).squeeze(-1) + db - (dw.abs() @ fr[:, :, None]).squeeze(-1)
                ver += margin_certified(m, y).sum().item()
            else:
                ver += certified(lo, hi, y).sum().item()
            pred = out.argmax(1, keepdim=True)
            w_list.append((hi - lo).gather(1, pred).squeeze(1))
        widths[op] = torch.cat(w_list)
        n_v = min(n_inputs, len(y_all))
        for k in tqdm(range(0, n_v, batch), desc=f"  {op}: sampling", leave=False):
            x = x_all[k:min(k + batch, n_v)]
            lo, hi = model.bounds(x, e, task.lo, task.hi)
            box_c, box_r = input_box(x, e, task.lo, task.hi)
            box_lo, box_hi = box_c - box_r, box_c + box_r
            bad = torch.zeros_like(lo, dtype=torch.bool)
            wbad = None
            for s in range(n_samples + n_corners):
                u = torch.rand_like(x) if s < n_samples else (torch.rand_like(x) < 0.5).float()
                z = box_lo + u * (box_hi - box_lo)
                out = model(z)
                bad |= (out < lo - 1e-5) | (out > hi + 1e-5)
                plo, phi, act = _first_pool(model, z, box_c, box_r)
                t = F.max_pool2d(act, model.k)
                ex = torch.maximum(plo - t, t - phi)
                wmax = max(wmax, ex.max().item())
                wb = ex > window_tol
                wbad = wb if wbad is None else wbad | wb
            viol += bad.float().sum().item()
            wviol += wbad.float().sum().item()
            n_win += wbad.numel()
        rows.append(dict(op=op, eps=eps, clean_acc=acc / len(y_all), verified_acc=ver / len(y_all),
                         violation_rate=viol / (n_v * lo.size(1)), window_violation_rate=wviol / n_win,
                         max_window_excess=max(wmax, 0.0),
                         mean_width=widths[op].mean().item(), n_inputs_sampled=n_v,
                         n_points=n_samples + n_corners))
    for r in rows:
        r["width_ratio"] = (widths[r["op"]] / widths["lattice"].clamp_min(1e-12)).mean().item()
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", help="run directories (default: all finished runs under runs/A)")
    ap.add_argument("--root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--n-inputs", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-corners", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    runs = a.run or sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.root, "A", "*", "*", "seed*", "done.flag")))
    for path in runs:
        out = os.path.join(path, "swap_results.csv")
        if os.path.exists(out) and not a.force:
            print("skip (exists):", out)
            continue
        cfg, task, model = load_run(path, a.device)
        if cfg["model"]["type"] != "pooled_cnn7" or not model.pools:     # the cnn7 reference has no pooling
            continue
        print(path)
        df = swap_eval(model, task, cfg["train"]["eps"], a.n_inputs, a.n_samples, a.n_corners,
                       margin=cfg["train"].get("margin", False))
        df.insert(0, "trained_with", cfg["model"]["pool"] if cfg["train"]["ibp"] else "std")
        df.to_csv(out, index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
