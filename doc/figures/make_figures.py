"""
Data figures of the paper, drawn at their printed size (text width 390 pt = 5.4 in).

    python figures/make_figures.py                 # runs are looked up next to the paper folder
    python figures/make_figures.py --runs <dir>    # <dir> contains A/<dataset>/<name>/seed<k>/

Writes figures/Figure_3.pdf (training curves), Figure_4.pdf (single-window simulation) and
Figure_5.pdf (excess width of Lex2). Figures 1 and 2 are TikZ drawings (Figure_1.tex, Figure_2.tex).
Text is typeset by LaTeX in Latin Modern, the font of the manuscript, and embedded in the PDF.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
WIDTH = 5.4                                    # inches; \textwidth of elsarticle (preprint) is 390 pt

# one colour per pooling operator, the same in every figure (checked for colour-vision deficiencies)
C = {"lattice": "#2a78d6", "lex2": "#eb6834", "lex1": "#1baf7a", "xuyager": "#b5438f"}
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
    "font.family": "serif", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 600,
})

BENCHMARKS = [("mnist", "eps01", r"MNIST, $\varepsilon=0.1$"), ("mnist", "eps03", r"MNIST, $\varepsilon=0.3$"),
              ("cifar10", "eps2", r"CIFAR-10, $\varepsilon=2/255$"), ("cifar10", "eps8", r"CIFAR-10, $\varepsilon=8/255$")]


def find_runs(arg):
    for root in ([arg] if arg else [os.path.join(os.path.dirname(PAPER), "runs"), os.path.dirname(PAPER)]):
        if os.path.isdir(os.path.join(root, "A")):
            return root
    raise SystemExit("no run directory found; pass --runs <dir containing A/>")


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=200)      # preview only, not for submission
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------------------------
def figure_3(runs):
    """Training curves of the pooled CNN-7 with one pooling layer: lattice maximum against Lex2."""
    fig, axes = plt.subplots(4, 2, figsize=(WIDTH, 6.5))
    styles = {"lattice": dict(color=C["lattice"], ls="-"), "lex2": dict(color=C["lex2"], ls=(0, (4, 1.6)))}
    for row, (ds, tag, label) in enumerate(BENCHMARKS):
        ax_v, ax_g = axes[row]
        ramp_end = None
        for op in ("lattice", "lex2"):
            files = sorted(glob.glob(os.path.join(runs, "A", ds, f"{op}_pool1_{tag}", "seed*", "metrics.csv")))
            m = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(files)])
            ramp_end = int(m[m.eps >= m.eps.max() * (1 - 1e-9)].epoch.min())
            for ax, col, scale in ((ax_v, "val_verified", 100.0), (ax_g, "grad_norm", 1.0)):
                g = m.groupby("epoch")[col]
                mean, lo, hi = g.mean() * scale, g.min() * scale, g.max() * scale
                ax.fill_between(mean.index, lo, hi, color=styles[op]["color"], alpha=0.22, lw=0)
                ax.plot(mean.index, mean, **styles[op])
        last = int(m.epoch.max())
        for ax in (ax_v, ax_g):
            ax.axvline(ramp_end, color=MUTED, lw=0.6, ls=":")
            ax.set_xlim(0, last)
            ax.set_xticks(np.arange(0, last + 1, 10 if last <= 80 else 40))
        ax_g.set_yscale("log")
        # limits on 1-2-5 steps around the data; label those steps within about a decade, decades only otherwise
        lo, hi = ax_g.dataLim.y0, ax_g.dataLim.y1
        steps = np.array([m * 10.0 ** e for e in range(-3, 4) for m in (1, 2, 5)])
        lo, hi = steps[steps <= lo].max(), steps[steps >= hi].min()
        ticks = steps[(steps >= lo) & (steps <= hi)]
        if len(ticks) > 5:
            ticks = np.array([t for t in ticks if abs(np.log10(t) - round(np.log10(t))) < 1e-9])
        ax_g.set_ylim(lo, hi)
        ax_g.yaxis.set_major_locator(FixedLocator(ticks))
        ax_g.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax_g.yaxis.set_minor_locator(NullLocator())
        ax_g.yaxis.set_minor_formatter(NullFormatter())
        ax_v.set_ylim(0, None)
        ax_v.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax_v.set_ylabel(r"Verified acc.\ (\%)")
        ax_g.set_ylabel("Gradient norm")
        ax_v.set_title(r"\textbf{(%s)} %s" % ("abcd"[row], label), loc="left", pad=5)
        if row == 0:
            ax_v.annotate(r"end of $\varepsilon$ ramp", xy=(ramp_end, 30), xytext=(ramp_end + 6, 30),
                          fontsize=9, color=MUTED, va="center",
                          arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=1))
        if row in (1, 3):
            ax_v.set_xlabel("Epoch"); ax_g.set_xlabel("Epoch")
    handles = [Line2D([], [], label="Lattice maximum", **styles["lattice"]), Line2D([], [], label="Lex2", **styles["lex2"]),
               Patch(facecolor=MUTED, alpha=0.3, lw=0, label="range over three seeds")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.9, w_pad=1.6)
    save(fig, "Figure_3.pdf")


# ---------------------------------------------------------------------------------------------
def figure_4():
    """Simulated 2x2 windows: predicted against sampled violation probability, and the excess width of Lex2."""
    df = pd.read_csv(os.path.join(PAPER, "results", "violation_sim.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1.3]), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.03)
    for ax, op, name, tag in ((axes[0], "lex1", "Lex1", "a"), (axes[1], "xuyager", "Xu--Yager", "b")):
        d = df[df.op == op].dropna(subset=["predicted"]).sample(frac=1, random_state=0).head(3000)
        ax.plot([0, 1], [0, 1], color=INK, lw=0.6, zorder=1)
        ax.scatter(d.predicted, d.empirical, s=5, color=C[op], alpha=0.35, lw=0, zorder=2, rasterized=True)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal", xlabel="Predicted probability")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.grid(False)
        ax.set_title(r"\textbf{(%s)} %s" % (tag, name), loc="left", pad=5)
    axes[0].set_ylabel("Sampled frequency")
    axes[1].set_yticklabels([])
    lex2 = df[(df.op == "lex2") & (df.width > 0)]
    rel = (lex2.excess / lex2.width).clip(0, 1)
    ax = axes[2]
    ax.hist(rel, bins=np.linspace(0, 1, 21), weights=np.full(len(rel), 100.0 / len(rel)), color=C["lex2"],
            edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 130)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlim=(0, 1), xlabel=r"Relative excess $e/w$", ylabel=r"Windows (\%)")
    ax.set_xticks([0, 0.5, 1])
    ax.set_title(r"\textbf{(c)} Lex2", loc="left", pad=5)
    zero = 100.0 * float((rel < 0.05).mean())
    ax.annotate(r"$%.0f\,\%%$ below $0.05$" % zero, xy=(0.06, zero), xytext=(0.2, zero * 0.75), fontsize=9, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=1))
    save(fig, "Figure_4.pdf")


# ---------------------------------------------------------------------------------------------
def figure_5(runs):
    """Excess width of Lex2 over the lattice maximum on lattice-trained weights, one marker per network."""
    rows = []
    for f in glob.glob(os.path.join(runs, "A", "*", "*", "seed*", "swap_results.csv")):
        parts = os.path.normpath(f).split(os.sep)
        d = pd.read_csv(f)
        d = d[(d.trained_with == "lattice") & (d.op == "lex2")]
        name = parts[-3].split("_")
        for r in d.itertuples():
            rows.append(dict(dataset=parts[-4], tag=(name + ["old"])[2], layout=name[1],
                             excess=100.0 * (r.width_ratio - 1.0)))
    d = pd.DataFrame(rows)
    benches = [(ds, tag, lab.replace(", ", "\n")) for ds, tag, lab in BENCHMARKS] + \
              [("fmnist", "old", "FMNIST$^\\dagger$\n$\\varepsilon=0.1$")]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    marks = {"pool1": dict(dx=-0.17, mfc="white", label="One pooling layer"),
             "pool2": dict(dx=0.17, mfc=C["lex2"], label="Two pooling layers")}
    for i, (ds, tag, _) in enumerate(benches):
        for layout, mk in marks.items():
            v = np.sort(d[(d.dataset == ds) & (d.tag == tag) & (d.layout == layout)].excess.to_numpy())
            if len(v) == 0:
                continue
            x = i + mk["dx"]
            ax.plot(x + np.linspace(-0.055, 0.055, len(v)), v, ls="none", marker="o", ms=5.5, mfc=mk["mfc"],
                    mec=C["lex2"], mew=1.2, zorder=3)
            ax.plot([x - 0.13, x + 0.13], [v.mean()] * 2, color=INK, lw=1.2, zorder=2, solid_capstyle="butt")
            ax.annotate(f"{v.mean():.1f}", (x, v.max()), xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 200)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(len(benches)), [b[2] for b in benches])
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.55, len(benches) - 0.45)
    ax.set_ylabel(r"Excess width of Lex2 (\%)")
    handles = [Line2D([], [], ls="none", marker="o", ms=5.5, mfc=m["mfc"], mec=C["lex2"], mew=1.2, label=m["label"])
               for m in marks.values()] + [Line2D([], [], color=INK, lw=1.2, label="mean of three seeds")]
    ax.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.14), columnspacing=1.4,
              handletextpad=0.5, borderaxespad=0)
    fig.tight_layout()
    save(fig, "Figure_5.pdf")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="directory that contains A/<dataset>/<name>/seed<k>/")
    a = ap.parse_args()
    runs = find_runs(a.runs)
    figure_3(runs)
    figure_4()
    figure_5(runs)
