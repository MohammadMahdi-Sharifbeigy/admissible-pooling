# Admissible Pooling

> **Which Admissible Orders Make Interval Max-Pooling Inclusion-Preserving?**  
> A Characterization for Interval-Valued Convolutional Networks  
> *Akbar Dehghan Nezhad · Mohammadmahdi SharifBeygi*  
> Iran University of Science and Technology — submitted to **Fuzzy Sets and Systems**

---

## Overview

This repository accompanies the paper that characterises exactly which
*admissible orders* on intervals yield **sound** (inclusion-preserving) pooling
layers for interval-valued convolutional networks trained with
*Interval Bound Propagation* (IBP).

**Key result.** Among all admissible orders, the unique sound choice for
max-pooling is the **Lex2** order (compare upper endpoints first, break ties by
lower endpoint). Dually, only **Lex1** is sound for min-pooling. The
**Xu–Yager** order is sound for neither. The lattice (endpoint-wise) maximum is
always sound but is not induced by any admissible order.

The code implements four 2-D interval pooling operators, certified-training
pipelines for image CNNs (MNIST, Fashion-MNIST, Kuzushiji-MNIST, CIFAR-10) and
graph neural networks (Cora, CiteSeer, PubMed, MUTAG), and all analysis /
visualisation utilities used in the paper.

---

## Repository Structure

```
admissible-pooling/
├── interval_gnn/           # Core library (PyTorch)
│   ├── pooling.py          # ★ interval_pool / IntervalPool2d  (Lattice, Lex2, Lex1, Xu–Yager)
│   ├── pooled_cnn.py       # PooledCNN7 – CNN-7 with pluggable interval pooling
│   ├── cnn_layers.py       # IntervalConv2d, IntervalLinear, interval_relu/bn/avgpool …
│   ├── cnn_models.py       # IntervalCNN / BaselineCNN (small & cnn7 architectures)
│   ├── layers.py           # GNN interval layers (IntervalConvLayer, IntervalGINConv)
│   ├── models.py           # IntervalGCN / IntervalGIN / baseline variants
│   ├── losses.py           # ibp_loss, composite_loss, eps_kappa_schedule …
│   ├── train.py            # Trainer, ImageTask, GraphTask, RunDir
│   ├── experiments.py      # Build tasks/models from JSON configs; run_experiment
│   ├── data.py             # GPUImageLoader, dataset_to_tensors
│   ├── interpret.py        # Certified-radius maps, saliency, AUROC, width analysis
│   ├── regularizers.py     # Lipschitz and mean-radius penalties
│   ├── envelope.py         # Enclosure / violation diagnostics
│   └── utils.py            # RadiusTracker, plotting helpers, set_seed
├── configs/
│   ├── make_configs.py     # Generates all experiment JSON files
│   ├── A/                  # Draft A: image classification (MNIST, FMNIST, CIFAR-10)
│   ├── B/                  # Draft B: controlled ablations
│   └── C/                  # Draft C: graph node / graph classification
├── scripts/
│   ├── run.py              # Run one experiment from a config file
│   ├── sweep.py            # Sweep over multiple configs
│   ├── swap_eval.py        # Swap-eval: test unsound operators on trained weights
│   └── collect.py          # Aggregate results into a summary CSV
├── doc/
│   ├── main.tex            # Paper source (Elsevier elsarticle)
│   ├── refs.bib            # Bibliography
│   └── figures/            # PDF figures used in the paper
├── train.ipynb             # Interactive training & analysis notebook
├── analysis.ipynb          # Results analysis & figure generation notebook
├── requirements.txt
├── CITATION.cff
└── LICENSE                 # MIT
```

---

## Pooling Operators

| Operator | Key | Sound? | Description |
|---|---|---|---|
| **Lattice** | `lattice` | ✅ Yes | `[max lo, max hi]` — exact range of the pointwise max |
| **Lex2** | `lex2` | ✅ Yes | Admissible max for the order *upper endpoint first* |
| **Lex1** | `lex1` | ❌ No | Admissible max for *lower endpoint first* (may under-cover) |
| **Xu–Yager** | `xuyager` | ❌ No | Admissible max by midpoint (ties by upper) |

> [!NOTE]
> With `r = 0` all four operators reduce exactly to `F.max_pool2d`.
> The **clean accuracy** of a trained model is therefore independent of which
> operator was used during training.

> [!IMPORTANT]
> Only `lattice` and `lex2` are **sound** (inclusion-preserving). Use them
> whenever the network is intended to propagate certified bounds.

---

## Quick Start

### Installation

```bash
git clone https://github.com/MohammadMahdi-Sharifbeigy/admissible-pooling.git
cd admissible-pooling
pip install -r requirements.txt
# For graph experiments (Draft C) also install torch_geometric:
pip install torch_geometric
```

### Using the pooling layer

```python
import torch
from interval_gnn import IntervalPool2d, interval_pool, POOL_OPS, SOUND_OPS

# Center-radius intervals: c ± r represents [c - r, c + r]
c = torch.randn(4, 32, 16, 16)   # batch × channels × H × W
r = torch.rand_like(c) * 0.1     # non-negative radii

# Functional API
c_out, r_out = interval_pool(c, r, op="lex2", k=2)

# Module API (op can be swapped at any time, e.g. for swap-eval)
pool = IntervalPool2d(op="lattice", k=2)
c_out, r_out = pool(c, r)

print(POOL_OPS)   # ('lattice', 'lex2', 'lex1', 'xuyager')
print(SOUND_OPS)  # ('lattice', 'lex2')
```

### Training a certified model

```bash
# Generate all experiment configs
python configs/make_configs.py

# Train one config (Draft A, Lex2, one pooling layer, MNIST ε = 0.3)
python scripts/run.py --config configs/A/mnist_lex2_pool1_eps03.json --seed 0

# Swap unsound operators onto already-trained weights and evaluate
python scripts/swap_eval.py --run runs/A/mnist/lex2_pool1_eps03/seed0

# Aggregate all results
python scripts/collect.py --draft A --out results_A.csv
```

### Notebooks

| Notebook | Purpose |
|---|---|
| [`train.ipynb`](train.ipynb) | End-to-end training, IBP schedule, enclosure violation monitoring |
| [`analysis.ipynb`](analysis.ipynb) | Paper figures: certified-radius maps, width analysis, result tables |

---

## Reproducing Paper Results

All experiments in the paper use the JSON configs under `configs/A/` (image) and
`configs/C/` (graph). The reference CNN-7 architecture follows
[Shi et al. (2021)](https://arxiv.org/abs/2103.17268); the IBP training recipe
(smoothed epsilon schedule, `kappa` annealing, IBP initialisation) follows the
same work and [CTBench (Mao et al. 2024)](https://arxiv.org/abs/2406.04848).

```bash
# Full Draft A sweep (requires GPU; each run ~1–3 h on a single GPU)
python scripts/sweep.py --draft A --seeds 0 1 2
```

Pre-trained checkpoints for the CIFAR-10 Lex2 and Lattice models are stored in
`CIFAR/` (excluded from git via `.gitignore`; available on request).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{admissible_pooling2026,
  author  = {Dehghan Nezhad, Akbar and SharifBeygi, Mohammadmahdi},
  title   = {Which Admissible Orders Make Interval Max-Pooling Inclusion-Preserving?
             A Characterization for Interval-Valued Convolutional Networks},
  journal = {Fuzzy Sets and Systems},
  year    = {2026}
}
```

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

### Related Works

The implementation builds on the following key references (see also
[`doc/refs.bib`](doc/refs.bib)):

| Reference | Role |
|---|---|
| Gowal et al. (ICCV 2019) | IBP training framework |
| Shi et al. (NeurIPS 2021) | CNN-7 architecture & smoothed-eps schedule |
| Zhang et al. (ICLR 2020) | CROWN-IBP |
| Mao et al. (ICLR 2024) | Understanding certified training with IBP |
| Mao et al. (2024) — CTBench | Benchmark methodology |
| Bustince et al. (FSS 2013) | Admissible orders & aggregation on intervals |
| Xu & Yager (2006) | Xu–Yager order on intuitionistic fuzzy sets |
| Moore et al. (2009) | Introduction to Interval Analysis |
| Rodríguez-Martínez et al. (NN 2022, IF 2023) | Pooling via aggregation functions in CNNs |

---

## License

This project is released under the [MIT License](LICENSE).

© 2026 Akbar Dehghan Nezhad & Mohammadmahdi SharifBeygi,
Iran University of Science and Technology.
