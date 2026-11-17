"""
Generate every experiment config of drafts A and C (draft B's numerics are in configs/B/*.json).

    python configs/make_configs.py

Edit the settings here, not the generated JSON files.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def write(sub, name, cfg):
    os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    with open(os.path.join(HERE, sub, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Draft A: the standard certified-training benchmarks, with the recipe of Shi et al. (2021)
# as used by CTBench: pure robust loss on margin bounds, smoothed eps schedule, warm-up
# regularizers, IBP initialization, full training split, final epoch, train eps = test eps.
#   ref   the unmodified CNN-7 (stride-2 convolution, no pooling), checks the recipe
#   lattice / lex2   pooled CNN-7 with IBP
#   std   pooled CNN-7, normally trained control
# Lex1 and Xu-Yager are unsound and are only swapped in at evaluation (scripts/A_swap_eval.py).
# ---------------------------------------------------------------------------
A_DATA = {
    ("mnist", "eps01"): dict(eps=0.1, epochs=70, warmup=0, ramp=20, ms=[50 / 70, 60 / 70],
                             data=dict(batch_size=256, augment=False, norm="none", full_train=True)),
    ("mnist", "eps03"): dict(eps=0.3, epochs=70, warmup=0, ramp=20, ms=[50 / 70, 60 / 70],
                             data=dict(batch_size=256, augment=False, norm="none", full_train=True)),
    ("cifar10", "eps2"): dict(eps=2 / 255, epochs=160, warmup=1, ramp=80, ms=[0.75, 0.875],
                              data=dict(batch_size=128, augment=True, pad=2, pad_mode="replicate",
                                        norm="cifar10", full_train=True)),
    ("cifar10", "eps8"): dict(eps=8 / 255, epochs=160, warmup=1, ramp=80, ms=[0.75, 0.875],
                              data=dict(batch_size=128, augment=True, pad=2, pad_mode="replicate",
                                        norm="cifar10", full_train=True)),
}
for (ds, tag), d in A_DATA.items():
    for op, layout in (("ref", "cnn7"), ("lattice", "pool1"), ("lex2", "pool1"), ("std", "pool1"),
                       ("lattice", "pool2"), ("lex2", "pool2")):
        ibp = op != "std"
        write("A", f"{ds}_{op}_{layout}_{tag}", dict(
            draft="A", dataset=ds, name=f"{op}_{layout}_{tag}",
            model=dict(type="pooled_cnn7", layout=layout, pool="lex2" if op == "lex2" else "lattice"),
            init="ibp_shi" if ibp else None, data=d["data"],
            train=dict(epochs=d["epochs"], lr=5e-4, weight_decay=0.0, scheduler="multistep",
                       milestones=d["ms"], gamma=0.2, grad_clip=10.0, ibp=ibp, eps=d["eps"],
                       warmup_epochs=d["warmup"], ramp_epochs=d["ramp"], kappa_end=0.0,
                       eps_schedule="smoothed", margin=True, reg_lambda=0.5 if ibp else 0.0, reg_tol=0.5,
                       select="last"),
            eps_eval=[0.0, d["eps"]],
        ))

# The finished FMNIST runs used an earlier recipe (mixed loss, logit intervals, linear ramp).
for op in ("lattice", "lex2", "std"):
    ibp = op != "std"
    write("A", f"fmnist_{op}_pool1", dict(
        draft="A", dataset="fmnist", name=f"{op}_pool1",
        model=dict(type="pooled_cnn7", layout="pool1", pool="lattice" if op == "std" else op),
        init="ibp" if ibp else None, data=dict(batch_size=256, augment=False),
        train=dict(epochs=100, lr=5e-4, weight_decay=0.0, scheduler="multistep", milestones=[0.75, 0.875],
                   gamma=0.2, grad_clip=10.0, ibp=ibp, eps=0.1, warmup_epochs=1, ramp_epochs=40, kappa_end=0.5),
        eps_eval=[0.0, 0.025, 0.05, 0.1],
    ))

# ---------------------------------------------------------------------------
# Draft C: node classification (Planetoid), depth/aggregation study, MUTAG 10-fold CV.
# ---------------------------------------------------------------------------
NODE_TRAIN = dict(epochs=300, lr=1e-2, weight_decay=5e-4, scheduler="cosine", grad_clip=None,
                  eps=0.01, warmup_epochs=20, ramp_epochs=100, kappa_end=0.5)
NODE_EPS = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]
for ds in ("cora", "citeseer", "pubmed"):
    write("C", f"{ds}_baseline", dict(draft="C", dataset=ds, name="baseline_gcn",
                                      model=dict(type="baseline_gcn", hidden=64, n_layers=2),
                                      train=dict(NODE_TRAIN, ibp=False), eps_eval=NODE_EPS))
    write("C", f"{ds}_ibp", dict(draft="C", dataset=ds, name="ibp_gcn",
                                 model=dict(type="gcn", hidden=64, n_layers=2, agg="gcn"),
                                 train=dict(NODE_TRAIN, ibp=True), eps_eval=NODE_EPS))

for agg in ("mean", "gcn", "sum"):
    for L in range(1, 6):
        write("C/depth", f"cora_{agg}_L{L}", dict(draft="C", dataset="cora", name=f"depth_{agg}_L{L}",
                                                  model=dict(type="gcn", hidden=64, n_layers=L, agg=agg),
                                                  train=dict(NODE_TRAIN, ibp=True), eps_eval=[0.0, 0.005, 0.01]))

GRAPH_TRAIN = dict(epochs=300, lr=1e-2, weight_decay=1e-4, scheduler="cosine", grad_clip=None,
                   eps=0.05, warmup_epochs=20, ramp_epochs=150, kappa_end=0.5)
for fold in range(10):
    split = dict(type="kfold", k=10, fold=fold, seed=0)
    for kind, mtype, ibp in (("baseline_gin", "baseline_gin", False), ("ibp_gin", "gin", True)):
        write("C/mutag", f"mutag_{kind}_fold{fold}", dict(
            draft="C", dataset="mutag", name=f"{kind}_fold{fold}",
            model=dict(type=mtype, hidden=32, n_layers=3), data=dict(batch_size=32, split=split),
            train=dict(GRAPH_TRAIN, ibp=ibp), eps_eval=[0.0, 0.0125, 0.025, 0.05, 0.1]))

print("configs written to", HERE)
