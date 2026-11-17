"""
interval_gnn
============
Center-Radius Interval-Valued GNN (CR-INN) pipeline.
"""
from .layers import IntervalConvLayer, IntervalGINConv, propagation_matrix, propagation_inf_norm
from .models import IntervalGCN, IntervalGIN, BaselineGCN, BaselineGIN
from .cnn_layers import (IntervalConv2d, IntervalLinear, IntervalConv2dLayer, Transition2dLayer,
                         interval_relu, interval_maxpool, interval_avgpool, interval_bn, input_box, kappa, ibp_init_)
from .cnn_models import IntervalCNN, BaselineCNN
from .pooling import IntervalPool2d, interval_pool, POOL_OPS, SOUND_OPS
from .pooled_cnn import PooledCNN7
from .losses import composite_loss, ibp_loss, certified, eps_kappa_schedule, worst_case_logits
from .interpret import (logit_intervals, worst_margin, plot_logit_intervals, certified_radius,
                        plot_certified_radius, radius_maps, plot_radius_maps, unstable_fraction,
                        interval_saliency, gradient_saliency, plot_saliency, auroc,
                        width_uncertainty, plot_width_vs_error)
from .data import GPUImageLoader, dataset_to_tensors
from .utils import RadiusTracker, compute_rho, verify_contraction, verify_width_bound, verify_graph_width_bound, plot_radius_evol, plot_train_curves, summarize_results, set_seed

__all__ = [
    "logit_intervals", "worst_margin", "plot_logit_intervals", "certified_radius",
    "plot_certified_radius", "radius_maps", "plot_radius_maps", "unstable_fraction",
    "interval_saliency", "gradient_saliency", "plot_saliency", "auroc",
    "width_uncertainty", "plot_width_vs_error",
    "IntervalConvLayer", "IntervalGINConv", "propagation_matrix", "propagation_inf_norm",
    "IntervalGCN", "IntervalGIN", "BaselineGCN", "BaselineGIN",
    "IntervalConv2d", "IntervalLinear", "IntervalConv2dLayer", "Transition2dLayer",
    "interval_relu", "interval_maxpool", "interval_avgpool", "interval_bn", "input_box", "kappa", "ibp_init_",
    "IntervalCNN", "BaselineCNN",
    "IntervalPool2d", "interval_pool", "POOL_OPS", "SOUND_OPS", "PooledCNN7",
    "composite_loss", "ibp_loss", "certified", "eps_kappa_schedule", "worst_case_logits",
    "GPUImageLoader", "dataset_to_tensors",
    "RadiusTracker", "compute_rho", "verify_contraction", "verify_width_bound", "verify_graph_width_bound", "plot_radius_evol", "plot_train_curves", "summarize_results", "set_seed",
]