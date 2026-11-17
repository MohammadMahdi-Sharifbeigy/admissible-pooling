"""
Numerics for draft B: envelopes of real trigonometric polynomials on the circle.

A polynomial h in T_N is a coefficient vector [a0, a1..aN, b1..bN]:
    h(x) = a0 + sum_k a_k cos(kx) + b_k sin(kx).
Values live on the uniform grid x_g = g * delta, delta = 2*pi/G.

    L_eps h(x) = min_{|s|<=eps} h(x+s),   U_eps h(x) = max_{|s|<=eps} h(x+s)

are computed exactly on the grid restricted to grid shifts (scipy's O(G) sliding filters,
periodic boundary), i.e. the discrete envelopes L^delta, U^delta of Prop. prop:disc, with
half-width w = round(eps/delta) grid steps; the effective radius w*delta is returned.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def n_coef(N: int) -> int:
    return 2 * N + 1


def basis(N: int, G: int) -> np.ndarray:
    """[G, 2N+1] matrix B with h_grid = B @ coef."""
    x = 2 * np.pi * np.arange(G) / G
    k = np.arange(1, N + 1)
    return np.concatenate([np.ones((G, 1)), np.cos(np.outer(x, k)), np.sin(np.outer(x, k))], axis=1)


def half_width(eps: float, G: int) -> int:
    return int(round(eps * G / (2 * np.pi)))


def envelopes(h: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete L and U of grid values h (last axis), window of 2w+1 grid points, periodic."""
    if w <= 0:
        return h, h
    size = 2 * w + 1
    if size >= h.shape[-1]:
        lo = h.min(axis=-1, keepdims=True)
        hi = h.max(axis=-1, keepdims=True)
        return np.broadcast_to(lo, h.shape), np.broadcast_to(hi, h.shape)
    return (minimum_filter1d(h, size, axis=-1, mode="wrap"),
            maximum_filter1d(h, size, axis=-1, mode="wrap"))


def env_dist(h1: np.ndarray, h2: np.ndarray, w: int) -> float:
    """d(Phi(h1), Phi(h2)) = max(||L h1 - L h2||_inf, ||U h1 - U h2||_inf) on the grid."""
    l1, u1 = envelopes(h1, w)
    l2, u2 = envelopes(h2, w)
    return float(max(np.abs(l1 - l2).max(), np.abs(u1 - u2).max()))


def disc_error(h1: np.ndarray, h2: np.ndarray, N: int, G: int) -> float:
    """Prop. prop:disc: |d_grid - d| <= N (||h1|| + ||h2||) delta / 2 at grid points."""
    return 0.5 * N * (np.abs(h1).max() + np.abs(h2).max()) * 2 * np.pi / G


def sup_upper(q_grid: np.ndarray, N: int, G: int) -> float:
    """Upper bound on sup|q| from grid values: sup|q| <= max_grid|q| / cos(N delta / 2) (Lemma lem:cos)."""
    return float(np.abs(q_grid).max() / math.cos(N * np.pi / G))


def ratio(coef: np.ndarray, B: np.ndarray, w: int) -> float:
    """d(Phi h1, Phi h2) / ||h1 - h2||_inf for the stacked coefficient vector [c1, c2] (grid values)."""
    m = B.shape[1]
    h1, h2 = B @ coef[:m], B @ coef[m:]
    q = np.abs(h1 - h2).max()
    if q < 1e-12:
        return np.inf
    return env_dist(h1, h2, w) / q


def to_grid(coef: np.ndarray, N: int, G: int) -> np.ndarray:
    return basis(N, G) @ coef


def fit_T_N(values: np.ndarray, N: int) -> np.ndarray:
    """Least-squares coefficients in T_N of grid values (exact projection for a uniform grid)."""
    G = values.shape[-1]
    return np.linalg.lstsq(basis(N, G), values, rcond=None)[0]


def periodic_convolve(f_coef: np.ndarray, g_coef: np.ndarray, N: int) -> np.ndarray:
    """Coefficients of f (*) g = (1/2pi) int f(x-t) g(t) dt for f, g in T_N (hat h = hat f * hat g)."""
    a_f, c_f, s_f = f_coef[0], f_coef[1:N + 1], f_coef[N + 1:]
    a_g, c_g, s_g = g_coef[0], g_coef[1:N + 1], g_coef[N + 1:]
    # complex coefficients for k >= 1: hat u(k) = (a_k - i b_k) / 2
    zf, zg = (c_f - 1j * s_f) / 2, (c_g - 1j * s_g) / 2
    zh = zf * zg
    return np.concatenate([[a_f * a_g], 2 * zh.real, -2 * zh.imag])


def complex_coef(coef: np.ndarray, N: int) -> np.ndarray:
    """hat u(k) for k = 0..N."""
    return np.concatenate([[coef[0]], (coef[1:N + 1] - 1j * coef[N + 1:]) / 2])
