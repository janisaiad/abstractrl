"""Sanity checks for compressed-sensing AMP (not Wigner AMP)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from grmdil.amp.compressed_sensing import amp_cs, ista_cs, soft_threshold, soft_threshold_prime


def test_amp_cs_beats_zero_on_sparse_signal() -> None:
    m, n = 300, 600
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, n)) / np.sqrt(m)
    k = 25
    sup = rng.choice(n, size=k, replace=False)
    x_true = np.zeros(n)
    x_true[sup] = rng.standard_normal(k)
    y = a @ x_true + 0.02 * rng.standard_normal(m)
    a_j = jnp.asarray(a)
    y_j = jnp.asarray(y)
    x_true_j = jnp.asarray(x_true)
    lam = 0.18
    eta = lambda r: soft_threshold(r, lam)
    etap = lambda r: soft_threshold_prime(r, lam)
    hist = amp_cs(a_j, y_j, eta, etap, t_steps=150)
    err = float(jnp.mean((hist[-1] - x_true_j) ** 2))
    err0 = float(jnp.mean(x_true_j**2))
    assert err < 0.25 * err0


def test_ista_runs() -> None:
    m, n = 80, 160
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, n)) / np.sqrt(m)
    x_true = rng.standard_normal(n) * (rng.random(n) > 0.7)
    y = a @ x_true
    a_j = jnp.asarray(a)
    y_j = jnp.asarray(y)
    lam = 0.05
    eta = lambda r: soft_threshold(r, lam)
    lip = float(jnp.linalg.norm(a_j, ord=2)) ** 2
    hist = ista_cs(a_j, y_j, eta, t_steps=50, step=1.0 / lip)
    assert hist[-1].shape == (n,)
