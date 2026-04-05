"""Tests for additional synthetic corruption / matrix generators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from grmdil.amp.synthetic import (
    dense_symmetric_gaussian_noise,
    entrywise_sparse_corruption,
    goe_matrix,
    laplace_wigner,
    principal_minor_corruption,
)


def test_symmetry_and_shape() -> None:
    n = 40
    key = jax.random.PRNGKey(0)
    x = goe_matrix(n, key)
    y = entrywise_sparse_corruption(x, jax.random.PRNGKey(1), p=0.05, strength=2.0)
    assert y.shape == (n, n)
    assert float(jnp.max(jnp.abs(y - y.T))) < 1e-10

    z = dense_symmetric_gaussian_noise(x, jax.random.PRNGKey(2), sigma=0.1)
    assert z.shape == (n, n)
    assert float(jnp.max(jnp.abs(z - z.T))) < 1e-10

    w = laplace_wigner(n, jax.random.PRNGKey(3))
    assert w.shape == (n, n)
    assert float(jnp.max(jnp.abs(w - w.T))) < 1e-10


def test_principal_minor_returns_indices() -> None:
    n = 20
    x = goe_matrix(n, jax.random.PRNGKey(0))
    y, s = principal_minor_corruption(x, jax.random.PRNGKey(1), eps=0.1, block_strength=1.0)
    assert s.shape[0] >= 1
    assert y.shape == (n, n)
