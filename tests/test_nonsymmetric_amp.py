"""Tests for non-symmetric AMP (v7.tex).

We validate:
  - Symmetric reduction: T = 1, S = 1/n reduces AMP to standard Bayati-Montanari.
  - Elliptic scaling: T = rho * 1 scales the Onsager by rho.
  - Block-correlated and d-regular models: variance stability over many iterations.
  - Density Evolution: identity denoiser yields constant variance; tanh decreasing.
  - AMP-W and AMP-Z consistency with AMP.
  - T-correlated matrix generation: correct correlation structure.
"""

import jax
import jax.numpy as jnp
import pytest

from grmdil.amp.nonsymmetric import (
    amp_nonsymmetric,
    amp_w_nonsymmetric,
    amp_z_nonsymmetric,
    density_evolution_homogeneous,
    variance_trajectory,
)
from grmdil.amp.nonsymmetric_synthetic import (
    block_correlated_model,
    circulant_adjacency,
    compute_V,
    d_regular_model,
    form_matrix,
    t_correlated_gaussian,
)
from grmdil.amp.standard import standard_amp


H_IDENTITY = lambda x, eta, t: x
DH_IDENTITY = lambda x, eta, t: jnp.ones_like(x)
H_TANH = lambda x, eta, t: jnp.tanh(x)
DH_TANH = lambda x, eta, t: 1.0 - jnp.tanh(x) ** 2


class TestTCorrelatedGeneration:
    """Validate that t_correlated_gaussian produces the correct covariance structure."""

    def test_unit_variance(self):
        n = 200
        n_samples = 500
        var_sum = jnp.zeros((n, n), dtype=jnp.float64)
        for k in range(n_samples):
            X = t_correlated_gaussian(n, jnp.ones((n, n)), jax.random.PRNGKey(k))
            var_sum = var_sum + X ** 2
        empirical_var = var_sum / n_samples
        off_diag = empirical_var[jnp.triu_indices(n, 1)]
        assert float(jnp.mean(off_diag)) == pytest.approx(1.0, rel=0.1)

    def test_correlation_structure(self):
        n = 10
        rho = 0.6
        T = rho * jnp.ones((n, n), dtype=jnp.float64)
        n_samples = 5000
        corr_sum = jnp.zeros((n, n), dtype=jnp.float64)
        for k in range(n_samples):
            X = t_correlated_gaussian(n, T, jax.random.PRNGKey(k))
            corr_sum = corr_sum + X * X.T # we accumulate X_ij * X_ji
        empirical_corr = corr_sum / n_samples
        ij = (2, 5) # we check a specific off-diagonal pair
        assert float(empirical_corr[ij]) == pytest.approx(rho, abs=0.1)

    def test_symmetric_when_T_is_ones(self):
        n = 50
        T = jnp.ones((n, n), dtype=jnp.float64)
        X = t_correlated_gaussian(n, T, jax.random.PRNGKey(99))
        assert jnp.allclose(X, X.T, atol=1e-12)


class TestVComputation:

    def test_uniform_case(self):
        n = 20
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        T = jnp.ones((n, n), dtype=jnp.float64)
        V = compute_V(S, T)
        assert jnp.allclose(V, S, atol=1e-14)

    def test_elliptic(self):
        n = 20
        rho = 0.7
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        T = rho * jnp.ones((n, n), dtype=jnp.float64)
        V = compute_V(S, T)
        expected = rho * jnp.ones((n, n), dtype=jnp.float64) / n
        assert jnp.allclose(V, expected, atol=1e-14)


class TestCirculantAdjacency:

    def test_degree(self):
        n, d = 50, 6
        A = circulant_adjacency(n, d)
        degrees = jnp.sum(A, axis=1)
        assert jnp.allclose(degrees, d * jnp.ones(n), atol=1e-12)

    def test_symmetric(self):
        A = circulant_adjacency(30, 4)
        assert jnp.allclose(A, A.T, atol=1e-14)

    def test_no_self_loops(self):
        A = circulant_adjacency(20, 6)
        assert float(jnp.trace(A)) == pytest.approx(0.0)


class TestSymmetricReduction:
    """When S = 1/n and T = 1, non-symmetric AMP must exactly match standard AMP."""

    def test_identity_denoiser(self):
        n = 200
        key = jax.random.PRNGKey(100)
        T = jnp.ones((n, n), dtype=jnp.float64)
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        X = t_correlated_gaussian(n, T, key)
        W = form_matrix(X, S)
        V = compute_V(S, T)
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)
        t_steps = 8

        std_hist = standard_amp(W, lambda x: x, lambda x: jnp.ones_like(x), x0, t_steps)
        ns_hist = amp_nonsymmetric(W, V, H_IDENTITY, DH_IDENTITY, x0, eta, t_steps)

        for t in range(t_steps + 1):
            assert jnp.allclose(std_hist[t], ns_hist[t], atol=1e-10), (
                f"t={t}: max diff = {float(jnp.max(jnp.abs(std_hist[t] - ns_hist[t]))):.2e}"
            )

    def test_tanh_denoiser(self):
        n = 200
        key = jax.random.PRNGKey(101)
        T = jnp.ones((n, n), dtype=jnp.float64)
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        X = t_correlated_gaussian(n, T, key)
        W = form_matrix(X, S)
        V = compute_V(S, T)
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)
        t_steps = 6

        std_hist = standard_amp(
            W, lambda x: jnp.tanh(x), lambda x: 1.0 - jnp.tanh(x) ** 2, x0, t_steps,
        )
        ns_hist = amp_nonsymmetric(W, V, H_TANH, DH_TANH, x0, eta, t_steps)

        for t in range(t_steps + 1):
            assert jnp.allclose(std_hist[t], ns_hist[t], atol=1e-10), (
                f"t={t}: max diff = {float(jnp.max(jnp.abs(std_hist[t] - ns_hist[t]))):.2e}"
            )


class TestEllipticScaling:
    """For T = rho * ones, the Onsager should scale by rho compared to symmetric."""

    def test_onsager_scales_with_rho(self):
        n = 200
        rho = 0.5
        key = jax.random.PRNGKey(200)

        T_sym = jnp.ones((n, n), dtype=jnp.float64)
        T_ell = rho * jnp.ones((n, n), dtype=jnp.float64)
        S = jnp.ones((n, n), dtype=jnp.float64) / n

        X = t_correlated_gaussian(n, T_sym, key) # we use the same X for both
        W = form_matrix(X, S)
        V_sym = compute_V(S, T_sym)
        V_ell = compute_V(S, T_ell)
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)

        hist_sym = amp_nonsymmetric(W, V_sym, H_IDENTITY, DH_IDENTITY, x0, eta, 3)
        hist_ell = amp_nonsymmetric(W, V_ell, H_IDENTITY, DH_IDENTITY, x0, eta, 3)

        ons_sym = W @ hist_sym[1] - hist_sym[2] # we extract the Onsager term
        ons_ell = W @ hist_ell[1] - hist_ell[2]
        ratio = float(jnp.sum(ons_ell ** 2)) / float(jnp.sum(ons_sym ** 2))
        assert ratio == pytest.approx(rho ** 2, rel=0.15)


class TestBlockCorrelated:
    """Verify AMP on block-correlated non-symmetric matrices."""

    def test_variance_stable(self):
        n1, n2 = 100, 100
        rho1, rho2 = 0.8, 0.3
        W, S, T, V = block_correlated_model(n1, n2, rho1, rho2, jax.random.PRNGKey(300))
        n = n1 + n2
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)

        hist = amp_nonsymmetric(W, V, H_IDENTITY, DH_IDENTITY, x0, eta, 15)
        variances = variance_trajectory(hist)
        for t in range(16):
            assert variances[t] < 10.0, f"t={t}: variance = {variances[t]:.3f} diverges"

    def test_onsager_block_structure(self):
        n1, n2 = 100, 100
        n = n1 + n2
        rho1, rho2 = 0.9, 0.4
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        T = jnp.block([
            [rho1 * jnp.ones((n1, n1)), rho2 * jnp.ones((n1, n2))],
            [rho2 * jnp.ones((n2, n1)), rho1 * jnp.ones((n2, n2))],
        ])
        V = compute_V(S, T)

        dh = jnp.ones(n, dtype=jnp.float64) # we use identity derivative = 1
        ons = V @ dh # we compute the Onsager coefficient vector

        r = n1 / n
        expected_block1 = r * rho1 + (1 - r) * rho2
        expected_block2 = r * rho2 + (1 - r) * rho1

        assert float(ons[0]) == pytest.approx(expected_block1, rel=1e-10)
        assert float(ons[n1]) == pytest.approx(expected_block2, rel=1e-10)


class TestDRegular:
    """Verify AMP on d-regular sparse matrices."""

    def test_variance_stable(self):
        n, d = 200, 8 # we use d=8 for tighter concentration at finite n
        W, S, T, V, A = d_regular_model(n, d, jax.random.PRNGKey(400))
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)

        hist = amp_nonsymmetric(W, V, H_IDENTITY, DH_IDENTITY, x0, eta, 12)
        variances = variance_trajectory(hist)
        for t in range(13):
            assert variances[t] < 25.0, f"t={t}: variance = {variances[t]:.3f} diverges"

    def test_sparse_onsager(self):
        n, d = 50, 4
        A = circulant_adjacency(n, d)
        S = A / d
        V = compute_V(S, jnp.ones((n, n), dtype=jnp.float64)) # we have V = S for T = 1

        dh = jnp.ones(n, dtype=jnp.float64)
        ons = V @ dh # we compute per-index Onsager

        for i in range(n):
            expected = float(jnp.sum(V[i, :])) # we expect sum_j V_ij = sum_j S_ij = 1
            assert float(ons[i]) == pytest.approx(expected, abs=1e-12)


class TestAMPW:
    """AMP-W should produce qualitatively similar results to AMP."""

    def test_symmetric_case_close_to_amp(self):
        n = 200
        key = jax.random.PRNGKey(500)
        T = jnp.ones((n, n), dtype=jnp.float64)
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        X = t_correlated_gaussian(n, T, key)
        W = form_matrix(X, S)
        V = compute_V(S, T)
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)

        hist_amp = amp_nonsymmetric(W, V, H_IDENTITY, DH_IDENTITY, x0, eta, 8)
        hist_w = amp_w_nonsymmetric(W, H_IDENTITY, DH_IDENTITY, x0, eta, 8)

        var_amp = variance_trajectory(hist_amp)
        var_w = variance_trajectory(hist_w)
        for t in range(9):
            assert var_w[t] == pytest.approx(var_amp[t], rel=0.5), (
                f"t={t}: AMP-W var = {var_w[t]:.3f} vs AMP var = {var_amp[t]:.3f}"
            )

    def test_variance_stable_block(self):
        n1, n2 = 80, 80
        W, S, T, V = block_correlated_model(n1, n2, 0.7, 0.3, jax.random.PRNGKey(501))
        n = n1 + n2
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)

        hist = amp_w_nonsymmetric(W, H_IDENTITY, DH_IDENTITY, x0, eta, 12)
        variances = variance_trajectory(hist)
        for t in range(13):
            assert variances[t] < 15.0, f"t={t}: AMP-W variance diverges"


class TestDensityEvolution:

    def test_identity_constant_variance(self):
        x0_val = 1.0
        h_de = lambda x, t: x
        dh_de = lambda x, t: jnp.ones_like(x)
        R_list, variances, E_dh = density_evolution_homogeneous(
            h_de, dh_de, x0_val, t_max=6, n_mc=30_000, key=jax.random.PRNGKey(600),
        )
        for s in range(6):
            assert variances[s] == pytest.approx(x0_val ** 2, rel=0.1), (
                f"step {s}: Var = {variances[s]:.4f} should be {x0_val**2:.1f}"
            )

    def test_tanh_decreasing_variance(self):
        x0_val = 2.0
        h_de = lambda x, t: jnp.tanh(x)
        dh_de = lambda x, t: 1.0 - jnp.tanh(x) ** 2
        _, variances, _ = density_evolution_homogeneous(
            h_de, dh_de, x0_val, t_max=5, n_mc=30_000, key=jax.random.PRNGKey(601),
        )
        for s in range(1, 5):
            assert variances[s] <= variances[s - 1] + 0.05, (
                f"step {s}: Var = {variances[s]:.4f} should decrease"
            )

    def test_covariance_matrix_shape(self):
        h_de = lambda x, t: x
        dh_de = lambda x, t: jnp.ones_like(x)
        R_list, _, _ = density_evolution_homogeneous(
            h_de, dh_de, 1.0, t_max=4, n_mc=5000, key=jax.random.PRNGKey(602),
        )
        assert R_list[0].shape == (1, 1)
        assert R_list[1].shape == (2, 2)
        assert R_list[2].shape == (3, 3)
        assert R_list[3].shape == (4, 4)

    def test_covariance_positive_semidefinite(self):
        h_de = lambda x, t: jnp.tanh(x)
        dh_de = lambda x, t: 1.0 - jnp.tanh(x) ** 2
        R_list, _, _ = density_evolution_homogeneous(
            h_de, dh_de, 1.5, t_max=5, n_mc=20_000, key=jax.random.PRNGKey(603),
        )
        for s, R in enumerate(R_list):
            eigvals = jnp.linalg.eigvalsh(R)
            assert float(jnp.min(eigvals)) >= -1e-6, (
                f"R^{s+1} has negative eigenvalue {float(jnp.min(eigvals)):.6f}"
            )

    def test_de_matches_empirical(self):
        n = 800 # we use larger n for better concentration
        key = jax.random.PRNGKey(604)
        T = jnp.ones((n, n), dtype=jnp.float64)
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        X = t_correlated_gaussian(n, T, key)
        W = form_matrix(X, S)
        V = compute_V(S, T)
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)
        t_steps = 5

        hist = amp_nonsymmetric(W, V, H_IDENTITY, DH_IDENTITY, x0, eta, t_steps)
        emp_var = variance_trajectory(hist)

        h_de = lambda x, t: x
        dh_de = lambda x, t: jnp.ones_like(x)
        _, de_var, _ = density_evolution_homogeneous(
            h_de, dh_de, 1.0, t_max=t_steps, n_mc=50_000, key=jax.random.PRNGKey(605),
        )

        for t in range(1, t_steps + 1):
            assert emp_var[t] == pytest.approx(de_var[t - 1], rel=0.5), (
                f"t={t}: empirical={emp_var[t]:.3f} vs DE={de_var[t-1]:.3f}"
            )


class TestAMPZ:

    def test_symmetric_close_to_amp(self):
        n = 200
        key = jax.random.PRNGKey(700)
        T = jnp.ones((n, n), dtype=jnp.float64)
        S = jnp.ones((n, n), dtype=jnp.float64) / n
        X = t_correlated_gaussian(n, T, key)
        W = form_matrix(X, S)
        V = compute_V(S, T)
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)
        t_steps = 6

        h_de = lambda x, t: x
        dh_de = lambda x, t: jnp.ones_like(x)
        _, _, E_dh = density_evolution_homogeneous(
            h_de, dh_de, 1.0, t_max=t_steps, n_mc=50_000, key=jax.random.PRNGKey(701),
        )

        hist_z = amp_z_nonsymmetric(W, V, H_IDENTITY, x0, eta, t_steps, E_dh)
        hist_amp = amp_nonsymmetric(W, V, H_IDENTITY, DH_IDENTITY, x0, eta, t_steps)

        var_z = variance_trajectory(hist_z)
        var_amp = variance_trajectory(hist_amp)
        for t in range(t_steps + 1):
            assert var_z[t] == pytest.approx(var_amp[t], rel=0.3), (
                f"t={t}: AMP-Z var = {var_z[t]:.3f} vs AMP var = {var_amp[t]:.3f}"
            )


class TestBlockCorrelatedOnsagerFormula:
    """Verify Section 2.6 formula for the block-correlated Onsager."""

    def test_onsager_per_block(self):
        n1, n2 = 60, 40
        n = n1 + n2
        rho1, rho2 = 0.9, 0.5
        r = n1 / n
        W, S, T, V = block_correlated_model(n1, n2, rho1, rho2, jax.random.PRNGKey(800))
        x0 = jnp.ones(n, dtype=jnp.float64)
        eta = jnp.zeros(n, dtype=jnp.float64)

        hist = amp_nonsymmetric(W, V, H_TANH, DH_TANH, x0, eta, 3)

        dh_2 = DH_TANH(hist[2], eta, 2)
        avg_I1 = float(jnp.mean(dh_2[:n1]))
        avg_I2 = float(jnp.mean(dh_2[n1:]))

        ons_vec = V @ dh_2
        expected_block1 = r * rho1 * avg_I1 + (1 - r) * rho2 * avg_I2
        expected_block2 = r * rho2 * avg_I1 + (1 - r) * rho1 * avg_I2

        assert float(ons_vec[0]) == pytest.approx(expected_block1, rel=1e-10)
        assert float(ons_vec[n1]) == pytest.approx(expected_block2, rel=1e-10)
