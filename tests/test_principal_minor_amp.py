"""Comprehensive tests for the AMP module.

We validate correctness against known theoretical predictions:
  - Standard AMP with identity denoiser on GOE: variance stays constant (SE).
  - Power iteration (no Onsager): variance diverges.
  - Spectral cleaning reduces operator norm of corrupted matrices.
  - Robust (clipped) AMP approximates standard AMP on corrupted data.
"""

import jax
import jax.numpy as jnp
import pytest

from grmdil.amp.principal_minor import (
    clip_coordinate,
    cutoff_eps,
    op_norm_symmetric,
    robust_amp,
    run_robust_amp_principal_minor,
    spectral_cleaning,
    spectral_threshold_k,
)
from grmdil.amp.standard import (
    power_iteration,
    standard_amp,
    state_evolution_identity,
    variance_trajectory,
)
from grmdil.amp.synthetic import goe_matrix, principal_minor_corruption


IDENTITY = lambda x: x
IDENTITY_DERIV = lambda x: jnp.ones_like(x)


class TestCoreUtilities:
    def test_cutoff_matches_paper_footnote(self):
        c = cutoff_eps(0.02, 16.0)
        assert c == pytest.approx((16.0 * jnp.log(1.0 / 0.02)) ** 0.5, rel=1e-6)

    def test_clip_saturates(self):
        y = jnp.array([-100.0, 0.5, 3.0])
        out = clip_coordinate(y, 0.02, 16.0)
        lim = cutoff_eps(0.02, 16.0)
        assert float(jnp.max(jnp.abs(out))) <= lim + 1e-9

    def test_clip_preserves_small_values(self):
        y = jnp.array([0.1, -0.2, 0.0])
        out = clip_coordinate(y, 0.02, 16.0)
        assert jnp.allclose(y, out)

    def test_op_norm_diagonal(self):
        d = jnp.diag(jnp.array([3.0, -2.0, 1.0]))
        assert float(op_norm_symmetric(d)) == pytest.approx(3.0)

    def test_spectral_threshold(self):
        assert spectral_threshold_k(5.0) == pytest.approx(10.0)


class TestGOEGeneration:
    def test_symmetric(self):
        x = goe_matrix(32, jax.random.PRNGKey(0))
        assert jnp.allclose(x, x.T, atol=1e-14)

    def test_variance_scaling(self):
        n = 500
        x = goe_matrix(n, jax.random.PRNGKey(1))
        off_diag = x[jnp.triu_indices(n, k=1)]
        empirical_var = float(jnp.var(off_diag))
        assert empirical_var == pytest.approx(1.0 / n, rel=0.15)

    def test_operator_norm_near_2(self):
        n = 500
        x = goe_matrix(n, jax.random.PRNGKey(2))
        norm = float(op_norm_symmetric(x))
        assert 1.5 < norm < 3.0


class TestPrincipalMinorCorruption:
    def test_shape_and_symmetry(self):
        n = 20
        x = goe_matrix(n, jax.random.PRNGKey(3))
        y, _ = principal_minor_corruption(x, jax.random.PRNGKey(4), 0.1, 5.0)
        assert y.shape == (n, n)
        assert jnp.allclose(y, y.T, atol=1e-14)

    def test_corruption_support_size(self):
        n = 100
        x = goe_matrix(n, jax.random.PRNGKey(5))
        _, s_idx = principal_minor_corruption(x, jax.random.PRNGKey(6), 0.05, 5.0)
        assert s_idx.shape[0] == 5

    def test_corruption_increases_op_norm(self):
        n = 200
        x = goe_matrix(n, jax.random.PRNGKey(7))
        y, _ = principal_minor_corruption(x, jax.random.PRNGKey(8), 0.05, 20.0)
        assert float(op_norm_symmetric(y)) > float(op_norm_symmetric(x))


class TestStandardAMP:
    """The key validation: identity-denoiser AMP on clean GOE should have
    ||x^{(t)}||^2/n ~ 1 for all t (state evolution prediction)."""

    def test_identity_variance_stable(self):
        n = 800
        w = goe_matrix(n, jax.random.PRNGKey(10))
        x0 = jnp.ones(n)
        history = standard_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=10)
        variances = variance_trajectory(history)
        se = state_evolution_identity(1.0, 10)
        for t in range(11):
            assert variances[t] == pytest.approx(se[t], rel=0.25), (
                f"t={t}: ||x||^2/n = {variances[t]:.3f}, SE = {se[t]:.3f}"
            )

    def test_power_iteration_diverges(self):
        n = 800
        w = goe_matrix(n, jax.random.PRNGKey(11))
        x0 = jnp.ones(n)
        history = power_iteration(w, IDENTITY, x0, t_steps=10)
        variances = variance_trajectory(history)
        assert variances[10] > 10 * variances[0], (
            "power iteration should diverge without Onsager"
        )

    def test_onsager_decorrelation(self):
        """AMP iterates should be approximately uncorrelated with past iterates.
        Without Onsager, they accumulate correlation."""
        n = 800
        w = goe_matrix(n, jax.random.PRNGKey(12))
        x0 = jnp.ones(n)
        amp_hist = standard_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=6)
        pow_hist = power_iteration(w, IDENTITY, x0, t_steps=6)
        amp_corr = abs(float(jnp.dot(amp_hist[6], amp_hist[4]) / n))
        pow_corr = abs(float(jnp.dot(pow_hist[6], pow_hist[4]) / n))
        assert amp_corr < pow_corr, (
            f"AMP correlation {amp_corr:.3f} should be < power iter {pow_corr:.3f}"
        )


class TestSpectralCleaning:
    def test_reduces_spiked_matrix(self):
        n = 32
        y = jnp.eye(n) * 0.1
        y = y.at[0, 0].set(50.0)
        k = spectral_threshold_k(5.0)
        y_hat, removed = spectral_cleaning(y, k, jax.random.PRNGKey(20))
        assert float(op_norm_symmetric(y_hat)) <= k + 1e-4
        assert 0 in removed.tolist()

    def test_noop_when_below_threshold(self):
        n = 16
        y = jnp.eye(n) * 0.01
        y_hat, removed = spectral_cleaning(y, 1.0, jax.random.PRNGKey(21))
        assert removed.size == 0
        assert jnp.allclose(y_hat, y)

    def test_on_corrupted_goe(self):
        n = 200
        x = goe_matrix(n, jax.random.PRNGKey(22))
        y, _ = principal_minor_corruption(x, jax.random.PRNGKey(23), 0.05, 15.0)
        k = spectral_threshold_k(5.0)
        y_hat, removed = spectral_cleaning(y, k, jax.random.PRNGKey(24), max_removals=n)
        assert float(op_norm_symmetric(y_hat)) <= k + 1e-4
        assert removed.shape[0] <= int(0.5 * n)


class TestRobustAMP:
    def test_matches_standard_on_clean_matrix(self):
        """Without corruption, robust AMP (with very loose clip) should match standard."""
        n = 200
        w = goe_matrix(n, jax.random.PRNGKey(30))
        x0 = jnp.ones(n)
        std_hist = standard_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=5)
        rob_hist = robust_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=5, eps=1e-10, c_t=16.0)
        for t in range(6):
            assert jnp.allclose(std_hist[t], rob_hist[t], atol=1e-10), (
                f"t={t}: robust != standard on clean matrix"
            )

    def test_clipping_is_active(self):
        """With aggressive clip, some coordinates should be saturated."""
        n = 200
        w = goe_matrix(n, jax.random.PRNGKey(31))
        x0 = jnp.ones(n) * 5.0
        rob_hist = robust_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=3, eps=0.3, c_t=4.0)
        lim = cutoff_eps(0.3, 4.0)
        for t in range(1, 4):
            assert float(jnp.max(jnp.abs(rob_hist[t]))) <= lim + 1e-10

    def test_robust_variance_stable(self):
        """Robust AMP on a clean matrix should also have stable variance."""
        n = 800
        w = goe_matrix(n, jax.random.PRNGKey(32))
        x0 = jnp.ones(n)
        rob_hist = robust_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=8, eps=1e-10, c_t=16.0)
        variances = variance_trajectory(rob_hist)
        for t in range(9):
            assert variances[t] == pytest.approx(1.0, rel=0.25), (
                f"t={t}: ||y||^2/n = {variances[t]:.3f}"
            )


class TestEndToEndPipeline:
    def test_robust_approximates_standard_on_corruption(self):
        """Main theorem test: ||y^{(T)} - x^{(T)}||^2 should be O(eps) * ||x^{(T)}||^2."""
        n = 400
        eps_corruption = 0.02
        x = goe_matrix(n, jax.random.PRNGKey(40))
        y, _ = principal_minor_corruption(x, jax.random.PRNGKey(41), eps_corruption, 8.0)
        x0 = jnp.ones(n)
        t_steps = 5
        std_hist = standard_amp(x, IDENTITY, IDENTITY_DERIV, x0, t_steps=t_steps)
        rob_hist, w_hat, removed = run_robust_amp_principal_minor(
            y, IDENTITY, IDENTITY_DERIV, x0, t_steps,
            eps=eps_corruption, c_t=16.0,
            key=jax.random.PRNGKey(42), max_removals=n,
        )
        x_T = std_hist[-1]
        y_T = rob_hist[-1]
        mse = float(jnp.sum((y_T - x_T) ** 2))
        signal_energy = float(jnp.sum(x_T ** 2))
        relative_error = mse / max(signal_energy, 1e-12)
        assert relative_error < 2.0, (
            f"relative error {relative_error:.3f} too large (expected O(eps))"
        )

    def test_pipeline_spectral_cleaning_works(self):
        n = 200
        x = goe_matrix(n, jax.random.PRNGKey(43))
        y, _ = principal_minor_corruption(x, jax.random.PRNGKey(44), 0.05, 12.0)
        _, w_hat, removed = run_robust_amp_principal_minor(
            y, IDENTITY, IDENTITY_DERIV, jnp.ones(n), 3,
            eps=0.05, c_t=16.0,
            key=jax.random.PRNGKey(45), max_removals=n,
        )
        assert float(op_norm_symmetric(w_hat)) <= spectral_threshold_k(5.0) + 1e-4


class TestManualStepVerification:
    """Step-by-step verification against hand computation."""

    def test_standard_amp_first_two_steps(self):
        n = 8
        w = jax.random.normal(jax.random.PRNGKey(50), (n, n))
        w = (w + w.T) / jnp.sqrt(2.0)
        x0 = jnp.ones(n)
        history = standard_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=2)
        x1_expected = w @ x0
        x2_expected = w @ x1_expected - 1.0 * x0
        assert jnp.allclose(history[1], x1_expected, atol=1e-12)
        assert jnp.allclose(history[2], x2_expected, atol=1e-12)

    def test_robust_amp_first_step(self):
        n = 8
        w = jax.random.normal(jax.random.PRNGKey(51), (n, n))
        w = (w + w.T) / jnp.sqrt(2.0)
        x0 = jnp.ones(n)
        rob_hist = robust_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=1, eps=0.02, c_t=16.0)
        expected = clip_coordinate(w @ x0, 0.02, 16.0)
        assert jnp.allclose(rob_hist[1], expected, atol=1e-12)

    def test_robust_amp_second_step_onsager(self):
        n = 8
        w = jax.random.normal(jax.random.PRNGKey(52), (n, n))
        w = (w + w.T) / jnp.sqrt(2.0)
        x0 = jnp.ones(n)
        eps, c_t = 0.02, 16.0
        y1 = clip_coordinate(w @ x0, eps, c_t)
        b_11 = float(jnp.mean(jnp.ones(n)))
        y2_raw = w @ y1 - b_11 * x0
        y2_expected = clip_coordinate(y2_raw, eps, c_t)
        rob_hist = robust_amp(w, IDENTITY, IDENTITY_DERIV, x0, t_steps=2, eps=eps, c_t=c_t)
        assert jnp.allclose(rob_hist[2], y2_expected, atol=1e-12)
