"""Independent gradient, support, gauge, and optimizer accounting checks."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/window_training"))
import expected_window as experiment


def test_gradient_against_finite_differences_including_nonedge() -> None:
    positive = np.array([[0.0, 0.3], [0.2, 0.5]])
    negative = np.array([[0.7, 1.4], [0.6, 1.3]])
    packed = np.array([0.2, -0.7, 0.6, 0.1, -0.3, 0.8, 0.5, -0.4])
    value, gradient = experiment.loss_gradient(packed, positive, negative, 2)
    numerical = approx_derivative(
        lambda x: experiment.loss_gradient(x, positive, negative, 2)[0],
        packed,
        method="3-point",
    ).ravel()
    np.testing.assert_allclose(gradient, numerical, atol=1e-10, rtol=1e-7)
    center, context = packed.reshape(2, 2, 2)
    direct = sum(
        positive[i, j] * np.logaddexp(0, -np.dot(center[i], context[j]))
        + negative[i, j] * np.logaddexp(0, np.dot(center[i], context[j]))
        for i in range(2)
        for j in range(2)
    )
    assert value == pytest.approx(direct)


def test_infimum_and_boundary_are_not_smoothed() -> None:
    positive = np.array([[0.0, 0.5], [0.5, 0.0]])
    negative = np.full((2, 2), 0.25)
    benchmark = experiment.free_score_benchmark(positive, negative)
    assert not benchmark["finite_full_rank_norms_defined"]
    assert benchmark["minus_infinity_entries"] == 2
    assert "balanced_norm" not in benchmark
    low = experiment.unrestricted_infimum(positive, negative)
    assert low == pytest.approx(2 * (0.5 * np.log(1.5) + 0.25 * np.log(3)))


def test_full_support_exact_benchmark_and_balanced_readout() -> None:
    positive = np.array([[0.1, 0.4], [0.2, 0.3]])
    negative = np.full((2, 2), 0.25)
    result = experiment.free_score_benchmark(positive, negative)
    assert result["finite_full_rank_norms_defined"]
    assert result["gradient_max"] < 1e-12
    scores = np.log(positive / negative)
    eig, vectors = np.linalg.eigh(scores @ scores.T)
    independent = np.sqrt(vectors**2 @ np.sqrt(np.maximum(eig, 0)))
    np.testing.assert_allclose(result["balanced_norm"], independent)
    rng = np.random.default_rng(4)
    center, context = rng.normal(size=(2, 5, 3))
    left, singular, _ = np.linalg.svd(center @ context.T, full_matrices=False)
    direct = np.linalg.norm(left * np.sqrt(singular), axis=1)
    np.testing.assert_allclose(experiment.balanced_left_norm(center, context), direct)


def test_window_uses_positive_lags_and_preserves_zero_support() -> None:
    transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = experiment.uniform_windows(transition, (1, 2, 3))
    np.testing.assert_array_equal(result[1], transition)
    np.testing.assert_array_equal(result[2], np.full((2, 2), 0.5))
    np.testing.assert_allclose(result[3], [[1 / 3, 2 / 3], [2 / 3, 1 / 3]])


def test_fit_keeps_initialization_and_reports_actual_iterations() -> None:
    config = experiment.Config(
        nodes=3, dimension=2, maxiter=3, checkpoints=(0, 1, 2, 3)
    )
    conditional = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    result, center, context = experiment.fit(
        conditional, np.full(3, 1 / 3), np.full(3, 2), config, 12
    )
    first, last = result["snapshots"][0], result["snapshots"][-1]
    assert np.all(np.asarray(first["raw_norm"]) > 0)
    np.testing.assert_array_equal(first["balanced_norm"], 0)
    assert last["iteration"] == result["termination"]["iterations"] == 3
    assert last["expected_loss"] < first["expected_loss"]
    assert last["excess_over_unrestricted_infimum"] >= 0
    assert center.shape == context.shape == (3, 2)
