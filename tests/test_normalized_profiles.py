"""Analytic fixtures for the read-only balanced-radius profile audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from window_training import normalized_profiles as profiles  # noqa: E402


def test_balance_uses_score_geometry_not_factor_scaling() -> None:
    scores = np.array([[3.0, -1.0], [-1.0, 3.0]])
    # Positive definite: sqrt(M M^T)=M, so both squared radii are exactly 3.
    np.testing.assert_allclose(profiles.balanced_radius(scores), np.sqrt([3.0, 3.0]))
    np.testing.assert_allclose(profiles.balanced_radius(-scores), np.sqrt([3.0, 3.0]))
    np.testing.assert_allclose(
        profiles.balanced_radius(np.diag([4.0, -9.0])), [2.0, 3.0]
    )


def test_normalization_preserves_shape_and_handles_zero_explicitly() -> None:
    first_rms, first = profiles.normalized_radius(np.array([1.0, 2.0, 3.0]))
    second_rms, second = profiles.normalized_radius(np.array([10.0, 20.0, 30.0]))
    assert second_rms == pytest.approx(first_rms * 10)
    np.testing.assert_allclose(first, second)
    assert np.mean(first**2) == pytest.approx(1)
    assert profiles.normalized_radius(np.zeros(3)) == (0, None)
    with pytest.raises(ValueError):
        profiles.normalized_radius(np.array([-1.0, 2.0]))


def test_normalization_must_precede_start_averaging() -> None:
    starts = [np.array([1.0, 0.0]), np.array([0.0, 100.0])]
    normalized = [profiles.normalized_radius(row)[1] for row in starts]
    rows = [
        {"seed": seed, "curve": {"values": row.tolist()}}
        for seed in (0, 1)
        for row in normalized
    ]
    aggregate = profiles.aggregate_starts(rows, [0, 1], "curve")
    np.testing.assert_allclose(aggregate["mean"], np.full(2, 1 / np.sqrt(2)))
    np.testing.assert_allclose(aggregate["se"], [0, 0])
    wrong_order = profiles.normalized_radius(np.mean(starts, axis=0))[1]
    assert np.max(np.abs(wrong_order - aggregate["mean"])) > 0.6


def test_loss_and_infimum_match_scalar_exact_fixture() -> None:
    positive = np.array([[0.25, 0.0], [0.0, 0.75]])
    negative = np.ones((2, 2)) * 0.5
    scores = np.zeros((2, 2))
    assert profiles.score_loss(scores, positive, negative) == pytest.approx(
        3 * np.log(2)
    )
    expected = (
        0.25 * np.log(3) + 0.5 * np.log(1.5) + 0.75 * np.log(5 / 3) + 0.5 * np.log(2.5)
    )
    assert profiles.score_infimum(positive, negative) == pytest.approx(expected)
    limiting = np.full((2, 2), -40.0)
    np.fill_diagonal(limiting, np.log([0.5, 1.5]))
    assert profiles.score_loss(limiting, positive, negative) == pytest.approx(expected)


def test_full_support_bound_covers_known_diagonal_perturbation() -> None:
    target = np.diag([1.0, 4.0])
    negative = np.ones((2, 2))
    positive = np.exp(target)
    fitted = np.diag([1.2, 4.1])
    target_radius = np.array([1.0, 2.0])
    target_rms, target_normalized = profiles.normalized_radius(target_radius)
    fit_radius = np.sqrt([1.2, 4.1])
    _, fit_normalized = profiles.normalized_radius(fit_radius)
    gap = profiles.score_loss(fitted, positive, negative) - profiles.score_infimum(
        positive, negative
    )
    bound = profiles.full_support_bound(positive, negative, gap, target_rms)
    assert np.linalg.norm(fitted - target) <= bound["score_frobenius_error_bound"]
    assert (
        np.linalg.norm(fit_radius - target_radius) / np.sqrt(2)
        <= bound["radius_rms_error_bound"]
    )
    assert (
        np.linalg.norm(fit_normalized - target_normalized) / np.sqrt(2)
        <= bound["normalized_profile_rms_error_bound"]
    )


def test_zero_target_and_sparse_target_never_get_invented_certificates() -> None:
    weights = np.ones((2, 2))
    bound = profiles.full_support_bound(weights, weights, 0.0, 0.0)
    assert bound["normalized_profile_rms_error_bound"] is None
    assert "undefined" in bound["status"]
    with pytest.raises(ValueError, match="full positive support"):
        profiles.full_support_bound(np.eye(2), weights, 0.1, 1.0)
    bound = profiles.full_support_bound(weights, weights, 100.0, 1.0)
    assert bound["normalized_profile_rms_error_bound"] == pytest.approx(np.sqrt(2))
    assert "not certified" in bound["status"]
