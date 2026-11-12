"""Check target boundaries and window semantics without training embeddings."""

import sys
from pathlib import Path

import pytest

for dependency in (
    "numpy",
    "scipy",
    "matplotlib",
    "networkx",
    "sympy",
    "pymupdf",
):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_realized_graph import (  # noqa: E402
    Config,
    balanced_factors,
    counterexamples,
    pairwise_scores,
    population_checks,
    symbolic_checks,
    uniform_windows,
)


def test_score_boundaries_and_no_smoothing() -> None:
    scores = pairwise_scores(
        np.array([0.0, 2.0, 0.0, 2.0]), np.array([3.0, 0.0, 0.0, 4.0])
    )
    assert np.isneginf(scores[0])
    assert np.isposinf(scores[1])
    assert np.isnan(scores[2])
    assert scores[3] == pytest.approx(-np.log(2))
    with pytest.raises(ValueError, match="finite exact"):
        balanced_factors(scores.reshape(2, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        pairwise_scores(np.array([-1.0]), np.array([1.0]))


def test_stationary_point_is_strict_maximum() -> None:
    positive = np.array([0.1, 0.5, 4.0])
    negative = np.array([2.0, 0.3, 1.0])
    optimum = pairwise_scores(positive, negative)

    def objective(x: np.ndarray) -> np.ndarray:
        return positive * np.log(expit(x)) + negative * np.log(expit(-x))

    assert np.all(objective(optimum) > objective(optimum - 0.5))
    assert np.all(objective(optimum) > objective(optimum + 0.5))


def test_skipped_negative_collision_changes_expected_objective() -> None:
    conditional = np.array([0.4, 0.6])
    noise = np.array([0.8, 0.2])
    proposals = 5 * conditional[:, None] * noise[None, :]
    np.fill_diagonal(proposals, 0)
    accepted_weights = proposals.sum(axis=0)
    target = pairwise_scores(conditional, accepted_weights)
    np.testing.assert_allclose(
        target, np.log(conditional / (5 * noise * (1 - conditional)))
    )
    ordinary = pairwise_scores(conditional, 5 * noise)
    assert np.all(target > ordinary)


def test_uniform_window_includes_all_positive_lags() -> None:
    edge = np.array([[0.0, 1.0], [1.0, 0.0]])
    windows = uniform_windows(edge, (1, 2, 3))
    np.testing.assert_array_equal(windows[1], edge)
    np.testing.assert_allclose(windows[2], np.full((2, 2), 0.5))
    np.testing.assert_allclose(windows[3], [[1 / 3, 2 / 3], [2 / 3, 1 / 3]])
    with pytest.raises(ValueError, match="positive"):
        uniform_windows(edge, (0,))


def test_zero_support_and_disconnected_classes_remain() -> None:
    transition = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rows = uniform_windows(transition, (10,))[10]
    np.testing.assert_array_equal(rows[:2, 2], [0.0, 0.0])
    assert rows[2, 2] == 1
    scores = pairwise_scores(rows, np.full((3, 3), 1 / 3))
    assert np.isneginf(scores[0, 2])
    assert np.isfinite(scores[0, :2]).all()


def test_finite_balanced_factor_norms_are_gauge_fixed() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    left, right = balanced_factors(matrix)
    assert left.shape[1] == 1
    np.testing.assert_allclose(left @ right.T, matrix, atol=1e-12)
    np.testing.assert_allclose(left.T @ left, right.T @ right, atol=1e-12)
    assert np.linalg.norm(left[1]) / np.linalg.norm(left[0]) == pytest.approx(2)


def test_population_fixtures_and_exact_symbolic_objective() -> None:
    checks = population_checks(Config())
    assert [row["communities"] for row in checks] == [1, 3]
    assert symbolic_checks()["identity_residuals"]["stationary_point"] == "0"
    for row in checks:
        assert max(row["checks"].values()) < 1e-10
        for score in row["scores"]:
            np.testing.assert_allclose(
                score["context_slopes"], 1 - score["alpha"], atol=1e-10
            )


def test_counterexamples_prevent_universal_inferences() -> None:
    examples = counterexamples(Config())
    edge = examples["periodic_edge"]["node_kl_by_window"]
    assert edge["1"][0] == pytest.approx(np.log(2))
    assert edge["2"][0] == pytest.approx(0)
    assert edge["3"][0] > 0
    before, after = examples["edge_addition"]["kl_before_after"]
    assert before == pytest.approx(np.log(2))
    assert after == pytest.approx(np.log(8 / 3))
    reversal = examples["kl_norm_reversal"]
    assert reversal["kl"][0] > reversal["kl"][1]
    assert reversal["balanced_center_norm"][0] < reversal["balanced_center_norm"][1]
