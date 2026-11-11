"""Independent checks of balancing, tangent gradients and stability gates."""

import sys
from pathlib import Path

import pytest

for dependency in (
    "numpy",
    "scipy",
    "matplotlib",
    "networkx",
    "threadpoolctl",
):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from window_training import continue_windows as continuation  # noqa: E402


def test_balancing_preserves_scores_loss_and_canonical_norm() -> None:
    rng = np.random.default_rng(19)
    u, v = rng.normal(size=(2, 7, 3))
    scales = np.array([1e-3, 2, 1e3])
    u, v = u * scales, v / scales
    balanced_u, balanced_v, singular = continuation.balanced_factors(u, v)
    scores = u @ v.T
    np.testing.assert_allclose(balanced_u @ balanced_v.T, scores, atol=1e-12)
    np.testing.assert_allclose(
        balanced_u.T @ balanced_u, balanced_v.T @ balanced_v, atol=1e-12
    )
    left, full_singular, _ = np.linalg.svd(scores, full_matrices=False)
    np.testing.assert_allclose(
        np.linalg.norm(balanced_u, axis=1), np.sqrt(left**2 @ full_singular), atol=1e-7
    )
    positive, negative = rng.random((2, 7, 7))
    old_loss = np.sum(
        positive * np.logaddexp(0, -scores) + negative * np.logaddexp(0, scores)
    )
    new_loss, _ = continuation.original.loss_gradient(
        np.stack([balanced_u, balanced_v]).ravel(), positive, negative, 3
    )
    assert new_loss == pytest.approx(old_loss, abs=1e-10)
    assert len(singular) == 3


def test_projected_gradient_is_orthogonal_tangent_projection() -> None:
    rng = np.random.default_rng(82)
    u, v, s = continuation.balanced_factors(*rng.normal(size=(2, 6, 2)))
    g = rng.normal(size=(6, 6))
    left = u / np.sqrt(s)
    right = v / np.sqrt(s)
    projection = g - (np.eye(6) - left @ left.T) @ g @ (np.eye(6) - right @ right.T)
    value, rank = continuation.projected_gradient(g, u, v, s)
    assert rank == 2
    assert value == pytest.approx(np.linalg.norm(projection), abs=1e-12)
    assert abs(np.sum((g - projection) * projection)) < 1e-12


def test_each_stability_gate_is_required() -> None:
    config = continuation.Config()
    fields = [
        "balanced_gradient_max",
        "relative_loss_change",
        "relative_score_rms_change",
        "relative_norm_rms_change",
        "max_relative_bin_change",
        "absolute_upper_gap_change",
    ]
    values = dict.fromkeys(fields, 0.0)
    assert all(continuation.stable_gates(values, config).values())
    for field in fields:
        bad = {**values, field: 1.0}
        assert (
            sum(not value for value in continuation.stable_gates(bad, config).values())
            == 1
        )


def test_continuation_retains_target_and_stop_counts() -> None:
    rng = np.random.default_rng(5)
    u, v = rng.normal(size=(2, 4, 2)) * 0.1
    positive = np.array(
        [
            [0, 0.1, 0.15, 0],
            [0.1, 0, 0.1, 0.05],
            [0.05, 0.1, 0, 0.1],
            [0.1, 0.05, 0.1, 0],
        ]
    )
    positive /= positive.sum()
    negative = np.full((4, 4), 5 / 16)
    a, b = positive.copy(), negative.copy()
    config = continuation.Config(
        rank=2, block_iterations=3, max_blocks=2, control_blocks=2
    )
    result, _, _ = continuation.optimize(
        u, v, positive, negative, np.array([0, 1, 2, 3]), config, True
    )
    np.testing.assert_array_equal(positive, a)
    np.testing.assert_array_equal(negative, b)
    records = result["snapshots"]
    assert [r["block"] for r in records] == [0, 1, 2]
    assert records[-1]["expected_loss"] < records[0]["expected_loss"]
    assert records[-1]["cumulative_iterations"] == sum(
        r["optimizer"]["iterations"] for r in records[1:]
    )
    assert records[0]["score_quantiles"]["nonedge"] is not None
