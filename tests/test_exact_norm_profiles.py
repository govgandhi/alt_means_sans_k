"""Check the construction using factorized scores and independently evaluated loss."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from window_training.exact_norm_profiles import Config, construct, weights


@pytest.mark.parametrize("tolerance", [10.0, 1e-2, 1e-6, 1e-10])
def test_distinct_shapes_share_rms_and_approach_same_infimum(tolerance: float) -> None:
    config = Config()
    degree, positive, negative = weights(config)
    profiles = {
        name: np.array(h)[degree.astype(int) - 1] for name, h in config.degree_profiles
    }
    result = construct(positive, negative, profiles, tolerance)
    assert result["zero_pair_negative_mass"] == pytest.approx(35 / 16)
    normal_shapes = []
    for row in result["examples"]:
        matrix = np.array(row["scores"])
        values, vectors = np.linalg.eigh(-matrix)
        u = vectors * np.sqrt(values)
        v = -u
        np.testing.assert_allclose(u @ v.T, matrix, atol=1e-12)
        np.testing.assert_allclose(u.T @ u, v.T @ v, atol=1e-12)
        radii = np.linalg.norm(u, axis=1)
        target = profiles[row["profile"]]
        np.testing.assert_allclose(
            radii / np.sqrt(np.mean(radii**2)),
            target / np.sqrt(np.mean(target**2)),
            atol=1e-12,
        )
        assert np.mean(radii**2) == pytest.approx(result["common_diagonal_scale"])
        total = positive + negative
        free = -np.sum(negative * np.log(negative / total))
        mask = positive > 0
        free -= np.sum(positive[mask] * np.log(positive[mask] / total[mask]))
        actual = np.sum(
            positive * np.logaddexp(0, -matrix) + negative * np.logaddexp(0, matrix)
        )
        assert actual - free == pytest.approx(row["excess_loss"], abs=1e-14)
        assert actual - free <= tolerance
        normal_shapes.append(target / np.sqrt(np.mean(target**2)))
    assert (
        min(
            np.linalg.norm(a - b)
            for i, a in enumerate(normal_shapes)
            for b in normal_shapes[i + 1 :]
        )
        > 0.5
    )


def test_arbitrary_positive_profiles_not_only_three_chosen_shapes() -> None:
    _, positive, negative = weights(Config())
    rng = np.random.default_rng(6721)
    profiles = {str(i): np.exp(rng.normal(size=4)) for i in range(7)}
    result = construct(positive, negative, profiles, 1e-8)
    for row in result["examples"]:
        target = profiles[row["profile"]]
        np.testing.assert_allclose(
            row["normalized_radius"], target / np.sqrt(np.mean(target**2)), atol=1e-11
        )


@pytest.mark.parametrize("tolerance", [0, -1, float("nan")])
def test_invalid_tolerance_is_rejected(tolerance: float) -> None:
    _, positive, negative = weights(Config())
    with pytest.raises(ValueError, match="tolerance"):
        construct(positive, negative, {"a": np.ones(4)}, tolerance)


def test_no_zero_or_undefined_profile() -> None:
    _, positive, negative = weights(Config())
    with pytest.raises(ValueError, match="profile"):
        construct(positive, negative, {"a": np.array([1, 0, 1, 1])}, 1e-6)
