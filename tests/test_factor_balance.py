"""Independent checks of issue 29's diagnostics and finite-step qualification."""

import sys
from pathlib import Path

import pytest

for dependency in (
    "numpy",
    "scipy",
    "matplotlib",
    "networkx",
    "threadpoolctl",
    "gensim",
):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from factor_balance import Config, measures, output_directory  # noqa: E402


def test_balanced_rank_deficient_factors() -> None:
    center = np.array([[3.0, 0.0], [4.0, 0.0]])
    context = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
    result = measures(center, context)
    assert result["relative_imbalance"] == 0.0
    np.testing.assert_allclose(result["raw_norm"], result["balanced_norm"])


def test_small_initialization_is_not_relative_balance() -> None:
    center = np.array([[1e-5, 0.0], [0.0, 2e-5]])
    result = measures(center, np.zeros_like(center))
    assert result["relative_imbalance"] == 1.0
    assert result["relative_norm_l2_difference"] == 1.0


def test_equal_row_norms_do_not_imply_factor_balance() -> None:
    center = np.array([[1.0, 0.0], [0.6, 0.8]])
    context = np.linalg.inv(center).T
    result = measures(center, context)
    np.testing.assert_allclose(center @ context.T, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(result["raw_norm"], result["balanced_norm"])
    assert result["relative_imbalance"] > 0.5


def test_balanced_norm_matches_full_score_svd() -> None:
    rng = np.random.default_rng(29)
    center, context = rng.normal(size=(5, 2)), rng.normal(size=(7, 2))
    p, s, _ = np.linalg.svd(center @ context.T, full_matrices=False)
    expected = np.sqrt((p**2) @ s)
    np.testing.assert_allclose(measures(center, context)["balanced_norm"], expected)


def test_simultaneous_step_has_quadratic_drift() -> None:
    rng = np.random.default_rng(29)
    u, v, g = rng.normal(size=(5, 2)), rng.normal(size=(7, 2)), rng.normal(size=(5, 7))
    du, dv = -g @ v, -g.T @ u
    np.testing.assert_allclose(u.T @ du + du.T @ u, v.T @ dv + dv.T @ v)
    for eta in (0.1, 0.05):
        new_u, new_v = u + eta * du, v + eta * dv
        drift = new_u.T @ new_u - new_v.T @ new_v - (u.T @ u - v.T @ v)
        predicted = eta**2 * (v.T @ g.T @ g @ v - u.T @ g @ g.T @ u)
        np.testing.assert_allclose(drift, predicted, atol=1e-14)
        assert np.linalg.norm(drift) > 0


def test_output_cannot_replace_source(tmp_path: Path) -> None:
    for output in (
        "results",
        "results/subsampling_origin",
        "results/subsampling_origin/new",
    ):
        with pytest.raises(ValueError, match="overlaps"):
            output_directory(Config(root=tmp_path, output=Path(output)))


def test_zero_center_rows_are_not_silently_divided() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        measures(np.zeros((2, 2)), np.ones((2, 2)))
