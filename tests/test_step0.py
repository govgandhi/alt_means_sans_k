"""Tests for the step-0 (Fig. 1) pipeline: corpora, theory line and figure binning."""

import sys
from pathlib import Path

import pytest

for dependency in ("numpy", "scipy", "matplotlib", "networkx"):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "step0"), str(ROOT / "scripts")]
import corpora  # noqa: E402
import figure_paper  # noqa: E402
import theory  # noqa: E402
from simulate_norm_frequency import balanced_left_norm  # noqa: E402


def test_tokenize_drops_math_and_commands_and_splits_punctuation() -> None:
    assert corpora.tokenize(r"We study $S^1$ and \cite{x} the CSM.") == [
        "we",
        "study",
        "and",
        "x",
        "the",
        "csm",
        ".",
    ]


def test_walks_start_by_strength_and_step_by_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpora, "WALKS_PER_NODE", 2000)
    src = np.array([0, 1, 0, 2])
    dst = np.array([1, 0, 2, 0])
    w = np.array([3.0, 3.0, 1.0, 1.0])
    graph = corpora.csr(3, src, dst, w)
    walks = corpora.walks(graph, np.random.default_rng(0))
    starts = np.bincount(walks[:, 0], minlength=3) / len(walks)
    np.testing.assert_allclose(starts, [0.5, 0.375, 0.125], atol=0.06)
    after_zero = walks[:, 1:][walks[:, :-1] == 0]
    assert abs(np.mean(after_zero == 1) - 0.75) < 0.02


def test_null_keeps_total_strength_up_to_removed_self_loops() -> None:
    strength = np.array([40.0, 30.0, 20.0, 10.0])
    src, dst, w, loops = corpora.null_edges(strength, np.random.default_rng(1))
    realized = np.bincount(src, weights=w, minlength=4)
    assert realized.sum() + 2 * loops == strength.sum()
    assert np.all(realized <= strength)


def test_theory_scores_for_pnas_settings() -> None:
    counts = np.array([500.0, 300.0, 200.0])
    pi = counts / counts.sum()
    np.testing.assert_allclose(
        theory.scores(counts, 0.0, 1.0), -np.log(5) - np.log(1 - pi)
    )


def test_theory_level_equals_balanced_readout_of_rank_one_optimum() -> None:
    counts = np.random.default_rng(2).integers(50, 5000, size=40).astype(float)
    score = theory.scores(counts, 1e-3, 0.75)
    level = np.mean(score**2) ** 0.25
    np.testing.assert_allclose(
        balanced_left_norm(np.ones((40, 1)), score[:, None]), level, rtol=1e-12
    )


def test_curve_places_points_at_mean_log_x_and_drops_thin_bins(tmp_path: Path) -> None:
    x = np.concatenate([np.full(12, 2.0), np.full(12, 8.0), np.full(3, 1000.0)])
    norm = np.where(x < 5, 1.0, 3.0)
    for seed in range(2):
        np.savez(tmp_path / f"run_s{seed}.npz", x=x, raw_norm=norm)
    position, mean, se = figure_paper.curve(sorted(tmp_path.glob("*.npz")))
    np.testing.assert_allclose(position, [2.0, 8.0])  # the 3-token bin is dropped
    np.testing.assert_allclose(mean, [1.0, 3.0] / norm.mean())  # mean over all tokens
    np.testing.assert_allclose(se, [0.0, 0.0])
