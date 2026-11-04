"""Check equal-graph binning, paired contrasts, and descriptive shape readouts."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from window_training.report import node_curve, shape, summarize_nodes  # noqa: E402


def test_graphs_are_equally_weighted_not_nodes() -> None:
    degrees = [np.array([1, 1, 2, 3]), np.array([1, 2, 2, 3])]
    values = [np.array([2, 2, 4, 8]), np.array([10, 12, 12, 16])]
    result = summarize_nodes(degrees, values)
    assert [r["mean"] for r in result["bins"]] == [6, 8, 12]
    assert [r["standard_error"] for r in result["bins"]] == pytest.approx([4, 4, 4])
    assert [r["counts_by_seed"] for r in result["bins"]] == [[2, 1], [1, 2], [1, 1]]
    assert result["mean_curve"]["high_minus_penultimate"] == 4
    assert not result["mean_curve"]["interior_peak"]


def test_upper_gap_and_interior_maximum_are_different() -> None:
    result = shape([4, 3, 2])
    assert result["high_minus_penultimate"] < 0
    assert not result["interior_peak"]
    assert shape([1, 4, 3])["interior_peak"]
    curve = node_curve(np.array([0, 0, 1, 2]), np.array([1, 3, 5, 4]))
    assert curve["values"] == [2, 5, 4]
    assert curve["high_minus_low"] == 2
