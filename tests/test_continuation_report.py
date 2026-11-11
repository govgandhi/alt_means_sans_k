"""Check paired decline estimates and graph-level uncertainty independently."""

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
from window_training.report_continuation import aggregate_drops  # noqa: E402


def test_fractional_decline_is_paired_before_averaging() -> None:
    result = aggregate_drops(
        {
            "per_graph": [
                {"values": [2, 1], "high_minus_penultimate": -1},
                {"values": [10, 12], "high_minus_penultimate": 2},
            ]
        }
    )
    np.testing.assert_allclose(result["fractional_drop_by_graph"], [0.5, -0.2])
    assert result["fractional_mean"] == pytest.approx(0.15)
    assert result["fractional_se"] == pytest.approx(0.35)
    assert result["absolute_mean"] == pytest.approx(0.5)
    assert result["absolute_se"] == pytest.approx(1.5)
