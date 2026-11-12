"""Guard the issue 44 audit helpers and its protection of historical outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

for dependency in ("numpy", "gensim", "matplotlib", "networkx", "sklearn"):
    pytest.importorskip(dependency, reason="Requires the pinned scientific environment")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue44_token_counts as audit  # noqa: E402


def test_legacy_counter_counts_walks_not_tokens() -> None:
    walks = [[0, 1, 0], [1, 1, 1]]
    np.testing.assert_array_equal(audit.legacy_visits(walks, 2), [1, 2])
    np.testing.assert_array_equal(np.bincount(np.concatenate(walks)), [2, 4])


def test_slope_is_undefined_on_regular_graphs() -> None:
    assert audit.slope(np.full(4, 6.0), np.array([1, 2, 3, 4])) is None
    assert audit.slope(np.array([1.0, 2.0, 4.0]), np.array([3, 6, 12])) == (
        pytest.approx(1.0)
    )


@pytest.mark.parametrize("name", audit.PROTECTED)
def test_output_cannot_overlap_historical_results(name: str) -> None:
    with pytest.raises(ValueError, match="protected"):
        audit.check_output((audit.ROOT / name).resolve())
    with pytest.raises(ValueError, match="protected"):
        audit.check_output((audit.ROOT / name / "rerun").resolve())
