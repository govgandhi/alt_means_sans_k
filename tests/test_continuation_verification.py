"""Independent continuation audit, including deliberately corrupted stop records."""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from window_training import continue_windows as runner  # noqa: E402
from window_training import verify_continuation as check  # noqa: E402


@pytest.fixture
def fixture() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(88)
    u, v = rng.normal(size=(2, 6, 3)) * 0.2
    scales = np.array([0.001, 5.0, 1000.0])
    u, v = u * scales, v / scales
    positive = rng.random((6, 6))
    np.fill_diagonal(positive, 0)
    positive /= 6 * positive.sum(axis=1)[:, None]
    negative = np.full((6, 6), 5 / 36)
    groups = np.arange(6)
    return u, v, positive, negative, groups


def test_full_score_svd_independently_matches_core_svd_readouts(
    fixture: tuple[np.ndarray, ...],
) -> None:
    u, v, positive, negative, groups = fixture
    record, _ = runner.snapshot(u, v, positive, negative, groups, None, None)
    actual = check.verify_snapshot(record, u, v, positive, negative, groups)
    left, _, right = np.linalg.svd(u @ v.T, full_matrices=False)
    pu, pv = left[:, :3] @ left[:, :3].T, right[:3].T @ right[:3]
    probability = 1 / (1 + np.exp(-(u @ v.T)))
    gradient = (positive + negative) * probability - positive
    projected = gradient - (np.eye(6) - pu) @ gradient @ (np.eye(6) - pv)
    assert actual["projected_score_gradient_frobenius"] == pytest.approx(
        np.linalg.norm(projected), abs=1e-12
    )
    assert abs(np.sum(projected * (gradient - projected))) < 1e-12


@pytest.mark.parametrize(
    "field",
    [
        "expected_loss",
        "balanced_gradient_max",
        "balanced_gradient_rms",
        "projected_score_gradient_frobenius",
        "balanced_norm",
        "raw_norm",
        "node_loss",
        "score_singular_values",
    ],
)
def test_full_factor_audit_rejects_altered_diagnostics(
    fixture: tuple[np.ndarray, ...], field: str
) -> None:
    u, v, positive, negative, groups = fixture
    record, _ = runner.snapshot(u, v, positive, negative, groups, None, None)
    record[field] = np.asarray(record[field]) + 1
    with pytest.raises(AssertionError):
        check.verify_snapshot(record, u, v, positive, negative, groups)


def test_zero_score_numerical_rank_is_handled_without_division_by_zero() -> None:
    u, v = np.zeros((3, 2)), np.zeros((3, 2))
    actual = check.full_score_diagnostics(u, v, np.eye(3) / 3, np.full((3, 3), 5 / 9))
    assert actual["numerical_rank"] == 0
    assert actual["projected_score_gradient_frobenius"] == 0
    assert actual["balanced_gradient_max"] == 0


def make_run(
    fixture: tuple[np.ndarray, ...], balanced: bool = True
) -> tuple[dict, dict]:
    u, v, positive, negative, groups = fixture
    config = runner.Config(
        rank=3,
        block_iterations=2,
        max_blocks=4,
        gradient_tolerance=1e9,
        loss_tolerance=1e9,
        stability_tolerance=1e9,
    )
    result, _, _ = runner.optimize(u, v, positive, negative, groups, config, balanced)
    return result, asdict(config)


def test_stop_requires_two_consecutive_checkpoints(
    fixture: tuple[np.ndarray, ...],
) -> None:
    result, config = make_run(fixture)
    actual = check.verify_stopping(result, config, fixture[-1])
    assert actual["blocks"] == 2 and actual["stability_reached"]
    changed = copy.deepcopy(result)
    changed["snapshots"][1]["balanced_gradient_max"] = 2e9
    changed["snapshots"][1]["stability_gates"]["gradient"] = False
    with pytest.raises(AssertionError, match="Wrong stability declaration"):
        check.verify_stopping(changed, config, fixture[-1])
    changed["stability_reached"] = False
    with pytest.raises(AssertionError, match="Stopped early"):
        check.verify_stopping(changed, config, fixture[-1])


def test_gate_and_score_preservation_mutations_are_rejected(
    fixture: tuple[np.ndarray, ...],
) -> None:
    result, config = make_run(fixture)
    changed = copy.deepcopy(result)
    changed["snapshots"][1]["stability_gates"]["loss"] = False
    with pytest.raises(AssertionError, match="Stored stopping gates"):
        check.verify_stopping(changed, config, fixture[-1])
    changed = copy.deepcopy(result)
    changed["snapshots"][1]["balancing"]["relative_score_error"] = 0.01
    with pytest.raises(AssertionError, match="Restart failed score preservation"):
        check.verify_stopping(changed, config, fixture[-1])
    changed = copy.deepcopy(result)
    changed["snapshots"][1]["absolute_score_rms_change"] += 1
    with pytest.raises(AssertionError):
        check.verify_stopping(changed, config, fixture[-1])


def test_raw_control_cap_and_actual_work_are_separate(
    fixture: tuple[np.ndarray, ...],
) -> None:
    result, config = make_run(fixture, False)
    audited = check.verify_stopping(result, config, fixture[-1])
    assert audited["blocks"] == 2
    assert audited["actual_iterations"] == sum(
        row["optimizer"]["iterations"] for row in result["snapshots"][1:]
    )
    assert audited["actual_evaluations"] == sum(
        row["optimizer"]["evaluations"] for row in result["snapshots"][1:]
    )
    changed = copy.deepcopy(result)
    changed["snapshots"][1]["balancing"] = {"relative_score_error": 0}
    with pytest.raises(AssertionError, match="Raw control unexpectedly balanced"):
        check.verify_stopping(changed, config, fixture[-1])


def test_repeat_requires_all_conditions_and_identical_bytes(tmp_path: Path) -> None:
    original, repeated = tmp_path / "original", tmp_path / "repeated"
    original.mkdir()
    repeated.mkdir()
    for index in range(30):
        for directory in (original, repeated):
            (directory / f"seed0_part{index}").write_text(str(index))
    result = check.compare_repeat(original, repeated, [0])
    assert result["byte_identical_scientific_artifacts"] == 30
    with pytest.raises(AssertionError, match="separate directory"):
        check.compare_repeat(original, original, [0])
    (repeated / "seed0_part0").write_text("changed")
    with pytest.raises(AssertionError, match="Repeated continuation differs"):
        check.compare_repeat(original, repeated, [0])


def test_paired_fractional_drop_averages_graph_ratios_not_pooled_ratio() -> None:
    gaps = np.array([-0.1, -0.2, 0.1, -0.4, -0.1])
    preceding = np.arange(2.0, 7.0)
    fractions = -gaps / preceding
    curves = {
        "per_graph": [
            {"high_minus_penultimate": gap, "values": [before, before + gap]}
            for before, gap in zip(preceding, gaps, strict=True)
        ]
    }
    record = {
        "absolute_by_graph": gaps,
        "absolute_mean": gaps.mean(),
        "absolute_se": gaps.std(ddof=1) / np.sqrt(5),
        "fractional_drop_by_graph": fractions,
        "fractional_mean": fractions.mean(),
        "fractional_se": fractions.std(ddof=1) / np.sqrt(5),
    }
    check.verify_drop(record, curves)
    record["fractional_mean"] = -gaps.mean() / preceding.mean()
    with pytest.raises(AssertionError, match="fractional_mean"):
        check.verify_drop(record, curves)
