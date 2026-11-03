"""Independent equations and deliberate-corruption tests for exploration audit."""

from __future__ import annotations

import json
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
from window_training import verify as audit  # noqa: E402


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([[0.2, -0.3], [0.6, 0.1], [-0.5, 0.9]]),
        np.array([[0.7, 0.4], [-0.2, 0.8], [0.2, -0.6]]),
        np.array([[0.0, 0.1, 0.15], [0.2, 0.0, 0.05], [0.1, 0.15, 0.25]]),
        np.array([[0.3, 0.7, 0.2], [0.6, 0.1, 0.3], [0.2, 0.5, 0.1]]),
    )


def test_independent_gradient_matches_every_finite_difference_component() -> None:
    center, context, positive, negative = fixture()
    values = np.stack([center, context]).ravel()
    actual = audit.objective(center, context, positive, negative)
    numerical = np.zeros_like(values)
    step = 1e-6
    for index in range(len(values)):
        perturbed = values.copy()
        perturbed[index] += step
        upper = audit.objective(*perturbed.reshape(2, 3, 2), positive, negative)[
            "expected_loss"
        ]
        perturbed[index] -= 2 * step
        lower = audit.objective(*perturbed.reshape(2, 3, 2), positive, negative)[
            "expected_loss"
        ]
        numerical[index] = (upper - lower) / (2 * step)
    np.testing.assert_allclose(actual["gradient"], numerical, rtol=1e-7, atol=1e-9)
    rows = positive.sum(axis=1)
    assert actual["expected_loss"] == pytest.approx(
        rows @ actual["node_loss"] / rows.sum()
    )


def test_loss_is_finite_and_boundary_mass_exact_for_extreme_scores() -> None:
    center, context = np.array([[1000.0], [-1000.0]]), np.ones((2, 1))
    positive = np.array([[1.0, 0.0], [0.0, 1.0]])
    actual = audit.objective(center, context, positive, np.ones((2, 2)))
    assert np.isfinite(actual["expected_loss"])
    assert np.isfinite(actual["gradient"]).all()
    assert actual["positive_mass_outside_update_window"] == 1
    assert actual["negative_mass_outside_update_window"] == 1


def test_full_svd_norm_matches_matrix_square_root_definition() -> None:
    center, context, _, _ = fixture()
    scores = center @ context.T
    eigenvalues, basis = np.linalg.eigh(scores @ scores.T)
    square_root = (basis * np.sqrt(np.maximum(eigenvalues, 0))) @ basis.T
    np.testing.assert_allclose(
        audit.full_svd_norm(scores) ** 2, np.diag(square_root), atol=1e-8
    )
    np.testing.assert_array_equal(audit.full_svd_norm(np.zeros((3, 3))), np.zeros(3))


def test_unrestricted_bound_handles_zero_positive_support() -> None:
    center, context, positive, negative = fixture()
    bound = audit.infimum(positive, negative)
    assert bound < audit.objective(center, context, positive, negative)["expected_loss"]
    scores = np.full(positive.shape, -1000.0)
    support = positive > 0
    scores[support] = np.log(positive[support] / negative[support])
    achieved = audit.objective(scores, np.eye(3), positive, negative)
    assert achieved["expected_loss"] == pytest.approx(bound, abs=1e-12)


def test_direct_finite_corpus_pairs_include_self_contexts() -> None:
    result = audit.unfiltered_pairs([[0, 0, 1], [1, 2, 0]], 3)
    np.testing.assert_array_equal(result, [[2, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert result.sum() == 8


@pytest.mark.parametrize("table", [[2, 4, 7], [0, 0, 7], [1, 6, 7]])
def test_discrete_noise_matches_exhaustive_small_bit_space(table: list[int]) -> None:
    cdf = np.array(table)
    slots = [np.searchsorted(cdf, value % cdf[-1], side="left") for value in range(16)]
    expected = np.bincount(slots, minlength=3) / 16
    np.testing.assert_array_equal(audit.discrete_proposal(cdf, bits=4), expected)


def test_curve_binning_retains_ties_and_single_degree() -> None:
    degree = np.array([1, 1, 2, 4, 8, 8])
    result = audit.curve(degree, np.array([1.0, 3.0, 5.0, 4.0, 1.0, 3.0]))
    assert sum(result["node_counts"]) == len(degree)
    assert result["interior_peak"]
    assert result["highest_minus_lowest"] == 0
    single = audit.curve(np.ones(3), np.array([1.0, 2.0, 3.0]))
    assert single["means"] == [2]
    assert single["highest_minus_penultimate"] is None
    with pytest.raises(AssertionError, match="Invalid node readout"):
        audit.curve(degree, np.ones(2))


@pytest.mark.parametrize(
    "field",
    [
        "raw_norm",
        "balanced_norm",
        "expected_loss",
        "node_loss",
        "gradient_rms",
        "input_sha256",
    ],
)
def test_independent_factor_audit_rejects_corrupted_readouts(field: str) -> None:
    center, context, positive, negative = fixture()
    metrics = audit.objective(center, context, positive, negative)
    snapshot = {
        "input_sha256": audit.array_sha(center),
        "output_sha256": audit.array_sha(context),
        "raw_norm": np.linalg.norm(center, axis=1),
        "balanced_norm": audit.full_svd_norm(center @ context.T),
        **{key: metrics[key] for key in ("expected_loss", "node_loss", "gradient_rms")},
    }
    audit.verify_factor_snapshot(
        snapshot, center, context, positive, negative, "gradient_rms"
    )
    snapshot[field] = (
        "changed" if field == "input_sha256" else np.asarray(snapshot[field]) + 1
    )
    with pytest.raises(AssertionError):
        audit.verify_factor_snapshot(
            snapshot, center, context, positive, negative, "gradient_rms"
        )


def test_hash_verification_rejects_changed_completed_artifact(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"complete": True}))
    manifest = {path.name: audit.sha(path)}
    audit.verify_hashes(tmp_path, manifest)
    path.write_text(json.dumps({"complete": False}))
    with pytest.raises(AssertionError, match="Changed artifact"):
        audit.verify_hashes(tmp_path, manifest)


def test_report_audit_checks_graph_weighting_and_standard_errors() -> None:
    degrees = [
        np.repeat(
            2.0 ** np.arange(8),
            [seed + 1 if index % 2 == 0 else 5 - seed for index in range(8)],
        )
        for seed in range(5)
    ]
    values = [(np.log2(degree) + 1) * (seed + 1) for seed, degree in enumerate(degrees)]

    def shape(series: list[float]) -> dict[str, object]:
        return {
            "peak_bin_zero_based": len(series) - 1,
            "interior_peak": False,
            "high_minus_low": series[-1] - series[0],
            "high_minus_penultimate": series[-1] - series[-2],
        }

    by_seed = [[(seed + 1.0) * (index + 1) for index in range(8)] for seed in range(5)]
    bins = []
    for index in range(8):
        series = [row[index] for row in by_seed]
        mean = sum(series) / 5
        variance = sum((value - mean) ** 2 for value in series) / 4
        bins.append(
            {
                "degree": 2.0**index,
                "counts_by_seed": [
                    seed + 1 if index % 2 == 0 else 5 - seed for seed in range(5)
                ],
                "values_by_seed": series,
                "mean": mean,
                "standard_error": np.sqrt(variance / 5),
            }
        )
    record = {
        "bins": bins,
        "per_graph": [
            {"seed": seed, "values": row, **shape(row)}
            for seed, row in enumerate(by_seed)
        ],
        "mean_curve": shape([3.0 * (index + 1) for index in range(8)]),
    }
    audit.validate_report_curve(record, degrees, values)
    record["bins"][0]["mean"] = np.average(
        bins[0]["values_by_seed"], weights=bins[0]["counts_by_seed"]
    )
    with pytest.raises(AssertionError, match="Equal graph-weighted mean"):
        audit.validate_report_curve(record, degrees, values)


def test_existing_repeat_certificate_rejects_changes(tmp_path: Path) -> None:
    original, repeat = tmp_path / "original", tmp_path / "repeat"
    for directory in (original, repeat):
        (directory / "schedule/seed0").mkdir(parents=True)
        for index in range(17):
            (directory / "schedule/seed0" / f"part{index}.json").write_text(str(index))
        for folder, count in (("graphs", 1), ("runs", 16), ("factors", 8)):
            (directory / "windows" / folder).mkdir(parents=True)
            for index in range(count):
                (directory / "windows" / folder / f"seed0_part{index}").write_text(
                    str(index)
                )
    certificate = audit.compare_repeat(original, repeat)
    assert certificate["byte_identical_scientific_artifacts"] == 42
    assert len(certificate["artifact_sha256"]) == 42
    assert all(not Path(name).is_absolute() for name in certificate["artifact_sha256"])
    (repeat / "windows/factors/seed0_part0").write_text("changed")
    with pytest.raises(AssertionError, match="Fresh repeat differs"):
        audit.compare_repeat(original, repeat)
