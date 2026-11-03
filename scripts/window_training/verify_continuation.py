"""Independently audit fixed-noise balanced-restart continuations.

Inputs: completed continuation artifacts, frozen source/runtime manifests and
original window endpoints. Checks score preservation, targets, full-SVD factor
readouts, projected gradients, every stopping gate, and paired control budgets.
Only final factors are saved: intermediate score-displacement/invariance scalars
receive consistency checks rather than complete matrix reconstruction.
Run: python3 scripts/window_training/verify_continuation.py [--partial]
--repeat --seeds 0 recreates all ten seed-zero conditions in a fresh directory;
--compare-repeat PATH checks an existing repeat. JSON records go to stdout.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from window_training import verify as audit  # noqa: E402


def load_factors(path: Path) -> np.ndarray:
    return np.load(io.BytesIO(gzip.decompress(path.read_bytes())), allow_pickle=False)


def common_groups(degrees: list[np.ndarray]) -> list[np.ndarray]:
    boundaries = np.unique(
        np.percentile(np.log(np.concatenate(degrees)), np.arange(9) * 12.5)
    )
    return [
        np.digitize(np.log(degree), boundaries[1:-1], right=True) for degree in degrees
    ]


def curve(norms: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    values = np.array([norms[groups == label].mean() for label in np.unique(groups)])
    peak = int(np.argmax(values))
    return {
        "values": values.tolist(),
        "peak_bin_zero_based": peak,
        "interior_peak": 0 < peak < len(values) - 1,
        "high_minus_low": float(values[-1] - values[0]),
        "high_minus_penultimate": float(values[-1] - values[-2]),
    }


def full_score_diagnostics(
    u: np.ndarray, v: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> dict[str, Any]:
    """Use a full score SVD, independent of the runner's small QR-core SVD."""
    scores = u @ v.T
    left, singular, right_t = np.linalg.svd(scores, full_matrices=False)
    rank = u.shape[1]
    ub = left[:, :rank] * np.sqrt(singular[:rank])
    vb = right_t[:rank].T * np.sqrt(singular[:rank])
    audit.compare(ub @ vb.T, scores, "Full-SVD score preservation", 1e-8)
    audit.compare(ub.T @ ub, vb.T @ vb, "Balanced factor Grams", 1e-8)
    metrics = audit.objective(ub, vb, positive, negative)
    original_metrics = audit.objective(u, v, positive, negative)
    audit.compare(
        metrics["expected_loss"],
        original_metrics["expected_loss"],
        "Score-preserving loss bridge",
        1e-10,
    )
    derivative = (positive + negative) * np.exp(-np.logaddexp(0, -scores)) - positive
    active = singular[:rank] > np.finfo(float).eps * len(u) * singular[0]
    a, b = left[:, :rank][:, active], right_t[:rank].T[:, active]
    normal_left = derivative - a @ (a.T @ derivative)
    projected = derivative - normal_left + (normal_left @ b) @ b.T
    return {
        "scores": scores,
        "expected_loss": original_metrics["expected_loss"],
        "balanced_gradient_max": metrics["gradient_max"],
        "balanced_gradient_rms": metrics["gradient_rms"],
        "projected_score_gradient_frobenius": float(np.linalg.norm(projected)),
        "numerical_rank": int(np.count_nonzero(active)),
        "raw_norm": np.linalg.norm(u, axis=1),
        "balanced_norm": np.sqrt(np.sum(left**2 * singular[None, :], axis=1)),
        "node_loss": original_metrics["node_loss"],
        "score_singular_values": singular[:rank],
        "score_rms": float(np.linalg.norm(scores) / len(u)),
        "center_frobenius": float(np.linalg.norm(u)),
        "context_frobenius": float(np.linalg.norm(v)),
        "score_quantiles": {
            name: np.quantile(scores[mask], [0, 0.01, 0.5, 0.99, 1]).tolist()
            if mask.any()
            else None
            for name, mask in (("positive", positive > 0), ("nonedge", positive == 0))
        },
    }


def verify_snapshot(
    record: dict[str, Any],
    u: np.ndarray,
    v: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    actual = full_score_diagnostics(u, v, positive, negative)
    for name in actual:
        if name in {"scores", "score_quantiles"}:
            continue
        audit.compare(
            record[name],
            actual[name],
            f"Independent continuation metric: {name}",
            1e-8 if name == "balanced_norm" else 1e-10,
        )
    for name, quantiles in actual["score_quantiles"].items():
        if quantiles is None:
            audit.require(
                record["score_quantiles"][name] is None,
                "Quantiles invented for empty support",
            )
        else:
            audit.compare(
                record["score_quantiles"][name],
                quantiles,
                f"Score-tail quantiles: {name}",
                1e-9,
            )
    binned = curve(actual["balanced_norm"], groups)
    audit.compare(
        record["curve"]["values"], binned["values"], "Full-SVD degree-bin means", 1e-8
    )
    audit.validate_shape(record["curve"], np.asarray(binned["values"]))
    return actual


def independent_gates(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    return {
        "gradient": record["balanced_gradient_max"] <= config["gradient_tolerance"],
        "loss": record["relative_loss_change"] <= config["loss_tolerance"],
        "scores": record["relative_score_rms_change"] <= config["stability_tolerance"],
        "node_norms": record["relative_norm_rms_change"]
        <= config["stability_tolerance"],
        "every_bin": record["max_relative_bin_change"] <= config["stability_tolerance"],
        "upper_gap": record["absolute_upper_gap_change"]
        <= config["stability_tolerance"],
    }


def verify_stopping(
    result: dict[str, Any], config: dict[str, Any], groups: np.ndarray
) -> dict[str, Any]:
    records = result["snapshots"]
    balanced = result["balanced_restart"]
    limit = config["max_blocks"] if balanced else config["control_blocks"]
    audit.require(
        [row["block"] for row in records] == list(range(len(records))),
        "Missing/reordered continuation block",
    )
    audit.require(
        config["control_blocks"] <= len(records) - 1 <= limit,
        "Invalid number of continuation blocks",
    )
    audit.require(records[0]["cumulative_iterations"] == 0, "Nonzero stage-zero effort")
    iterations, evaluations, streak = 0, 0, 0
    for index, row in enumerate(records):
        for name in ("raw_norm", "balanced_norm", "node_loss"):
            values = np.asarray(row[name])
            audit.require(
                values.shape == groups.shape
                and np.isfinite(values).all()
                and np.all(values >= 0),
                f"Invalid continuation node readout: {name}",
            )
        audit.compare(
            np.mean(row["node_loss"]),
            row["expected_loss"],
            "Equal-center node-loss average",
        )
        audit.compare(
            np.linalg.norm(row["raw_norm"]),
            row["center_frobenius"],
            "Saved native factor norm",
        )
        singular = np.asarray(row["score_singular_values"])
        audit.require(
            singular.shape == (config["rank"],)
            and np.isfinite(singular).all()
            and np.all(singular >= 0)
            and np.all(np.diff(singular) <= 1e-10),
            "Invalid score singular values",
        )
        audit.compare(
            np.linalg.norm(row["balanced_norm"]) ** 2,
            np.sum(singular),
            "Balanced norm/singular-value identity",
        )
        audit.compare(
            np.linalg.norm(singular) / len(groups),
            row["score_rms"],
            "Score RMS/singular-value identity",
        )
        binned = curve(np.asarray(row["balanced_norm"]), groups)
        audit.compare(binned["values"], row["curve"]["values"], "Saved block bin means")
        audit.validate_shape(row["curve"], np.asarray(binned["values"]))
        if index == 0:
            continue
        previous = records[index - 1]
        optimizer = row["optimizer"]
        audit.require(
            0 <= optimizer["iterations"] <= config["block_iterations"],
            "Block exceeded iteration cap",
        )
        audit.require(
            optimizer["evaluations"] >= max(1, optimizer["iterations"]),
            "Invalid actual evaluation count",
        )
        iterations += optimizer["iterations"]
        evaluations += optimizer["evaluations"]
        audit.require(
            row["cumulative_iterations"] == iterations,
            "Cumulative iteration accounting mismatch",
        )
        audit.require(
            row["expected_loss"] <= previous["expected_loss"] + 1e-9,
            "Accepted continuation loss increased",
        )
        delta = np.asarray(row["balanced_norm"]) - previous["balanced_norm"]
        old_bins = np.asarray(previous["curve"]["values"])
        expected = {
            "absolute_norm_rms_change": np.linalg.norm(delta) / np.sqrt(len(delta)),
            "relative_norm_rms_change": np.linalg.norm(delta)
            / max(1.0, np.linalg.norm(previous["balanced_norm"])),
            "relative_loss_change": abs(
                row["expected_loss"] - previous["expected_loss"]
            )
            / max(1.0, abs(previous["expected_loss"])),
            "max_relative_bin_change": np.max(
                np.abs(np.asarray(binned["values"]) - old_bins)
                / np.maximum(1.0, np.abs(old_bins))
            ),
            "absolute_upper_gap_change": abs(
                binned["high_minus_penultimate"]
                - previous["curve"]["high_minus_penultimate"]
            ),
            "relative_score_rms_change": row["absolute_score_rms_change"]
            * len(groups)
            / max(1.0, previous["score_rms"] * len(groups)),
        }
        for name, value in expected.items():
            audit.compare(
                row[name], value, f"Block stability diagnostic: {name}", 1e-10
            )
        displacement = row["absolute_score_rms_change"]
        audit.require(
            abs(row["score_rms"] - previous["score_rms"]) - 1e-8
            <= displacement
            <= row["score_rms"] + previous["score_rms"] + 1e-8,
            "Impossible saved score displacement",
        )
        if balanced:
            error = row["balancing"]["relative_score_error"]
            audit.require(
                np.isfinite(error) and 0 <= error <= 1e-8,
                "Restart failed score preservation",
            )
        else:
            audit.require(row["balancing"] == {}, "Raw control unexpectedly balanced")
        gates = independent_gates(row, config)
        audit.require(
            row["stability_gates"] == gates,
            "Stored stopping gates differ from frozen rules",
        )
        streak = streak + 1 if all(gates.values()) else 0
        if balanced and streak >= 2:
            audit.require(
                index == len(records) - 1, "Continued after two stable checkpoints"
            )
    audit.require(
        result["stability_reached"] == (streak >= 2), "Wrong stability declaration"
    )
    if balanced:
        audit.require(
            streak >= 2 or len(records) - 1 == limit,
            "Stopped early without required stability",
        )
    else:
        audit.require(
            len(records) - 1 == config["control_blocks"],
            "Raw control did not use its fixed block cap",
        )
    return {
        "blocks": len(records) - 1,
        "actual_iterations": iterations,
        "actual_evaluations": evaluations,
        "stability_reached": result["stability_reached"],
        "consecutive_stable_checkpoints": streak,
        "failed_final_gates": [
            name
            for name, passed in records[-1]["stability_gates"].items()
            if not passed
        ],
    }


def verify(output: Path, partial: bool = False) -> dict[str, Any]:
    plan = audit.verify_plan(output)
    config = plan["config"]
    reference = ROOT / config["reference"]
    audit.require(
        audit.sha(reference / "completion.json") == plan["reference_completion_sha256"],
        "Original endpoint manifest changed",
    )
    audit.verify_hashes(reference, audit.read_json(reference / "completion.json"))
    audit.verify_plan(reference)
    completion = output / "completion.json"
    if completion.exists():
        audit.verify_hashes(output, audit.read_json(completion))
    else:
        audit.require(partial, "Continuation is incomplete")
    expected = {
        f"seed{seed}_L{window}_init{offset}_{method}"
        for seed in config["seeds"]
        for window in config["windows"]
        for offset in config["offsets"]
        for method in (["balanced", "raw"] if window == 10 else ["balanced"])
    }
    markers = sorted(output.glob("*.sha256"))
    audit.require(
        {path.stem for path in markers} <= expected, "Unexpected continuation condition"
    )
    if not partial:
        audit.require(
            {path.stem for path in markers} == expected,
            "Missing continuation condition",
        )
    graphs = [
        audit.read_json(reference / f"graphs/seed{seed}.json")
        for seed in config["seeds"]
    ]
    degrees = [np.asarray(graph["degree"]) for graph in graphs]
    group_labels = common_groups(degrees)
    kernels = {}
    endpoints, paired = [], {}
    for marker in markers:
        audit.verify_hashes(output, audit.read_json(marker))
        result = audit.read_json(marker.with_suffix(".json.gz"))
        seed, window = result["seed"], result["window"]
        method = "balanced" if result["balanced_restart"] else "raw"
        name = f"seed{seed}_L{window}_init{result['initialization'] - seed}"
        audit.require(
            marker.stem == f"{name}_{method}",
            "Saved continuation identity differs from filename",
        )
        old_path = reference / "runs" / f"{name}.json.gz"
        old = audit.read_json(old_path)
        audit.require(
            audit.sha(old_path) == result["reference_run_sha256"],
            "Wrong original endpoint",
        )
        initial_path = reference / old["factor_file"]
        audit.require(
            audit.sha(initial_path)
            == old["factor_file_sha256"]
            == result["input_factor_sha256"],
            "Original factors changed",
        )
        if seed not in kernels:
            adjacency, _, degree = audit.graph_arrays(
                audit.configuration_graph(config["nodes"], config["attachment"], seed)
            )
            audit.compare(degree, degrees[seed], "Regenerated graph degree", 0)
            audit.require(
                audit.array_sha(adjacency) == graphs[seed]["graph_sha256"],
                "Regenerated graph differs",
            )
            kernels[seed] = audit.reconstruct_windows(
                adjacency, degree, config["windows"]
            )
        degree, groups = degrees[seed], group_labels[seed]
        conditional = kernels[seed][window]
        positive = conditional / len(degree)
        noise = degree / degree.sum()
        negative = np.broadcast_to(
            config["negatives"] * noise[None, :] / len(degree), positive.shape
        )
        audit.require(
            result["graph_sha256"] == graphs[seed]["graph_sha256"],
            "Continuation graph mismatch",
        )
        audit.require(
            audit.array_sha(conditional)
            == graphs[seed]["windows"][str(window)]["conditional_sha256"],
            "Context kernel changed",
        )
        audit.require(
            audit.array_sha(positive)
            == result["positive_sha256"]
            == old["positive_weights_sha256"],
            "Positive target changed",
        )
        audit.require(
            audit.array_sha(negative)
            == result["negative_sha256"]
            == old["negative_weights_sha256"],
            "Negative target changed",
        )
        effort = verify_stopping(result, config, groups)
        initial, factors = (
            load_factors(initial_path),
            load_factors(output / result["factor_file"]),
        )
        audit.require(
            initial.shape == factors.shape == (2, config["nodes"], config["rank"]),
            "Wrong continuation factor shape",
        )
        stage_zero = verify_snapshot(
            result["snapshots"][0], *initial, positive, negative, groups
        )
        for key in ("raw_norm", "balanced_norm", "node_loss", "expected_loss"):
            audit.compare(
                stage_zero[key],
                old["snapshots"][-1][key],
                f"Original endpoint/stage-zero bridge: {key}",
                1e-8,
            )
        final = verify_snapshot(
            result["snapshots"][-1], *factors, positive, negative, groups
        )
        lower = audit.infimum(positive, negative)
        audit.compare(
            lower,
            graphs[seed]["windows"][str(window)]["free_score"][
                "unrestricted_loss_infimum"
            ],
            "Stored unrestricted lower bound",
            1e-12,
        )
        audit.require(
            final["expected_loss"] >= lower - 1e-10, "Loss below unrestricted infimum"
        )
        endpoint = {
            "seed": seed,
            "window": window,
            "initialization": result["initialization"],
            "method": method,
            **effort,
            "initial_loss": stage_zero["expected_loss"],
            "final_loss": final["expected_loss"],
            "balanced_gradient_max": final["balanced_gradient_max"],
            "projected_score_gradient_frobenius": final[
                "projected_score_gradient_frobenius"
            ],
            "numerical_rank": final["numerical_rank"],
            "unrestricted_loss_infimum": lower,
            "objective_suboptimality_upper_bound": final["expected_loss"] - lower,
            "curve": curve(final["balanced_norm"], groups),
            "support_is_full": bool(np.all(positive > 0)),
        }
        endpoints.append(endpoint)
        if window == 10:
            paired[seed, result["initialization"], method] = result
    controls = []
    for seed in config["seeds"]:
        for offset in config["offsets"]:
            key = seed, seed + offset
            if (*key, "raw") not in paired or (*key, "balanced") not in paired:
                continue
            raw, balanced = paired[*key, "raw"], paired[*key, "balanced"]
            audit.require(
                raw["snapshots"][0] == balanced["snapshots"][0],
                "Coordinate controls started from different states",
            )
            for field in (
                "reference_run_sha256",
                "input_factor_sha256",
                "graph_sha256",
                "positive_sha256",
                "negative_sha256",
            ):
                audit.require(
                    raw[field] == balanced[field],
                    f"Coordinate controls not paired: {field}",
                )
            blocks = config["control_blocks"]
            controls.append(
                {
                    "seed": seed,
                    "initialization": seed + offset,
                    "matched_block_cap": blocks,
                    "iterations_per_block_cap": config["block_iterations"],
                    "balanced_actual_iterations": balanced["snapshots"][blocks][
                        "cumulative_iterations"
                    ],
                    "raw_actual_iterations": raw["snapshots"][blocks][
                        "cumulative_iterations"
                    ],
                    "balanced_actual_evaluations": sum(
                        row["optimizer"]["evaluations"]
                        for row in balanced["snapshots"][1 : blocks + 1]
                    ),
                    "raw_actual_evaluations": sum(
                        row["optimizer"]["evaluations"]
                        for row in raw["snapshots"][1 : blocks + 1]
                    ),
                }
            )
    if not partial:
        audit.require(len(controls) == 10, "Missing matched coordinate control")
    record = {
        "verified_conditions": len(markers),
        "expected_conditions": len(expected),
        "endpoints": endpoints,
        "paired_controls": controls,
        "scope": (
            "Final and stage-zero matrices independently reconstructed; "
            "intermediate saved node diagnostics/stopping logic checked. "
            "Intermediate score displacements and restart errors receive "
            "consistency checks because their factors were not saved."
        ),
        "provenance": {
            "verification_source_sha256": {
                str(path.relative_to(ROOT)): audit.sha(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("verify.py"),
                    ROOT / "tests/test_continuation_verification.py",
                )
            },
            "input_manifest_sha256": {
                name: audit.sha(output / name)
                for name in ("plan.json", "completion.json")
                if (output / name).exists()
            },
        },
    }
    if not partial and (output / "report/manifest.json").exists():
        record["report"] = verify_report(output, endpoints)
        record["provenance"]["input_manifest_sha256"]["report/manifest.json"] = (
            audit.sha(output / "report/manifest.json")
        )
    return record


def verify_drop(record: dict[str, Any], curve_data: dict[str, Any]) -> None:
    gaps = np.array([row["high_minus_penultimate"] for row in curve_data["per_graph"]])
    preceding = np.array([row["values"][-2] for row in curve_data["per_graph"]])
    audit.require(np.all(preceding > 0), "Fractional drop denominator must be positive")
    fractions = -gaps / preceding
    for key, actual in {
        "absolute_by_graph": gaps,
        "absolute_mean": gaps.mean(),
        "absolute_se": gaps.std(ddof=1) / np.sqrt(len(gaps)),
        "fractional_drop_by_graph": fractions,
        "fractional_mean": fractions.mean(),
        "fractional_se": fractions.std(ddof=1) / np.sqrt(len(fractions)),
    }.items():
        audit.compare(record[key], actual, f"Paired graph-level drop: {key}", 1e-12)


def verify_report(output: Path, endpoints: list[dict[str, Any]]) -> dict[str, int]:
    directory = output / "report"
    manifest = audit.read_json(directory / "manifest.json")
    audit.require(
        manifest["script_sha256"]
        == audit.sha(Path(__file__).with_name("report_continuation.py")),
        "Continuation report source changed",
    )
    audit.require(
        manifest["input_completion_sha256"] == audit.sha(output / "completion.json"),
        "Continuation report input changed",
    )
    audit.require(
        "verification.json" not in manifest["artifact_sha256"],
        "Cyclic verification/report manifest",
    )
    audit.verify_hashes(directory, manifest["artifact_sha256"])
    data = audit.read_json(directory / "summary.json")
    config = audit.read_json(output / "plan.json")["config"]
    reference = ROOT / config["reference"]
    graphs = [
        audit.read_json(reference / f"graphs/seed{seed}.json")
        for seed in config["seeds"]
    ]
    degrees = [np.asarray(graph["degree"]) for graph in graphs]
    audit.compare(data["degrees"], degrees, "Report raw graph degrees", 0)
    runs = [audit.read_json(path) for path in sorted(output.glob("*.json.gz"))]
    audit.require(
        len(data["windows"]) == 4
        and len(data["endpoints"]) == 50
        and len(data["controls"]) == 10
        and len(data["initialization_comparison"]) == 20,
        "Incomplete continuation report",
    )

    def match(record: dict[str, Any]) -> dict[str, Any]:
        return next(
            run
            for run in runs
            if all(
                run[key] == record[key]
                for key in ("seed", "window", "initialization", "balanced_restart")
            )
        )

    for record in data["windows"]:
        window = record["window"]
        local = [
            run for run in runs if run["window"] == window and run["balanced_restart"]
        ]
        norm = [
            np.mean(
                [
                    run["snapshots"][-1]["balanced_norm"]
                    for run in local
                    if run["seed"] == seed
                ],
                axis=0,
            )
            for seed in config["seeds"]
        ]
        kl = [
            np.asarray(graph["windows"][str(window)]["context_kl"]) for graph in graphs
        ]
        for field, values in (("norm", norm), ("kl", kl)):
            audit.validate_report_curve(record[field], degrees, values)
            verify_drop(record[f"{field}_drop"], record[field])
            slopes = []
            for degree, response in zip(degrees, values, strict=True):
                centered_x = np.log(degree) - np.log(degree).mean()
                slopes.append(
                    float(
                        centered_x
                        @ (response - response.mean())
                        / (centered_x @ centered_x)
                    )
                )
            audit.compare(
                record[f"{field}_log_degree_slopes"],
                slopes,
                f"Independent {field}/log-degree slopes",
            )
        audit.require(
            record["stability_reached"]
            == sum(run["stability_reached"] for run in local),
            "Reported stability count mismatch",
        )
    for record in data["endpoints"]:
        run = match(record)
        last, initial = run["snapshots"][-1], run["snapshots"][0]
        independently_checked = next(
            row
            for row in endpoints
            if row["seed"] == run["seed"]
            and row["window"] == run["window"]
            and row["initialization"] == run["initialization"]
            and row["method"] == ("balanced" if run["balanced_restart"] else "raw")
        )
        for key in ("unrestricted_loss_infimum", "objective_suboptimality_upper_bound"):
            audit.compare(
                record[key],
                independently_checked[key],
                f"Independent rank-objective bound: {key}",
                1e-12,
            )
        for key, value in {
            "stability_reached": run["stability_reached"],
            "blocks": last["block"],
            "iterations": last["cumulative_iterations"],
            "loss": last["expected_loss"],
            "initial_loss": initial["expected_loss"],
            "gradient": last["balanced_gradient_max"],
            "final_gates": last["stability_gates"],
            "curve": last["curve"],
            "max_balanced_norm": max(last["balanced_norm"]),
            "balanced_norm": last["balanced_norm"],
            "max_relative_bin_change": last["max_relative_bin_change"],
            "absolute_upper_gap_change": last["absolute_upper_gap_change"],
        }.items():
            audit.require(record[key] == value, f"Reported endpoint changed: {key}")
    audit.require(
        len(data["progress"]) == sum(len(run["snapshots"]) for run in runs),
        "Missing continuation progress records",
    )
    for record in data["progress"]:
        run = match(record)
        snapshot = run["snapshots"][record["block"]]
        for key, value in record.items():
            if key not in {"seed", "window", "initialization", "balanced_restart"}:
                audit.require(
                    value == snapshot[key], f"Reported block diagnostic changed: {key}"
                )
    for record in data["initialization_comparison"]:
        pair = sorted(
            [
                run
                for run in runs
                if run["seed"] == record["seed"]
                and run["window"] == record["window"]
                and run["balanced_restart"]
            ],
            key=lambda run: run["initialization"],
        )
        first, second = [run["snapshots"][-1] for run in pair]
        differences = {
            "loss_difference": abs(first["expected_loss"] - second["expected_loss"]),
            "balanced_node_rms_difference": np.linalg.norm(
                np.asarray(first["balanced_norm"]) - second["balanced_norm"]
            )
            / np.sqrt(config["nodes"]),
            "bin_max_absolute_difference": np.max(
                np.abs(np.asarray(first["curve"]["values"]) - second["curve"]["values"])
            ),
            "upper_gap_difference": abs(
                first["curve"]["high_minus_penultimate"]
                - second["curve"]["high_minus_penultimate"]
            ),
        }
        for key, value in differences.items():
            audit.compare(record[key], value, f"Across-start discrepancy: {key}", 1e-12)
    for record in data["controls"]:
        raw = next(
            run
            for run in runs
            if run["seed"] == record["seed"]
            and run["initialization"] == record["initialization"]
            and not run["balanced_restart"]
        )
        balanced = next(
            run
            for run in runs
            if run["seed"] == record["seed"]
            and run["initialization"] == record["initialization"]
            and run["window"] == 10
            and run["balanced_restart"]
        )
        audit.compare(
            record["initial_loss"],
            raw["snapshots"][0]["expected_loss"],
            "Control common initial loss",
        )
        for label, run in (("raw", raw), ("balanced", balanced)):
            row = run["snapshots"][config["control_blocks"]]
            for key, field in (
                ("loss", "expected_loss"),
                ("gradient", "balanced_gradient_max"),
                ("iterations", "cumulative_iterations"),
            ):
                audit.compare(
                    record[f"{label}_{key}"],
                    row[field],
                    f"Matched block-two control: {label} {key}",
                )
    return {
        "verified_window_curves": 8,
        "verified_endpoints_and_bounds": 50,
        "verified_controls": 10,
        "verified_initialization_comparisons": 20,
        "verified_block_records": len(data["progress"]),
    }


def compare_repeat(output: Path, destination: Path, seeds: list[int]) -> dict[str, Any]:
    audit.require(
        output.resolve() != destination.resolve(), "Repeat must be a separate directory"
    )
    names = sorted(
        path.name
        for seed in seeds
        for path in output.glob(f"seed{seed}_*")
        if path.is_file()
    )
    audit.require(len(names) == 30 * len(seeds), "Missing repeat conditions")
    hashes = {}
    for name in names:
        digest = audit.sha(output / name)
        audit.require(
            audit.sha(destination / name) == digest,
            f"Repeated continuation differs: {name}",
        )
        hashes[name] = digest
    return {
        "seeds": seeds,
        "byte_identical_scientific_artifacts": len(names),
        "artifact_sha256": hashes,
    }


def repeat(output: Path, destination: Path | None, seeds: list[int]) -> dict[str, Any]:
    from window_training import continue_windows as runner

    destination = (
        destination or Path(tempfile.mkdtemp(prefix="window-continuation-repeat-"))
    ).resolve()
    audit.require(
        not destination.exists() or not any(destination.iterdir()),
        "Repeat destination must be fresh",
    )
    reference = Path(audit.read_json(output / "plan.json")["config"]["reference"])
    with redirect_stdout(sys.stderr):
        runner.run(
            runner.Config(reference=reference, output=destination), seeds, workers=1
        )
    result = compare_repeat(output, destination, seeds)
    print(f"Fresh continuation repeat retained at {destination}", file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/window_training_continuation"
    )
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--repeat", action="store_true")
    choices.add_argument("--compare-repeat", type=Path)
    parser.add_argument("--repeat-output", type=Path)
    args = parser.parse_args()
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or not set(args.seeds) <= set(range(5))
    ):
        parser.error("Choose a nonempty unique subset of graph seeds 0..4")
    with threadpool_limits(limits=1):
        result = verify(args.output, args.partial)
        if args.repeat:
            result["repeat"] = repeat(args.output, args.repeat_output, args.seeds)
        elif args.compare_repeat:
            result["repeat"] = compare_repeat(
                args.output, args.compare_repeat, args.seeds
            )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
