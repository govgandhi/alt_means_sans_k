"""Audit scale-normalized balanced radii from the 40 frozen final window fits.

Inputs: original and continued window artifacts and their frozen manifests.
Outputs: every node's readout, condition and paired-start tables, full-support
unrestricted-target diagnostics, and a source/runtime/input/output hash manifest.
No training is performed. Normalize each start before averaging starts within a
graph; uncertainty is the standard error across the five independent graphs.
Run with the pinned exploration environment:
    python3 scripts/window_training/normalized_profiles.py [--output DIRECTORY]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from window_training import verify as audit  # noqa: E402
from window_training import verify_continuation as continuation  # noqa: E402


@dataclass(frozen=True)
class Paths:
    source: Path = ROOT / "results/window_training_continuation"
    output: Path = ROOT / "results/norm_identifiability/empirical"


def normalized_radius(radius: np.ndarray) -> tuple[float, np.ndarray | None]:
    """A zero radius vector has no scale-normalized shape."""
    if radius.ndim != 1 or not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError("Expected a finite, nonnegative node-radius vector")
    rms = float(np.linalg.norm(radius) / np.sqrt(len(radius)))
    return rms, radius / rms if rms > 0 else None


def score_loss(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    return float(
        np.sum(positive * np.logaddexp(0, -scores) + negative * np.logaddexp(0, scores))
        / positive.sum()
    )


def score_infimum(positive: np.ndarray, negative: np.ndarray) -> float:
    """Evaluate each scalar optimum, including its zero-positive limiting loss."""
    mask = positive > 0
    optimum = np.log(positive[mask] / negative[mask])
    return float(
        np.sum(
            positive[mask] * np.logaddexp(0, -optimum)
            + negative[mask] * np.logaddexp(0, optimum)
        )
        / positive.sum()
    )


def balanced_radius(scores: np.ndarray) -> np.ndarray:
    left, singular, _ = np.linalg.svd(scores, full_matrices=False)
    return np.sqrt((left**2) @ singular)


def binned(radius: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    result = continuation.curve(radius, groups)
    result["bin_labels"] = np.unique(groups).tolist()
    result["node_counts"] = [
        int(np.sum(groups == label)) for label in np.unique(groups)
    ]
    return result


def shape_difference(
    first: np.ndarray, second: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    delta = first - second
    bins = (
        np.asarray(binned(first, groups)["values"]) - binned(second, groups)["values"]
    )
    return {
        "node_rms": float(np.linalg.norm(delta) / np.sqrt(len(delta))),
        "max_absolute_bin_difference": float(np.max(np.abs(bins))),
    }


def full_support_bound(
    positive: np.ndarray, negative: np.ndarray, gap: float, target_rms: float
) -> dict[str, Any]:
    """Conservative error to the unrestricted target, not a rank-r certificate.

    Normalize the weights by positive mass, consistently with score_loss. The
    Hessian lower scale is h=min ab/(a+b). From psi(s)>=s*s/(s+2), score error is at
    most E=(c+sqrt(c*c+8*c))/2, c=gap/h. Hermitian dilation and the square-root
    inequality give RMS radius error e=(2/n)**.25*sqrt(E). Normalization yields
    2e/target_rms, capped by the geometric maximum sqrt(2) for nonnegative profiles.
    """
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError("The finite-target bound requires full positive support")
    if gap < -1e-10 or target_rms < 0:
        raise ValueError("Invalid gap or target RMS")
    mass = positive.sum()
    a, b = positive / mass, negative / mass
    h = float(np.min(a * b / (a + b)))
    c = max(0.0, gap) / h
    error = float((c + np.sqrt(c * c + 8 * c)) / 2)
    radius_error = float((2 / len(a)) ** 0.25 * np.sqrt(error))
    normalized_error = (
        None
        if target_rms == 0
        else float(min(np.sqrt(2), 2 * radius_error / target_rms))
    )
    return {
        "minimum_curvature_scale_h": h,
        "score_frobenius_error_bound": error,
        "radius_rms_error_bound": radius_error,
        "normalized_profile_rms_error_bound": normalized_error,
        "status": (
            "undefined target shape: zero target RMS"
            if target_rms == 0
            else "not certified beyond trivial geometric bound"
            if normalized_error == np.sqrt(2)
            else "nontrivial target bound; no accuracy threshold imposed"
        ),
        "scope": (
            "Unrestricted finite score target; does not certify a unique "
            "rank-constrained optimum."
        ),
    }


def aggregate_starts(
    rows: list[dict[str, Any]], seeds: list[int], field: str
) -> dict[str, Any]:
    """Start-normalized curves enter here; equal graph weights determine SE."""
    graph_values = np.array(
        [
            np.mean(
                [row[field]["values"] for row in rows if row["seed"] == seed], axis=0
            )
            for seed in seeds
        ]
    )
    gaps = graph_values[:, -1] - graph_values[:, -2]
    low_gaps = graph_values[:, -1] - graph_values[:, 0]
    return {
        "seeds": seeds,
        "values_by_graph": graph_values.tolist(),
        "mean": graph_values.mean(axis=0).tolist(),
        "se": (graph_values.std(axis=0, ddof=1) / np.sqrt(len(seeds))).tolist(),
        "upper_gap_by_graph": gaps.tolist(),
        "upper_gap_mean": float(gaps.mean()),
        "upper_gap_se": float(gaps.std(ddof=1) / np.sqrt(len(seeds))),
        "high_minus_low_by_graph": low_gaps.tolist(),
        "high_minus_low_mean": float(low_gaps.mean()),
        "high_minus_low_se": float(low_gaps.std(ddof=1) / np.sqrt(len(seeds))),
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def condition_table(data: dict[str, Any]) -> str:
    lines = [
        "# Final balanced-radius profile audit",
        "",
        "All 40 frozen balanced-restart fits are included; raw-coordinate controls "
        "are outside this comparison. Each start is normalized by its own node "
        "RMS before averaging starts within a graph. Loss gaps are upper bounds "
        "on rank-24 suboptimality, relative to the unrestricted infimum; no "
        "arbitrary small-gap cutoff is imposed. Intermediate drift uses saved "
        "diagnostics, not reconstructed intermediate factors.",
        "",
        "| Graph | Window | Start | Loss gap to free infimum | Radius RMS | "
        "Normalized upper gap | Stable stopping | Final recorded normalized "
        "drift RMS |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in data["conditions"]:
        lines.append(
            f"| {row['seed']} | {row['window']} | {row['initialization']} | "
            f"{row['loss_gap_to_unrestricted_infimum']:.9g} | "
            f"{row['radius_rms']:.8f} | "
            f"{row['normalized_curve']['high_minus_penultimate']:.8f} | "
            f"{'yes' if row['stability_reached'] else 'no'} | "
            f"{row['endpoint_drift']['normalized_profile']['node_rms']:.9g} |"
        )
    lines += [
        "",
        "## Paired starts",
        "",
        "Different saved starts are compared node by node on the same graph. "
        "Both gaps are shown: disagreement alone does not prove that two "
        "globally near-optimal rank-24 shapes differ.",
        "",
        "| Graph | Window | First loss gap | Second loss gap | "
        "Normalized node RMS difference | Maximum bin difference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["paired_starts"]:
        lines.append(
            f"| {row['seed']} | {row['window']} | {row['loss_gaps'][0]:.9g} | "
            f"{row['loss_gaps'][1]:.9g} | "
            f"{row['normalized_profile']['node_rms']:.9g} | "
            f"{row['normalized_profile']['max_absolute_bin_difference']:.9g} |"
        )
    lines += [
        "",
        "## Window five versus ten",
        "",
        "The absolute window contrast is compared with the absolute difference "
        "between the two starts' normalized upper gaps at each window on the "
        "same graph. Exceeding both is a descriptive comparison, not an "
        "optimality certificate or a statistical significance test.",
        "",
        "| Graph | Start offset | Window-five normalized upper gap | "
        "Window-ten normalized upper gap | Absolute decline weakens | "
        "Start discrepancy, window five | Start discrepancy, window ten | "
        "Window contrast exceeds both |",
        "|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for row in data["window_five_ten_pairs"]:
        exceeds = row["window_contrast_exceeds_both_start_discrepancies"]
        lines.append(
            f"| {row['seed']} | {row['initialization_offset']} | "
            f"{row['window_five_upper_gap']:.9g} | "
            f"{row['window_ten_upper_gap']:.9g} | "
            f"{'yes' if row['absolute_decline_weakens'] else 'no'} | "
            f"{row['window_five_between_start_upper_gap_difference']:.9g} | "
            f"{row['window_ten_between_start_upper_gap_difference']:.9g} | "
            f"{'yes' if exceeds else 'no'} |"
        )
    lines += [
        "",
        "## Full-support unrestricted-target bounds",
        "",
        "Only window ten has full positive support. These general bounds are "
        "supplementary: reaching the trivial normalized-profile bound is a "
        "failure to certify accuracy, not evidence of instability. The target "
        "is the unrestricted optimum, not the optimum under rank 24.",
        "",
        "| Graph | Start | Target radius RMS | Actual normalized RMS error | "
        "Radius RMS error bound | Normalized RMS error bound |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["conditions"]:
        if row["target"] is not None:
            target = row["target"]
            lines.append(
                f"| {row['seed']} | {row['initialization']} | "
                f"{target['radius_rms']:.8g} | "
                f"{target['normalized_profile_error']['node_rms']:.8g} | "
                f"{target['bound']['radius_rms_error_bound']:.8g} | "
                f"{target['bound']['normalized_profile_rms_error_bound']:.8g} |"
            )
    return "\n".join(lines) + "\n"


def analyze(paths: Paths) -> dict[str, Any]:
    plan = audit.verify_plan(paths.source)
    config = plan["config"]
    reference = ROOT / config["reference"]
    audit.verify_plan(reference)
    audit.require(
        audit.sha(reference / "completion.json") == plan["reference_completion_sha256"],
        "Changed reference completion",
    )
    audit.verify_hashes(reference, audit.read_json(reference / "completion.json"))
    audit.verify_hashes(paths.source, audit.read_json(paths.source / "completion.json"))
    graphs = {
        seed: audit.read_json(reference / f"graphs/seed{seed}.json")
        for seed in config["seeds"]
    }
    degrees = {seed: np.asarray(graphs[seed]["degree"]) for seed in config["seeds"]}
    groups = dict(
        zip(
            config["seeds"],
            continuation.common_groups(list(degrees.values())),
            strict=True,
        )
    )
    records, nodes, targets, inputs = [], [], [], {}
    cache: dict[tuple[int, int, int], tuple[np.ndarray, dict[str, Any]]] = {}
    for directory in (reference, paths.source):
        for name in ("plan.json", "completion.json"):
            path = directory / name
            inputs[str(path.relative_to(ROOT))] = audit.sha(path)
    for seed in config["seeds"]:
        graph = graphs[seed]
        graph_path = reference / f"graphs/seed{seed}.json"
        inputs[str(graph_path.relative_to(ROOT))] = audit.sha(graph_path)
        adjacency, _, degree = audit.graph_arrays(
            audit.configuration_graph(config["nodes"], config["attachment"], seed)
        )
        audit.compare(degree, degrees[seed], "Graph degree", 0)
        audit.require(
            audit.array_sha(adjacency) == graph["graph_sha256"],
            "Regenerated graph hash mismatch",
        )
        kernels = audit.reconstruct_windows(adjacency, degree, config["windows"])
        for window in config["windows"]:
            conditional = kernels[window]
            positive = conditional / len(degree)
            negative = np.broadcast_to(
                config["negatives"] * (degree / degree.sum())[None, :] / len(degree),
                positive.shape,
            )
            audit.require(
                audit.array_sha(conditional)
                == graph["windows"][str(window)]["conditional_sha256"],
                "Context hash mismatch",
            )
            lower = score_infimum(positive, negative)
            audit.compare(
                lower,
                graph["windows"][str(window)]["free_score"][
                    "unrestricted_loss_infimum"
                ],
                "Free-score infimum",
                1e-12,
            )
            finite = bool(np.all(positive > 0))
            target_radius = target_normalized = None
            if finite:
                target_radius = balanced_radius(np.log(positive / negative))
                target_rms, target_normalized = normalized_radius(target_radius)
                targets.append(
                    {
                        "seed": seed,
                        "window": window,
                        "degree": degree.tolist(),
                        "bin": groups[seed].tolist(),
                        "radius": target_radius.tolist(),
                        "normalized_radius": None
                        if target_normalized is None
                        else target_normalized.tolist(),
                        "radius_rms": target_rms,
                    }
                )
            for offset in config["offsets"]:
                name = f"seed{seed}_L{window}_init{offset}_balanced"
                marker = paths.source / f"{name}.sha256"
                audit.verify_hashes(paths.source, audit.read_json(marker))
                path = paths.source / f"{name}.json.gz"
                result = audit.read_json(path)
                factor_path = paths.source / result["factor_file"]
                inputs.update(
                    {
                        str(p.relative_to(ROOT)): audit.sha(p)
                        for p in (marker, path, factor_path)
                    }
                )
                audit.require(
                    result["seed"] == seed
                    and result["window"] == window
                    and result["initialization"] == seed + offset
                    and result["balanced_restart"],
                    "Wrong run identity",
                )
                audit.require(
                    result["graph_sha256"] == graph["graph_sha256"],
                    "Run graph hash mismatch",
                )
                for weights, field in (
                    (positive, "positive_sha256"),
                    (negative, "negative_sha256"),
                ):
                    audit.require(
                        audit.array_sha(weights) == result[field],
                        "Objective weights hash mismatch",
                    )
                factors = continuation.load_factors(factor_path)
                audit.require(
                    factors.shape == (2, config["nodes"], config["rank"]),
                    "Wrong final factor shape",
                )
                scores = factors[0] @ factors[1].T
                loss, radius = (
                    score_loss(scores, positive, negative),
                    balanced_radius(scores),
                )
                previous, final = result["snapshots"][-2:]
                audit.compare(loss, final["expected_loss"], "Final loss", 1e-11)
                audit.compare(
                    radius,
                    final["balanced_norm"],
                    "Full-SVD final balanced radius",
                    1e-8,
                )
                audit.require(loss >= lower - 1e-10, "Loss below infimum")
                rms, normalized = normalized_radius(radius)
                audit.require(
                    normalized is not None, "Zero fitted radius has undefined shape"
                )
                _, previous_normalized = normalized_radius(
                    np.asarray(previous["balanced_norm"])
                )
                audit.require(
                    previous_normalized is not None, "Undefined previous shape"
                )
                effort = continuation.verify_stopping(result, config, groups[seed])
                target = None
                if finite:
                    target = {
                        "radius_rms": target_rms,
                        "raw_curve": binned(target_radius, groups[seed]),
                        "normalized_curve": None
                        if target_normalized is None
                        else binned(target_normalized, groups[seed]),
                        "radius_rms_error": float(
                            np.linalg.norm(radius - target_radius)
                            / np.sqrt(len(radius))
                        ),
                        "normalized_profile_error": None
                        if target_normalized is None
                        else shape_difference(
                            normalized, target_normalized, groups[seed]
                        ),
                        "bound": full_support_bound(
                            positive, negative, loss - lower, target_rms
                        ),
                    }
                row = {
                    "seed": seed,
                    "window": window,
                    "initialization": seed + offset,
                    "initialization_offset": offset,
                    "rank_constraint": config["rank"],
                    "final_loss": loss,
                    "unrestricted_loss_infimum": lower,
                    "loss_gap_to_unrestricted_infimum": loss - lower,
                    "zero_positive_pairs": int(np.count_nonzero(positive == 0)),
                    "finite_unrestricted_target": finite,
                    "radius_rms": rms,
                    "raw_curve": binned(radius, groups[seed]),
                    "normalized_curve": binned(normalized, groups[seed]),
                    **effort,
                    "endpoint_drift": {
                        "provenance": (
                            "Final interval from recorded checkpoint radius "
                            "diagnostics; intermediate factors were not saved."
                        ),
                        "previous_block": previous["block"],
                        "final_block": final["block"],
                        "absolute_radius_rms_change": final["absolute_norm_rms_change"],
                        "relative_radius_norm_change": final[
                            "relative_norm_rms_change"
                        ],
                        "normalized_profile": shape_difference(
                            normalized, previous_normalized, groups[seed]
                        ),
                        "relative_loss_change": final["relative_loss_change"],
                        "recorded_relative_score_rms_change": final[
                            "relative_score_rms_change"
                        ],
                    },
                    "target": target,
                }
                records.append(row)
                cache[seed, window, offset] = normalized, row
                nodes.append(
                    {
                        "seed": seed,
                        "window": window,
                        "initialization": seed + offset,
                        "degree": degree.tolist(),
                        "bin": groups[seed].tolist(),
                        "radius": radius.tolist(),
                        "normalized_radius": normalized.tolist(),
                    }
                )
    paired, comparisons = [], []
    for seed in config["seeds"]:
        for window in config["windows"]:
            (first, row1), (second, row2) = [
                cache[seed, window, offset] for offset in config["offsets"]
            ]
            paired.append(
                {
                    "seed": seed,
                    "window": window,
                    "initializations": [row1["initialization"], row2["initialization"]],
                    "loss_gaps": [
                        row1["loss_gap_to_unrestricted_infimum"],
                        row2["loss_gap_to_unrestricted_infimum"],
                    ],
                    "normalized_profile": shape_difference(first, second, groups[seed]),
                    "interpretation": (
                        "Observed disagreement; both actual gaps are reported, "
                        "with no automatic near-optimality or "
                        "nonidentification classification."
                    ),
                }
            )
        start_discrepancies = {}
        for window in (5, 10):
            start_gaps = [
                cache[seed, window, offset][1]["normalized_curve"][
                    "high_minus_penultimate"
                ]
                for offset in config["offsets"]
            ]
            start_discrepancies[window] = abs(start_gaps[1] - start_gaps[0])
        for offset in config["offsets"]:
            five, ten = [cache[seed, window, offset][1] for window in (5, 10)]
            gap5, gap10 = [
                row["normalized_curve"]["high_minus_penultimate"] for row in (five, ten)
            ]
            comparisons.append(
                {
                    "seed": seed,
                    "initialization_offset": offset,
                    "window_five_upper_gap": gap5,
                    "window_ten_upper_gap": gap10,
                    "difference_ten_minus_five": gap10 - gap5,
                    "absolute_decline_weakens": bool(gap5 < gap10 < 0),
                    "window_five_between_start_upper_gap_difference": (
                        start_discrepancies[5]
                    ),
                    "window_ten_between_start_upper_gap_difference": (
                        start_discrepancies[10]
                    ),
                    "window_contrast_exceeds_both_start_discrepancies": bool(
                        abs(gap10 - gap5) > max(start_discrepancies.values())
                    ),
                    "window_five_stable": five["stability_reached"],
                    "window_ten_stable": ten["stability_reached"],
                }
            )
    summary = {
        "definition": (
            "radius=sqrt(diag(sqrt(M M^T))); "
            "normalized_radius=radius/sqrt(mean(radius^2)); "
            "zero RMS has undefined normalized shape."
        ),
        "aggregation": (
            "Normalize each start separately; average its bin means within "
            "each graph; average graphs equally and report graph-level SE."
        ),
        "conditions": records,
        "paired_starts": paired,
        "window_five_ten_pairs": comparisons,
        "windows": [
            {
                "window": window,
                **{
                    field: aggregate_starts(
                        [row for row in records if row["window"] == window],
                        config["seeds"],
                        field,
                    )
                    for field in ("raw_curve", "normalized_curve")
                },
            }
            for window in config["windows"]
        ],
        "scope": (
            "40 balanced final fits only; no new fitting. Final factors "
            "independently reconstructed. The general bound concerns the "
            "unrestricted target and does not establish rank-24 uniqueness. "
            "A trivial bound means accuracy was not certified, not that "
            "a fit is unstable."
        ),
    }
    paths.output.mkdir(parents=True, exist_ok=True)
    write_json(paths.output / "summary.json", summary)
    write_json(paths.output / "nodes.json", nodes)
    write_json(paths.output / "targets.json", targets)
    (paths.output / "tables.md").write_text(condition_table(summary))
    sources = [
        Path(__file__),
        Path(__file__).with_name("verify.py"),
        Path(__file__).with_name("verify_continuation.py"),
        ROOT / "tests/test_normalized_profiles.py",
    ]
    manifest = {
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package) for package in plan["packages"]
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): audit.sha(path) for path in sources
        },
        "input_sha256": inputs,
        "artifact_sha256": {
            name: audit.sha(paths.output / name)
            for name in ("summary.json", "nodes.json", "targets.json", "tables.md")
        },
        "verified_balanced_conditions": len(records),
        "node_readouts": sum(len(row["degree"]) for row in nodes),
        "finite_targets": len(targets),
        "training_performed": False,
        "reproduction": (
            "python3 scripts/window_training/normalized_profiles.py --output DIRECTORY"
        ),
    }
    write_json(paths.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Paths.output)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(analyze(Paths(output=args.output)), indent=2))


if __name__ == "__main__":
    main()
