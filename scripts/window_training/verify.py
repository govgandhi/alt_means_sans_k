"""Independently audit the saved window/training exploration without changing it.

Reconstruct graph kernels and finite-corpus ordered pair counts, evaluate exact
logistic loss/gradients independently of the runners, and use full score SVDs
for balanced norms. Every saved node vector, pairing and artifact hash is checked.
Run: python3 scripts/window_training/verify.py [--output /experiment/root]
--partial audits completed immutable runs during a sweep. --repeat reruns seed0
in both lanes in a fresh temporary directory and requires byte-identical outputs.
--compare-repeat PATH audits a saved repeat; stdout is always a JSON record.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_exact_center_exposure import (  # noqa: E402
    configuration_graph,
    graph_arrays,
)
from simulate_norm_frequency import sample_walks  # noqa: E402


def read_json(path: Path) -> Any:
    contents = path.read_bytes()
    return json.loads(gzip.decompress(contents) if path.suffix == ".gz" else contents)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def compare(actual: Any, expected: Any, name: str, atol: float = 1e-9) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=atol, err_msg=name)


def verify_hashes(directory: Path, manifest: dict[str, str]) -> None:
    for name, digest in manifest.items():
        require(
            sha(directory / name) == digest, f"Changed artifact: {directory / name}"
        )


def verify_plan(directory: Path) -> dict[str, Any]:
    plan = read_json(directory / "plan.json")
    require(platform.python_version() == plan["python"], "Python runtime mismatch")
    verify_hashes(ROOT, plan["source_sha256"])
    for package, version in plan["packages"].items():
        require(
            importlib.metadata.version(package) == version,
            f"Runtime mismatch: {package}",
        )
    return plan


def objective(
    center: np.ndarray, context: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> dict[str, Any]:
    """Derive the normalized logistic objective from Bernoulli probabilities."""
    scores = center @ context.T
    probability = np.exp(-np.logaddexp(0, -scores))
    mass = positive.sum()
    rows = positive.sum(axis=1)
    require(mass > 0 and np.all(rows > 0), "Every center needs positive exposure")
    terms = positive * np.logaddexp(0, -scores) + negative * np.logaddexp(0, scores)
    derivative = ((positive + negative) * probability - positive) / mass
    gradient = np.concatenate(
        [(derivative @ context).ravel(), (derivative.T @ center).ravel()]
    )
    outside = np.abs(scores) >= 6
    return {
        "scores": scores,
        "expected_loss": float(terms.sum() / mass),
        "node_loss": terms.sum(axis=1) / rows,
        "gradient": gradient,
        "gradient_max": float(np.max(np.abs(gradient))),
        "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(gradient.size)),
        "positive_mass_outside_update_window": float(positive[outside].sum() / mass),
        "negative_mass_outside_update_window": float(
            negative[outside].sum() / negative.sum()
        ),
    }


def full_svd_norm(scores: np.ndarray) -> np.ndarray:
    if not np.any(scores):
        return np.zeros(len(scores))
    left, singular, _ = np.linalg.svd(scores, full_matrices=False)
    return np.sqrt(np.sum(left**2 * singular[None, :], axis=1))


def infimum(positive: np.ndarray, negative: np.ndarray) -> float:
    """Optimal binary entropy, with zero weights contributing exactly zero."""
    total = positive + negative
    value = 0.0
    for weights in (positive, negative):
        chosen = weights > 0
        value -= float(
            np.sum(weights[chosen] * np.log(weights[chosen] / total[chosen]))
        )
    return value / positive.sum()


def curve(degree: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    require(
        values.shape == degree.shape and np.isfinite(values).all(),
        "Invalid node readout",
    )
    require(np.all(degree > 0), "Invalid graph degree")
    log_degree = np.log(degree)
    edges = np.unique(np.quantile(log_degree, np.arange(9) / 8))
    groups = np.searchsorted(edges[1:-1], log_degree, side="left")
    means = np.array([values[groups == label].mean() for label in np.unique(groups)])
    peak = int(np.argmax(means))
    return {
        "means": means.tolist(),
        "node_counts": [int(np.sum(groups == label)) for label in np.unique(groups)],
        "peak_bin": peak + 1,
        "interior_peak": 0 < peak < len(means) - 1,
        "highest_minus_lowest": float(means[-1] - means[0]),
        "highest_minus_penultimate": None
        if len(means) < 2
        else float(means[-1] - means[-2]),
    }


def check_nodes(snapshot: dict[str, Any], degree: np.ndarray) -> None:
    for name in ("raw_norm", "balanced_norm", "node_loss", "score_change_rms_by_node"):
        values = np.asarray(snapshot[name])
        require(values.shape == degree.shape, f"Wrong node count for {name}")
        require(
            np.isfinite(values).all() and np.all(values >= -1e-12), f"Invalid {name}"
        )
    for name in ("raw_norm", "balanced_norm"):
        summary = curve(degree, np.asarray(snapshot[name]))
        require(sum(summary["node_counts"]) == len(degree), "Curve lost nodes")


def verify_factor_snapshot(
    snapshot: dict[str, Any],
    center: np.ndarray,
    context: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    gradient_key: str,
) -> dict[str, Any]:
    require(
        center.shape == context.shape and np.isfinite([center, context]).all(),
        "Invalid factors",
    )
    require(array_sha(center) == snapshot["input_sha256"], "Input factor hash differs")
    require(
        array_sha(context) == snapshot["output_sha256"], "Output factor hash differs"
    )
    calculated = objective(center, context, positive, negative)
    compare(calculated["expected_loss"], snapshot["expected_loss"], "Expected loss")
    compare(calculated["node_loss"], snapshot["node_loss"], "Per-node loss")
    compare(np.linalg.norm(center, axis=1), snapshot["raw_norm"], "Raw norms")
    compare(
        full_svd_norm(calculated["scores"]),
        snapshot["balanced_norm"],
        "Full-SVD balanced norms",
        1e-7,
    )
    compare(
        calculated["gradient_rms"], snapshot[gradient_key], "Idealized factor gradient"
    )
    return calculated


def reconstruct_windows(
    adjacency: np.ndarray, degree: np.ndarray, lengths: list[int]
) -> dict[int, np.ndarray]:
    transition = adjacency / degree[:, None]
    power, total = np.eye(len(degree)), np.zeros_like(adjacency)
    kernels = {}
    for step in range(1, max(lengths) + 1):
        power = power @ transition
        total += power
        if step in lengths:
            kernels[step] = total / step
    noise = degree / degree.sum()
    for conditional in kernels.values():
        compare(conditional.sum(axis=1), np.ones(len(degree)), "Stochastic rows", 1e-13)
        compare(noise @ conditional, noise, "Stationary window distribution", 1e-13)
        require(np.all(conditional >= 0), "Negative context probability")
    return kernels


def verify_free_score(
    info: dict[str, Any], positive: np.ndarray, negative: np.ndarray
) -> None:
    finite = (positive > 0) & (negative > 0)
    for name, mask in (
        ("finite_entries", finite),
        ("minus_infinity_entries", (positive == 0) & (negative > 0)),
        ("plus_infinity_entries", (positive > 0) & (negative == 0)),
        ("unidentified_entries", (positive == 0) & (negative == 0)),
    ):
        require(info[name] == int(mask.sum()), f"Incorrect free-score boundary: {name}")
    compare(
        infimum(positive, negative),
        info["unrestricted_loss_infimum"],
        "Unrestricted infimum",
    )
    require(
        info["finite_full_rank_norms_defined"] == bool(finite.all()),
        "Incorrect finite-target flag",
    )
    if not finite.all():
        require(
            "balanced_norm" not in info,
            "Infinite free scores received finite target norms",
        )
        return
    scores = np.log(positive / negative)
    require(array_sha(scores) == info["score_sha256"], "Free-score hash differs")
    compare(
        full_svd_norm(scores), info["balanced_norm"], "Full-rank free-score norms", 1e-7
    )
    compare(
        np.linalg.svd(scores, compute_uv=False),
        info["singular_values"],
        "Free-score singular values",
    )
    compare(
        info["reconstructed_loss"],
        infimum(positive, negative),
        "Free-score achieved lower bound",
    )
    require(info["gradient_max"] < 1e-10, "Nonzero finite free-score gradient")


def verify_windows(directory: Path, partial: bool = False) -> dict[str, Any]:
    plan = verify_plan(directory)
    config = plan["config"]
    complete = directory / "completion.json"
    if complete.exists():
        verify_hashes(directory, read_json(complete))
    else:
        require(partial, "Expected-window sweep is not complete")
    markers = sorted((directory / "runs").glob("*.sha256"))
    expected_names = {
        f"seed{seed}_L{length}_init{offset}"
        for seed in config["seeds"]
        for length in config["windows"]
        for offset in config["initialization_offsets"]
    }
    require({path.stem for path in markers} <= expected_names, "Unexpected window run")
    if not partial:
        require({path.stem for path in markers} == expected_names, "Missing window run")
    graphs, endpoints = {}, []
    for marker in markers:
        verify_hashes(directory, read_json(marker))
        result = read_json(marker.with_suffix(".json.gz"))
        seed, length = result["seed"], result["window"]
        expected_name = f"seed{seed}_L{length}_init{result['initialization'] - seed}"
        require(marker.stem == expected_name, "Result identity differs from filename")
        if seed not in graphs:
            metadata = read_json(directory / result["graph_file"])
            adjacency, _, degree = graph_arrays(
                configuration_graph(config["nodes"], config["attachment"], seed)
            )
            require(
                array_sha(adjacency) == metadata["graph_sha256"],
                "Window graph hash mismatch",
            )
            compare(degree, metadata["degree"], "Window node degrees", 0)
            noise = degree / degree.sum()
            compare(noise, metadata["noise"], "Fixed stationary noise", 0)
            compare(
                metadata["center_share"],
                np.full(len(degree), 1 / len(degree)),
                "Equal center share",
                0,
            )
            kernels = reconstruct_windows(adjacency, degree, config["windows"])
            for lag, kernel in kernels.items():
                info = metadata["windows"][str(lag)]
                require(
                    array_sha(kernel) == info["conditional_sha256"],
                    "Conditional target changed",
                )
                compare(
                    (kernel > 0).sum(axis=1),
                    info["positive_support"],
                    "Positive support",
                    0,
                )
                kl_terms = np.zeros_like(kernel)
                nonzero = kernel > 0
                ratio = kernel / noise[None, :]
                kl_terms[nonzero] = kernel[nonzero] * np.log(ratio[nonzero])
                compare(kl_terms.sum(axis=1), info["context_kl"], "Context KL")
                negative = np.broadcast_to(
                    config["negatives"] * noise[None, :] / len(degree), kernel.shape
                )
                verify_free_score(info["free_score"], kernel / len(degree), negative)
            graphs[seed] = degree, noise, kernels
        degree, noise, kernels = graphs[seed]
        conditional = kernels[length]
        positive = conditional / len(degree)
        negative = np.broadcast_to(
            config["negatives"] * noise[None, :] / len(degree), positive.shape
        )
        require(
            array_sha(positive) == result["positive_weights_sha256"],
            "Positive objective weights mismatch",
        )
        require(
            array_sha(negative) == result["negative_weights_sha256"],
            "Fixed noise changed across fits",
        )
        require(
            sha(directory / result["graph_file"]) == result["graph_file_sha256"],
            "Window metadata changed",
        )
        require(
            sha(directory / result["factor_file"]) == result["factor_file_sha256"],
            "Window factor file changed",
        )
        snapshots, termination = result["snapshots"], result["termination"]
        steps = [row["iteration"] for row in snapshots]
        require(
            steps == sorted(set(steps)) and steps[0] == 0,
            "Invalid window checkpoint order",
        )
        require(
            steps[-1] == termination["iterations"] <= config["maxiter"],
            "Invalid endpoint iteration",
        )
        expected_steps = [
            value for value in config["checkpoints"] if value <= steps[-1]
        ]
        if steps[-1] not in expected_steps:
            expected_steps.append(steps[-1])
        require(steps == expected_steps, "Missing window checkpoint")
        lower = infimum(positive, negative)
        compare(lower, result["unrestricted_loss_infimum"], "Window lower bound")
        for snapshot in snapshots:
            check_nodes(snapshot, degree)
            compare(
                np.mean(snapshot["node_loss"]),
                snapshot["expected_loss"],
                "Equal-center node loss averaging",
            )
            compare(
                snapshot["expected_loss"] - lower,
                snapshot["excess_over_unrestricted_infimum"],
                "Lower-bound excess",
            )
            require(
                snapshot["expected_loss"] >= lower - 1e-10,
                "Loss below unrestricted infimum",
            )
        require(
            np.all(np.diff([row["expected_loss"] for row in snapshots]) <= 1e-8),
            "L-BFGS accepted losses increased",
        )
        rank = config["dimension"]
        rng = np.random.default_rng(result["initialization"])
        initial_center = rng.uniform(-0.5 / rank, 0.5 / rank, (len(degree), rank))
        initial_context = np.zeros_like(initial_center)
        require(
            array_sha(np.stack([initial_center, initial_context]).ravel())
            == result["initial_factors_sha256"],
            "Initialization mismatch",
        )
        verify_factor_snapshot(
            snapshots[0],
            initial_center,
            initial_context,
            positive,
            negative,
            "gradient_rms",
        )
        factors = np.load(
            io.BytesIO(
                gzip.decompress((directory / result["factor_file"]).read_bytes())
            ),
            allow_pickle=False,
        )
        require(factors.shape == (2, len(degree), rank), "Wrong final factor shape")
        final = verify_factor_snapshot(
            snapshots[-1], factors[0], factors[1], positive, negative, "gradient_rms"
        )
        compare(
            final["gradient_max"],
            snapshots[-1]["gradient_max"],
            "Final gradient maximum",
        )
        endpoints.append(
            {
                "seed": seed,
                "window": length,
                "initialization": result["initialization"],
                "iterations": steps[-1],
                "optimizer_success": termination["success"],
                "gradient_max": final["gradient_max"],
                "expected_loss": final["expected_loss"],
                "raw_curve": curve(degree, np.asarray(snapshots[-1]["raw_norm"])),
                "balanced_curve": curve(
                    degree, np.asarray(snapshots[-1]["balanced_norm"])
                ),
            }
        )
    return {
        "completed_runs": len(markers),
        "expected_runs": len(expected_names),
        "endpoints": endpoints,
    }


def corpus_digest(walks: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for walk in walks:
        digest.update(np.asarray([len(walk), *walk], dtype="<i8").tobytes())
    return digest.hexdigest()


def unfiltered_pairs(walks: list[list[int]], nodes: int) -> np.ndarray:
    """Directly enumerate adjacent ordered pairs; no retention evaluator reused."""
    counts = np.zeros((nodes, nodes))
    for walk in walks:
        for first, second in zip(walk, walk[1:]):
            counts[first, second] += 1
            counts[second, first] += 1
    return counts


def discrete_proposal(cumulative: np.ndarray, bits: int = 32) -> np.ndarray:
    """Count bit integers in each modulo-domain/bisect-left interval."""
    domain = int(cumulative[-1])
    quotient, remainder = divmod(2**bits, domain)
    result, lower = [], 0
    for boundary in cumulative:
        upper = min(int(boundary), domain - 1)
        size = max(upper - lower + 1, 0)
        extra = max(min(upper, remainder - 1) - lower + 1, 0)
        result.append((quotient * size + extra) / 2**bits)
        lower = int(boundary) + 1
    return np.asarray(result)


def verify_schedule(directory: Path, partial: bool = False) -> dict[str, Any]:
    from check_control_gensim import blas_check

    plan = verify_plan(directory)
    require(blas_check() == plan["blas_check"], "Compiled Gensim runtime mismatch")
    complete = directory / "completion.json"
    if complete.exists():
        verify_hashes(directory, read_json(complete))
    else:
        require(partial, "Schedule sweep is not complete")
    reference = ROOT / plan["config"]["reference"]
    require(
        sha(reference / "completion.json") == plan["reference_completion_sha256"],
        "PR36 reference changed",
    )
    verify_hashes(reference, read_json(reference / "completion.json"))
    prior_plan = read_json(reference / "plan.json")
    baseline_dir = ROOT / prior_plan["config"]["reference"]
    verify_hashes(baseline_dir, read_json(baseline_dir / "completion.json"))
    baseline_checks = read_json(baseline_dir / "checks.json")
    training = read_json(baseline_dir / "plan.json")["config"]["training"]
    markers = sorted(directory.glob("seed*/completion.json"))
    require(
        {path.parent.name for path in markers} <= {f"seed{seed}" for seed in range(5)},
        "Unexpected schedule seed",
    )
    if not partial:
        require(len(markers) == 5, "Missing schedule seed")
    endpoints = []
    for marker in markers:
        seed_dir = marker.parent
        verify_hashes(seed_dir, read_json(marker))
        metadata = read_json(seed_dir / "metadata.json")
        seed = metadata["seed"]
        old = next(
            row
            for row in baseline_checks
            if row["seed"] == seed and row["threshold"] == 0
        )
        adjacency, neighbors, degree = graph_arrays(
            configuration_graph(training["nodes"], training["attachment"], seed)
        )
        walks, counts = sample_walks(
            neighbors,
            degree / degree.sum(),
            training["nodes"] * training["walks_per_node"],
            training["walk_length"],
            np.random.default_rng(seed + 10_000),
        )
        for name, digest in (
            ("graph_sha256", array_sha(adjacency)),
            ("walks_sha256", corpus_digest(walks)),
            ("counts_sha256", array_sha(counts)),
        ):
            require(
                metadata[name] == old[name] == digest,
                f"Schedule corpus mismatch: {name}",
            )
        compare(metadata["degree"], degree, "Schedule node degrees", 0)
        compare(metadata["tokens"], counts, "Schedule token counts", 0)
        vocabulary = np.asarray(old["initial"]["vocabulary"])
        cumulative = np.rint(
            np.cumsum(counts[vocabulary]) / counts.sum() * (2**31 - 1)
        ).astype(np.uint32)
        require(
            array_sha(cumulative) == metadata["fork_state"]["negative_cdf_sha256"],
            "Incorrect raw-token negative CDF",
        )
        proposal = np.asarray(metadata["noise_probability"])
        expected_proposal = np.empty_like(proposal)
        expected_proposal[vocabulary] = discrete_proposal(cumulative)
        compare(proposal, expected_proposal, "Exact discrete raw-token proposal", 1e-15)
        compare(proposal.sum(), 1, "Noise total", 1e-14)
        require(np.all(proposal > 0), "Missing schedule noise support")
        positive = unfiltered_pairs(walks, len(degree))
        negative = (
            training["negatives"]
            * proposal[None, :]
            * (positive.sum(axis=1)[:, None] - positive)
        )
        require(
            array_sha(positive) == metadata["positive_weights_sha256"],
            "Finite-corpus positive weights differ",
        )
        require(
            array_sha(negative) == metadata["negative_weights_sha256"],
            "Collision-skipping negative weights differ",
        )
        saved = read_json(
            reference / "runs" / f"seed{seed}_t0_baseline_then_floor.json"
        )
        require(
            sha(reference / "runs" / f"seed{seed}_t0_baseline_then_floor.json")
            == metadata["reference_run_sha256"],
            "PR36 source run changed",
        )
        fork_hashes, endings = [], []
        for branch, rate in (("floor", 0.0001), ("reheat", 0.0025)):
            result = read_json(seed_dir / f"{branch}.json")
            require(
                result["epoch_counts"] == saved["epoch_counts"],
                "Token accounting differs from PR36",
            )
            require(result["learning_rate"] == rate, "Wrong continuation rate")
            require(
                [row["epoch"] for row in result["snapshots"]] == [5, 6, 7, 10, 15, 20],
                "Invalid schedule checkpoints",
            )
            previous, fork_scores = None, None
            for snapshot in result["snapshots"]:
                check_nodes(snapshot, degree)
                require(
                    sha(seed_dir / snapshot["factor_file"])
                    == snapshot["factor_file_sha256"],
                    "Schedule factor archive changed",
                )
                with np.load(
                    seed_dir / snapshot["factor_file"], allow_pickle=False
                ) as data:
                    center, context = data["center"], data["context"]
                require(
                    center.shape == (len(degree), training["dimension"]),
                    "Wrong schedule factor shape",
                )
                calculated = verify_factor_snapshot(
                    snapshot,
                    center,
                    context,
                    positive,
                    negative,
                    "idealized_gradient_rms",
                )
                scores = calculated["scores"]
                if fork_scores is None:
                    fork_scores = scores.copy()
                expected_change = (
                    np.zeros(len(degree))
                    if previous is None
                    else np.sqrt(np.mean((scores - previous) ** 2, axis=1))
                )
                compare(
                    expected_change,
                    snapshot["score_change_rms_by_node"],
                    "Checkpoint score movement",
                )
                compare(
                    np.sqrt(np.mean((scores - fork_scores) ** 2, axis=1)),
                    snapshot["score_change_from_fork_rms_by_node"],
                    "Fork score movement",
                )
                for key in (
                    "positive_mass_outside_update_window",
                    "negative_mass_outside_update_window",
                ):
                    compare(calculated[key], snapshot[key], key)
                tokens = int(counts.sum()) * (snapshot["epoch"] - 5)
                require(
                    snapshot["post_fork_effective_tokens"]
                    == snapshot["post_fork_raw_tokens"]
                    == tokens,
                    "Checkpoint token accounting mismatch",
                )
                compare(
                    snapshot["post_fork_alpha_times_effective_tokens"],
                    rate * tokens,
                    "Learning-rate exposure",
                )
                reference_snapshot = next(
                    (
                        row
                        for row in saved["snapshots"]
                        if row["epoch"] == snapshot["epoch"]
                    ),
                    None,
                )
                if snapshot["epoch"] == 5 or (branch == "floor" and reference_snapshot):
                    for key in ("input_sha256", "output_sha256"):
                        require(
                            snapshot[key] == reference_snapshot[key],
                            f"PR36 checkpoint bridge failed: {key}",
                        )
                        if snapshot["epoch"] == 5:
                            require(
                                snapshot[key] == old[key],
                                "Five-epoch baseline bridge failed",
                            )
                previous = scores
            first, final = result["snapshots"][0], result["snapshots"][-1]
            require(
                result["effective_tokens"]
                == result["raw_tokens"]
                == int(counts.sum()) * 15,
                "Continuation total-token mismatch",
            )
            for key in ("rng_array_sha256", "rng_metadata"):
                require(
                    first[key] == metadata["fork_state"][key],
                    "Fork RNG differs from saved model state",
                )
                require(
                    final[key] == result["final_state"][key],
                    "Final RNG differs from checkpoint",
                )
            for snapshot, state in (
                (first, metadata["fork_state"]),
                (final, result["final_state"]),
            ):
                with np.load(
                    seed_dir / snapshot["factor_file"], allow_pickle=False
                ) as factors:
                    require(
                        array_sha(factors["center"][vocabulary].astype(np.float32))
                        == state["initial_input_sha256"],
                        "Native input state differs from factors",
                    )
                    require(
                        array_sha(factors["context"][vocabulary].astype(np.float32))
                        == state["initial_output_sha256"],
                        "Native output state differs from factors",
                    )
            fork_hashes.append(
                [
                    first[key]
                    for key in (
                        "input_sha256",
                        "output_sha256",
                        "rng_array_sha256",
                        "rng_metadata",
                    )
                ]
            )
            endings.append(result["final_state"])
            for key in (
                "vocabulary",
                "slot_counts_sha256",
                "negative_cdf_sha256",
                "sample_int_sha256",
                "lockf_sha256",
            ):
                require(
                    result["final_state"][key] == metadata["fork_state"][key],
                    f"Training changed sampling state: {key}",
                )
            endpoints.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "expected_loss": final["expected_loss"],
                    "raw_curve": curve(degree, np.asarray(final["raw_norm"])),
                    "balanced_curve": curve(degree, np.asarray(final["balanced_norm"])),
                }
            )
        require(
            fork_hashes[0] == fork_hashes[1],
            "Branches did not start from the same factors and RNG",
        )
        for key in ("rng_array_sha256", "rng_metadata", "epoch_counts"):
            require(
                endings[0][key] == endings[1][key],
                f"Final random stream/accounting mismatch: {key}",
            )
    return {
        "completed_seeds": len(markers),
        "expected_seeds": 5,
        "endpoints": endpoints,
    }


def validate_shape(record: dict[str, Any], values: np.ndarray) -> None:
    peak = int(np.argmax(values))
    require(record["peak_bin_zero_based"] == peak, "Reported peak bin mismatch")
    require(
        record["interior_peak"] == (0 < peak < len(values) - 1),
        "Reported interior peak mismatch",
    )
    compare(
        record["high_minus_low"], values[-1] - values[0], "Reported fixed endpoint gap"
    )
    compare(
        record["high_minus_penultimate"],
        values[-1] - values[-2],
        "Reported upper-bin gap",
    )


def validate_report_curve(
    record: dict[str, Any], degrees: list[np.ndarray], values: list[np.ndarray]
) -> None:
    boundaries = np.unique(
        np.percentile(np.log(np.concatenate(degrees)), np.arange(9) * 12.5)
    )
    labels = [
        np.digitize(np.log(degree), boundaries[1:-1], right=True) for degree in degrees
    ]
    occupied = sorted(set(np.concatenate(labels)))
    means = np.array(
        [
            [np.mean(response[group == label]) for label in occupied]
            for response, group in zip(values, labels, strict=True)
        ]
    )
    require(means.shape == (5, len(record["bins"])), "Incorrect report graph/bin count")
    require(
        [row["seed"] for row in record["per_graph"]] == list(range(5)),
        "Report omitted a graph",
    )
    for seed, row in enumerate(record["per_graph"]):
        compare(row["values"], means[seed], "Reported individual-graph means")
        validate_shape(row, means[seed])
    for index, (label, row) in enumerate(zip(occupied, record["bins"], strict=True)):
        counts = [int(np.count_nonzero(group == label)) for group in labels]
        require(row["counts_by_seed"] == counts, "Reported raw node counts differ")
        compare(row["values_by_seed"], means[:, index], "Reported graph values")
        compare(row["mean"], np.mean(means[:, index]), "Equal graph-weighted mean")
        compare(
            row["standard_error"],
            np.std(means[:, index], ddof=1) / np.sqrt(5),
            "Graph-level standard error",
        )
        selected = np.concatenate(
            [
                degree[group == label]
                for degree, group in zip(degrees, labels, strict=True)
            ]
        )
        compare(
            row["degree"],
            np.exp(np.mean(np.log(selected))),
            "Reported degree-bin coordinate",
        )
    validate_shape(record["mean_curve"], means.mean(axis=0))


def verify_report(output: Path) -> dict[str, int]:
    directory = output / "report"
    manifest = read_json(directory / "manifest.json")
    verify_hashes(directory, manifest["artifact_sha256"])
    require(
        manifest["script_sha256"] == sha(Path(__file__).with_name("report.py")),
        "Report source changed",
    )
    for lane, digest in manifest["input_completion_sha256"].items():
        require(
            digest == sha(output / lane / "completion.json"), "Report input changed"
        )
    reference = ROOT / read_json(output / "schedule/plan.json")["config"]["reference"]
    require(
        manifest["reference_completion_sha256"] == sha(reference / "completion.json"),
        "Archived report reference changed",
    )
    report = read_json(directory / "summary.json")
    graphs = [
        read_json(output / "windows" / f"graphs/seed{seed}.json") for seed in range(5)
    ]
    degrees = [np.asarray(graph["degree"]) for graph in graphs]
    runs = [
        read_json(path) for path in sorted((output / "windows/runs").glob("*.json.gz"))
    ]
    schedule = {
        (seed, branch): read_json(output / "schedule" / f"seed{seed}/{branch}.json")
        for seed in range(5)
        for branch in ("floor", "reheat")
    }
    require(
        len(report["schedule"]) == 60
        and len(report["constant_from_start_reference"]) == 4,
        "Incomplete schedule report",
    )
    common_steps = set.intersection(
        *({row["iteration"] for row in run["snapshots"]} for run in runs)
    )
    require(
        len(report["windows"]) == 4 * 4 * (len(common_steps) + 1),
        "Incomplete window report",
    )
    for record in report["schedule"]:
        values = [
            np.asarray(
                next(
                    row
                    for row in schedule[seed, record["branch"]]["snapshots"]
                    if row["epoch"] == record["epoch"]
                )[record["field"]]
            )
            for seed in range(5)
        ]
        validate_report_curve(record, degrees, values)
    for record in report["constant_from_start_reference"]:
        values = [
            np.asarray(
                next(
                    row
                    for row in read_json(
                        reference / f"runs/seed{seed}_t0_constant.json"
                    )["snapshots"]
                    if row["epoch"] == 20
                )[record["field"]]
            )
            for seed in range(5)
        ]
        validate_report_curve(record, degrees, values)
    for record in report["windows"]:
        values = []
        for seed in range(5):
            selected = [
                run
                for run in runs
                if run["seed"] == seed and run["window"] == record["window"]
            ]
            require(len(selected) == 2, "Initialization pair incomplete")
            starts = []
            for run in selected:
                snapshot = (
                    run["snapshots"][-1]
                    if record["step"] == "final"
                    else next(
                        row
                        for row in run["snapshots"]
                        if row["iteration"] == record["step"]
                    )
                )
                starts.append(np.asarray(snapshot[record["field"]]))
            values.append((starts[0] + starts[1]) / 2)
        validate_report_curve(record, degrees, values)
    for record in report["free_score"]:
        values = [
            np.asarray(
                graph["windows"][str(record["window"])]["free_score"]["balanced_norm"]
            )
            for graph in graphs
        ]
        validate_report_curve(record, degrees, values)
    require(
        len(report["initialization_comparison"]) == 40,
        "Missing initialization comparison",
    )
    for record in report["initialization_comparison"]:
        pair = sorted(
            [
                run
                for run in runs
                if run["seed"] == record["seed"] and run["window"] == record["window"]
            ],
            key=lambda run: run["initialization"],
        )
        require(
            record["initializations"] == [run["initialization"] for run in pair],
            "Initialization labels changed",
        )
        values = [np.asarray(run["snapshots"][-1][record["field"]]) for run in pair]
        compare(
            record["node_rms_difference"],
            np.linalg.norm(values[0] - values[1]) / np.sqrt(len(values[0])),
            "Between-start norm difference",
        )
        compare(
            record["losses"],
            [run["snapshots"][-1]["expected_loss"] for run in pair],
            "Between-start endpoint losses",
        )
    require(len(report["schedule_progress"]) == 60, "Missing schedule progress row")
    for record in report["schedule_progress"]:
        snapshot = next(
            row
            for row in schedule[record["seed"], record["branch"]]["snapshots"]
            if row["epoch"] == record["epoch"]
        )
        for key, value in record.items():
            if key not in {"seed", "branch"}:
                require(
                    value == snapshot[key], f"Reported schedule progress changed: {key}"
                )
    require(
        len(report["window_progress"]) == sum(len(run["snapshots"]) for run in runs),
        "Missing window progress row",
    )
    for record in report["window_progress"]:
        run = next(
            run
            for run in runs
            if all(
                run[key] == record[key] for key in ("seed", "window", "initialization")
            )
        )
        snapshot = next(
            row for row in run["snapshots"] if row["iteration"] == record["iteration"]
        )
        require(
            record["termination"] == run["termination"],
            "Reported termination status changed",
        )
        require(
            record["final"] == (snapshot is run["snapshots"][-1]),
            "Wrong final-point label",
        )
        for key, value in record.items():
            if key not in {"seed", "window", "initialization", "termination", "final"}:
                require(
                    value == snapshot[key], f"Reported window progress changed: {key}"
                )
    return {
        "verified_curves": sum(
            len(report[key])
            for key in (
                "schedule",
                "constant_from_start_reference",
                "windows",
                "free_score",
            )
        ),
        "verified_initialization_comparisons": 40,
    }


def verify(output: Path, partial: bool = False) -> dict[str, Any]:
    result = {
        "schedule": verify_schedule(output / "schedule", partial),
        "windows": verify_windows(output / "windows", partial),
    }
    if (output / "report/manifest.json").exists():
        result["report"] = verify_report(output)
    result["provenance"] = {
        "verification_source_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (Path(__file__), ROOT / "tests/test_window_verification.py")
        },
        "input_manifest_sha256": {
            name: sha(output / name)
            for name in (
                "schedule/plan.json",
                "schedule/completion.json",
                "windows/plan.json",
                "windows/completion.json",
                "report/manifest.json",
            )
            if (output / name).exists()
        },
    }
    return result


def compare_repeat(output: Path, destination: Path) -> dict[str, Any]:
    """Validate an existing fresh seed-zero repeat without rerunning training."""
    names = [
        path.relative_to(output)
        for path in sorted((output / "schedule" / "seed0").rglob("*"))
        if path.is_file()
    ]
    for folder in ("graphs", "runs", "factors"):
        names.extend(
            path.relative_to(output)
            for path in sorted((output / "windows" / folder).glob("seed0*"))
        )
    require(len(names) == 42, "Expected all42 seed-zero scientific artifacts")
    hashes = {}
    for name in names:
        original_hash = sha(output / name)
        require(
            original_hash == sha(destination / name), f"Fresh repeat differs: {name}"
        )
        hashes[str(name)] = original_hash
    return {
        "mode": "compare_existing_seed_zero_repeat",
        "byte_identical_scientific_artifacts": len(names),
        "artifact_sha256": hashes,
    }


def repeat(output: Path, destination: Path | None = None) -> dict[str, Any]:
    """Recreate all seed-zero arms without reusing any trained state or outputs."""
    from window_training import expected_window, schedule_fork

    destination = (
        destination or Path(tempfile.mkdtemp(prefix="window-training-repeat-"))
    ).resolve()
    require(
        not destination.exists() or not any(destination.iterdir()),
        "Repeat destination must be empty",
    )
    destination.mkdir(parents=True, exist_ok=True)
    reference = read_json(output / "schedule" / "plan.json")["config"]["reference"]
    with redirect_stdout(sys.stderr):
        schedule_fork.run(
            schedule_fork.Config(
                output=destination / "schedule", reference=Path(reference), seeds=(0,)
            )
        )
        expected_window.run(
            expected_window.Config(output=destination / "windows"), selected_seeds=(0,)
        )
    result = compare_repeat(output, destination)
    result["mode"] = "fresh_seed_zero_repeat"
    print(f"Fresh repeat artifacts retained at {destination}", file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/window_training_exploration"
    )
    parser.add_argument("--partial", action="store_true")
    repeats = parser.add_mutually_exclusive_group()
    repeats.add_argument("--repeat", action="store_true")
    repeats.add_argument("--compare-repeat", type=Path)
    parser.add_argument("--repeat-output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.output, args.partial)
        if args.repeat:
            result["repeat"] = repeat(args.output, args.repeat_output)
        elif args.compare_repeat:
            result["repeat"] = compare_repeat(args.output, args.compare_repeat)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
