"""Fit fixed-noise expected SGNS across exact context windows, without pair rounding.

Inputs: deterministic graph seeds and the frozen exploration protocol.
Outputs: node-level optimization checkpoints, final factors, finite free-score
benchmarks where defined, and source/runtime/input/artifact hashes.
Run: python3 scripts/window_training/expected_window.py [--output /fresh/path]
Optional --seeds selects resumable graph subsets; all five are required to finish.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from simulate_exact_center_exposure import (
    balanced_left_norm,
    configuration_graph,
    graph_arrays,
)
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    output: Path = Path("results/window_training_exploration/windows")
    protocol: Path = Path("analysis/window_training_exploration_protocol.md")
    requirements: Path = Path("analysis/window_training_exploration_requirements.txt")
    nodes: int = 600
    attachment: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    windows: tuple[int, ...] = (1, 2, 5, 10)
    dimension: int = 24
    negatives: int = 5
    initialization_offsets: tuple[int, ...] = (100_000, 200_000)
    checkpoints: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
    maxiter: int = 500
    maxfun: int = 5000
    maxls: int = 40
    maxcor: int = 10
    gtol: float = 1e-8
    ftol: float = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    data = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def read_json(path: Path) -> Any:
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write_factors(path: Path, center: np.ndarray, context: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, np.stack([center, context]), allow_pickle=False)
    path.write_bytes(gzip.compress(stream.getvalue(), mtime=0))


def uniform_windows(
    transition: np.ndarray, windows: tuple[int, ...]
) -> dict[int, np.ndarray]:
    if not windows or min(windows) < 1 or np.any(transition < 0):
        raise ValueError("Nonnegative transition matrix and positive windows required.")
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-14)
    power = np.eye(len(transition))
    total = np.zeros_like(transition)
    result = {}
    for lag in range(1, max(windows) + 1):
        power = power @ transition
        total += power
        if lag in windows:
            result[lag] = total / lag
    return result


def loss_gradient(
    packed: np.ndarray, positive: np.ndarray, negative: np.ndarray, rank: int
) -> tuple[float, np.ndarray]:
    n = len(positive)
    center, context = packed.reshape(2, n, rank)
    scores = center @ context.T
    loss = np.sum(
        positive * np.logaddexp(0, -scores) + negative * np.logaddexp(0, scores)
    )
    derivative = -positive * expit(-scores) + negative * expit(scores)
    gradient = np.stack([derivative @ context, derivative.T @ center]).ravel()
    return float(loss), gradient


def unrestricted_infimum(positive: np.ndarray, negative: np.ndarray) -> float:
    total = positive + negative
    return float(
        np.sum(
            xlogy(total, total) - xlogy(positive, positive) - xlogy(negative, negative)
        )
    )


def free_score_benchmark(positive: np.ndarray, negative: np.ndarray) -> dict[str, Any]:
    finite = (positive > 0) & (negative > 0)
    result = {
        "finite_entries": int(finite.sum()),
        "minus_infinity_entries": int(np.sum((positive == 0) & (negative > 0))),
        "plus_infinity_entries": int(np.sum((positive > 0) & (negative == 0))),
        "unidentified_entries": int(np.sum((positive == 0) & (negative == 0))),
        "unrestricted_loss_infimum": unrestricted_infimum(positive, negative),
        "finite_full_rank_norms_defined": bool(finite.all()),
    }
    if not finite.all():
        return result
    scores = np.log(positive / negative)
    left, singular, right = np.linalg.svd(scores, full_matrices=False)
    center, context = left * np.sqrt(singular), right.T * np.sqrt(singular)
    loss, gradient = loss_gradient(
        np.stack([center, context]).ravel(), positive, negative, len(scores)
    )
    np.testing.assert_allclose(center @ context.T, scores, atol=1e-11)
    np.testing.assert_allclose(loss, result["unrestricted_loss_infimum"], atol=1e-12)
    result.update(
        balanced_norm=np.linalg.norm(center, axis=1).tolist(),
        score_sha256=array_sha(scores),
        reconstructed_loss=loss,
        gradient_max=float(np.max(abs(gradient))),
        singular_values=singular.tolist(),
    )
    return result


def fit(
    conditional: np.ndarray,
    noise: np.ndarray,
    degree: np.ndarray,
    config: Config,
    initialization: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n, rank = len(degree), config.dimension
    positive = conditional / n
    negative = np.broadcast_to(config.negatives * noise[None, :] / n, positive.shape)
    lower = unrestricted_infimum(positive, negative)
    rng = np.random.default_rng(initialization)
    center = rng.uniform(-0.5 / rank, 0.5 / rank, (n, rank))
    context = np.zeros_like(center)
    packed = np.stack([center, context]).ravel()
    initial_hash = array_sha(packed)
    snapshots: list[dict[str, Any]] = []
    previous_scores = None
    iteration = 0

    def capture(values: np.ndarray, step: int) -> None:
        nonlocal previous_scores
        center, context = values.reshape(2, n, rank)
        scores = center @ context.T
        loss, gradient = loss_gradient(values, positive, negative, rank)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("Nonfinite expected-objective diagnostic.")
        if loss < lower - 1e-11:
            raise AssertionError("A factorized loss fell below its analytic infimum.")
        snapshots.append(
            {
                "iteration": step,
                "expected_loss": loss,
                "excess_over_unrestricted_infimum": loss - lower,
                "gradient_max": float(np.max(abs(gradient))),
                "gradient_rms": float(np.sqrt(np.mean(gradient**2))),
                "raw_norm": np.linalg.norm(center, axis=1).tolist(),
                "balanced_norm": balanced_left_norm(center, context).tolist(),
                "node_loss": np.sum(
                    conditional * np.logaddexp(0, -scores)
                    + config.negatives * noise[None, :] * np.logaddexp(0, scores),
                    axis=1,
                ).tolist(),
                "score_change_rms_by_node": (
                    np.zeros(n)
                    if previous_scores is None
                    else np.sqrt(np.mean((scores - previous_scores) ** 2, axis=1))
                ).tolist(),
                "input_sha256": array_sha(center),
                "output_sha256": array_sha(context),
            }
        )
        previous_scores = scores

    def callback(values: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if iteration in config.checkpoints:
            capture(values, iteration)

    capture(packed, 0)
    result = minimize(
        loss_gradient,
        packed,
        args=(positive, negative, rank),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            name: getattr(config, name)
            for name in ("maxiter", "maxfun", "maxls", "maxcor", "gtol", "ftol")
        },
    )
    if result.nit != iteration:
        raise AssertionError("Optimizer iteration accounting differs from callbacks.")
    if snapshots[-1]["iteration"] != iteration:
        capture(result.x, iteration)
    final_center, final_context = result.x.reshape(2, n, rank)
    return (
        {
            "initialization": initialization,
            "initial_factors_sha256": initial_hash,
            "positive_weights_sha256": array_sha(positive),
            "negative_weights_sha256": array_sha(negative),
            "unrestricted_loss_infimum": lower,
            "termination": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "gradient_evaluations": int(result.njev),
            },
            "snapshots": snapshots,
        },
        final_center,
        final_context,
    )


def plan_run(config: Config) -> tuple[Path, dict[str, Any]]:
    output = config.root / config.output
    source_paths = [
        Path(__file__).relative_to(config.root),
        Path("tests/test_expected_window.py"),
        Path("scripts/simulate_exact_center_exposure.py"),
        config.protocol,
        config.requirements,
    ]
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
        if key not in {"root", "output"}
    }
    plan = {
        "config": parameters,
        "source_sha256": {str(p): sha(config.root / p) for p in source_paths},
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib", "networkx", "threadpoolctl")
        },
        "scope": "Bounded exact expected-objective fits; no global optimum claim.",
    }
    plan = json.loads(json.dumps(plan))
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Nonempty output has no matching experiment plan.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        if read_json(output / "plan.json") != plan:
            raise ValueError("Plan changed; use a fresh output directory.")
    else:
        write_json(output / "plan.json", plan)
    return output, plan


def run(config: Config, selected_seeds: tuple[int, ...] | None = None) -> None:
    selected_seeds = config.seeds if selected_seeds is None else selected_seeds
    if not set(selected_seeds) <= set(config.seeds):
        raise ValueError("Unknown graph seed.")
    output, plan = plan_run(config)
    for name in ("graphs", "runs", "factors"):
        (output / name).mkdir(exist_ok=True)
    for seed in selected_seeds:
        adjacency, _, degree = graph_arrays(
            configuration_graph(config.nodes, config.attachment, seed)
        )
        noise = degree / degree.sum()
        transition = adjacency / degree[:, None]
        windows = uniform_windows(transition, config.windows)
        metadata = {
            "seed": seed,
            "degree": degree.tolist(),
            "graph_sha256": array_sha(adjacency),
            "noise": noise.tolist(),
            "center_share": [1 / len(degree)] * len(degree),
            "windows": {},
        }
        for length, conditional in windows.items():
            positive = conditional / len(degree)
            negative = np.broadcast_to(
                config.negatives * noise[None, :] / len(degree), positive.shape
            )
            metadata["windows"][str(length)] = {
                "conditional_sha256": array_sha(conditional),
                "context_kl": np.sum(
                    xlogy(conditional, conditional / noise[None, :]), axis=1
                ).tolist(),
                "positive_support": (conditional > 0).sum(axis=1).tolist(),
                "free_score": free_score_benchmark(positive, negative),
            }
        graph_path = output / "graphs" / f"seed{seed}.json"
        if graph_path.exists() and read_json(graph_path) != metadata:
            raise ValueError("Graph or expected target changed across resumes.")
        write_json(graph_path, metadata)
        for length, conditional in windows.items():
            for offset in config.initialization_offsets:
                name = f"seed{seed}_L{length}_init{offset}"
                result_path = output / "runs" / f"{name}.json.gz"
                factor_path = output / "factors" / f"{name}.npy.gz"
                marker = output / "runs" / f"{name}.sha256"
                if marker.exists():
                    saved = read_json(marker)
                    for key, expected in saved.items():
                        if sha(output / key) != expected:
                            raise ValueError(f"Modified completed artifact: {key}")
                    print(f"Verified {name}", flush=True)
                    continue
                print(f"Fitting {name}", flush=True)
                result, center, context = fit(
                    conditional, noise, degree, config, seed + offset
                )
                write_factors(factor_path, center, context)
                result.update(
                    seed=seed,
                    window=length,
                    graph_file=str(graph_path.relative_to(output)),
                    graph_file_sha256=sha(graph_path),
                    factor_file=str(factor_path.relative_to(output)),
                    factor_file_sha256=sha(factor_path),
                )
                write_json(result_path, result)
                write_json(
                    marker,
                    {
                        str(path.relative_to(output)): sha(path)
                        for path in (result_path, factor_path, graph_path)
                    },
                )
                last = result["snapshots"][-1]
                print(
                    f"  iterations={result['termination']['iterations']}, "
                    f"loss={last['expected_loss']:.6f}, "
                    f"gradient max={last['gradient_max']:.3g}",
                    flush=True,
                )
    for name, expected in plan["source_sha256"].items():
        if sha(config.root / name) != expected:
            raise ValueError(f"Source changed during experiment: {name}")
    expected_runs = (
        len(config.seeds) * len(config.windows) * len(config.initialization_offsets)
    )
    if len(list((output / "runs").glob("*.sha256"))) == expected_runs:
        write_json(
            output / "completion.json",
            {
                str(path.relative_to(output)): sha(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "completion.json"
            },
        )
        print(f"Completed all {expected_runs} expected-objective fits.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(
            Config(output=args.output),
            None if args.seeds is None else tuple(args.seeds),
        )


if __name__ == "__main__":
    main()
