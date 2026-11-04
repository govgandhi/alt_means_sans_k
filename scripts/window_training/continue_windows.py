"""Continue fixed-noise window fits with exact score-preserving balancing.

Inputs: all forty immutable previous endpoint factors and amendment protocol.
Outputs: every continuation block, ten matched raw-coordinate controls, final
factors and deterministic provenance under results/window_training_continuation.
Run: python3 scripts/window_training/continue_windows.py [--output /fresh/path]
Use --seeds for resumable subsets; --workers sets independent graph processes.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import io
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_training import expected_window as original
from window_training.report import bins_for, node_curve


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    reference: Path = Path("results/window_training_exploration/windows")
    output: Path = Path("results/window_training_continuation")
    protocol: Path = Path("analysis/window_optimization_amendment.md")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    windows: tuple[int, ...] = (1, 2, 5, 10)
    offsets: tuple[int, ...] = (100000, 200000)
    rank: int = 24
    nodes: int = 600
    attachment: int = 3
    negatives: int = 5
    block_iterations: int = 250
    max_blocks: int = 12
    control_blocks: int = 2
    gradient_tolerance: float = 1e-7
    loss_tolerance: float = 1e-8
    stability_tolerance: float = 1e-3
    maxfun: int = 5000
    maxls: int = 40
    maxcor: int = 10
    gtol: float = 1e-8
    ftol: float = 1e-12


def balanced_factors(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, ru = np.linalg.qr(u, mode="reduced")
    right, rv = np.linalg.qr(v, mode="reduced")
    a, singular, bt = np.linalg.svd(ru @ rv.T, full_matrices=False)
    return (left @ a) * np.sqrt(singular), (right @ bt.T) * np.sqrt(singular), singular


def stable_gates(current: dict[str, Any], config: Config) -> dict[str, bool]:
    return {
        "gradient": current["balanced_gradient_max"] <= config.gradient_tolerance,
        "loss": current["relative_loss_change"] <= config.loss_tolerance,
        "scores": current["relative_score_rms_change"] <= config.stability_tolerance,
        "node_norms": current["relative_norm_rms_change"] <= config.stability_tolerance,
        "every_bin": current["max_relative_bin_change"] <= config.stability_tolerance,
        "upper_gap": current["absolute_upper_gap_change"] <= config.stability_tolerance,
    }


def projected_gradient(
    g: np.ndarray, ub: np.ndarray, vb: np.ndarray, singular: np.ndarray
) -> tuple[float, int]:
    active = singular > np.finfo(float).eps * len(ub) * singular[0]
    left, right = (
        ub[:, active] / np.sqrt(singular[active]),
        vb[:, active] / np.sqrt(singular[active]),
    )
    a, b = left @ (left.T @ g), (g @ right) @ right.T
    return float(np.linalg.norm(a + b - (a @ right) @ right.T)), int(active.sum())


def snapshot(
    u: np.ndarray,
    v: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    groups: np.ndarray,
    previous: dict[str, Any] | None,
    previous_scores: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    ub, vb, singular = balanced_factors(u, v)
    scores = u @ v.T
    np.testing.assert_allclose(ub @ vb.T, scores, rtol=1e-9, atol=1e-8)
    loss, gradient = original.loss_gradient(
        np.stack([ub, vb]).ravel(), positive, negative, u.shape[1]
    )
    norm = np.linalg.norm(ub, axis=1)
    binned = node_curve(groups, norm)
    derivative = -positive * expit(-scores) + negative * expit(scores)
    tangent, rank = projected_gradient(derivative, ub, vb, singular)
    tails = {}
    for name, mask in (("positive", positive > 0), ("nonedge", positive == 0)):
        tails[name] = (
            np.quantile(scores[mask], [0, 0.01, 0.5, 0.99, 1]).tolist()
            if mask.any()
            else None
        )
    result = {
        "expected_loss": loss,
        "balanced_gradient_max": float(np.max(abs(gradient))),
        "balanced_gradient_rms": float(np.sqrt(np.mean(gradient**2))),
        "projected_score_gradient_frobenius": tangent,
        "numerical_rank": rank,
        "raw_norm": np.linalg.norm(u, axis=1).tolist(),
        "balanced_norm": norm.tolist(),
        "node_loss": (
            np.sum(
                positive * np.logaddexp(0, -scores)
                + negative * np.logaddexp(0, scores),
                axis=1,
            )
            * len(u)
        ).tolist(),
        "curve": binned,
        "score_singular_values": singular.tolist(),
        "score_quantiles": tails,
        "score_rms": float(np.sqrt(np.mean(scores**2))),
        "center_frobenius": float(np.linalg.norm(u)),
        "context_frobenius": float(np.linalg.norm(v)),
    }
    if previous is not None:
        delta = norm - previous["balanced_norm"]
        old_bins = np.array(previous["curve"]["values"])
        result.update(
            absolute_norm_rms_change=float(np.sqrt(np.mean(delta**2))),
            relative_norm_rms_change=float(
                np.linalg.norm(delta)
                / max(1.0, np.linalg.norm(previous["balanced_norm"]))
            ),
            absolute_score_rms_change=float(
                np.sqrt(np.mean((scores - previous_scores) ** 2))
            ),
            relative_score_rms_change=float(
                np.linalg.norm(scores - previous_scores)
                / max(1.0, np.linalg.norm(previous_scores))
            ),
            relative_loss_change=abs(loss - previous["expected_loss"])
            / max(1.0, abs(previous["expected_loss"])),
            max_relative_bin_change=float(
                np.max(
                    abs(np.array(binned["values"]) - old_bins)
                    / np.maximum(1.0, abs(old_bins))
                )
            ),
            absolute_upper_gap_change=abs(
                binned["high_minus_penultimate"]
                - previous["curve"]["high_minus_penultimate"]
            ),
        )
    return result, scores


def optimize(
    u: np.ndarray,
    v: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    groups: np.ndarray,
    config: Config,
    balanced: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n, rank = u.shape
    first, previous_scores = snapshot(u, v, positive, negative, groups, None, None)
    records = [{"block": 0, "cumulative_iterations": 0, **first}]
    consecutive, iterations = 0, 0
    for block in range(
        1, (config.max_blocks if balanced else config.control_blocks) + 1
    ):
        invariance = {}
        if balanced:
            before = u @ v.T
            u, v, _ = balanced_factors(u, v)
            after = u @ v.T
            invariance["relative_score_error"] = float(
                np.linalg.norm(after - before) / max(1.0, np.linalg.norm(before))
            )
            np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-8)
        result = minimize(
            original.loss_gradient,
            np.stack([u, v]).ravel(),
            args=(positive, negative, rank),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": config.block_iterations,
                **{
                    name: getattr(config, name)
                    for name in ("maxfun", "maxls", "maxcor", "gtol", "ftol")
                },
            },
        )
        u, v = result.x.reshape(2, n, rank)
        current, scores = snapshot(
            u, v, positive, negative, groups, records[-1], previous_scores
        )
        if (
            not np.isfinite(result.fun)
            or result.fun > records[-1]["expected_loss"] + 1e-9
        ):
            raise ValueError("A continuation produced nonfinite/increased loss.")
        iterations += int(result.nit)
        current.update(
            block=block,
            cumulative_iterations=iterations,
            optimizer={
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
            },
            balancing=invariance,
            stability_gates=stable_gates(current, config),
        )
        records.append(current)
        previous_scores = scores
        consecutive = consecutive + 1 if all(current["stability_gates"].values()) else 0
        print(
            f"    block {block}: loss={result.fun:.7f}, "
            f"grad={current['balanced_gradient_max']:.3g}, stable={consecutive}",
            flush=True,
        )
        if balanced and consecutive >= 2 and block >= config.control_blocks:
            break
    return (
        {
            "balanced_restart": balanced,
            "stability_reached": consecutive >= 2,
            "snapshots": records,
        },
        u,
        v,
    )


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if original.sha(root / name) != expected:
            raise ValueError(f"Changed input: {root / name}")


@threadpool_limits.wrap(limits=1)
def run_seed(config: Config, seed: int) -> None:
    output, reference = config.root / config.output, config.root / config.reference
    graph = original.read_json(reference / f"graphs/seed{seed}.json")
    degrees = [
        np.array(original.read_json(reference / f"graphs/seed{s}.json")["degree"])
        for s in config.seeds
    ]
    groups = bins_for(degrees)[seed]
    adjacency, _, degree = original.graph_arrays(
        original.configuration_graph(config.nodes, config.attachment, seed)
    )
    assert original.array_sha(adjacency) == graph["graph_sha256"]
    windows = original.uniform_windows(adjacency / degree[:, None], config.windows)
    positive_by_window = {w: p / len(degree) for w, p in windows.items()}
    negative = np.broadcast_to(
        config.negatives * (degree / degree.sum())[None, :] / len(degree),
        adjacency.shape,
    )
    with threadpool_limits(limits=1):
        for length in config.windows:
            for offset in config.offsets:
                name = f"seed{seed}_L{length}_init{offset}"
                reference_run = original.read_json(
                    reference / "runs" / f"{name}.json.gz"
                )
                assert (
                    original.array_sha(positive_by_window[length])
                    == (reference_run["positive_weights_sha256"])
                )
                assert (
                    original.array_sha(negative)
                    == (reference_run["negative_weights_sha256"])
                )
                initial = np.load(
                    io.BytesIO(
                        gzip.decompress(
                            (reference / reference_run["factor_file"]).read_bytes()
                        )
                    ),
                    allow_pickle=False,
                )
                for method in ["balanced", "raw"] if length == 10 else ["balanced"]:
                    destination = output / f"{name}_{method}"
                    marker = destination.with_suffix(".sha256")
                    if marker.exists():
                        verify_hashes(output, original.read_json(marker))
                        continue
                    print(f"Continuing {name} {method}", flush=True)
                    result, u, v = optimize(
                        initial[0].copy(),
                        initial[1].copy(),
                        positive_by_window[length],
                        negative,
                        groups,
                        config,
                        method == "balanced",
                    )
                    factors = destination.with_suffix(".npy.gz")
                    run = destination.with_suffix(".json.gz")
                    original.write_factors(factors, u, v)
                    result.update(
                        seed=seed,
                        window=length,
                        initialization=seed + offset,
                        reference_run_sha256=original.sha(
                            reference / "runs" / f"{name}.json.gz"
                        ),
                        input_factor_sha256=reference_run["factor_file_sha256"],
                        graph_sha256=graph["graph_sha256"],
                        positive_sha256=original.array_sha(positive_by_window[length]),
                        negative_sha256=original.array_sha(negative),
                        factor_file=factors.name,
                    )
                    original.write_json(run, result)
                    original.write_json(
                        marker,
                        {path.name: original.sha(path) for path in (run, factors)},
                    )


def run(config: Config, seeds: list[int], workers: int) -> None:
    output, reference = config.root / config.output, config.root / config.reference
    verify_hashes(reference, original.read_json(reference / "completion.json"))
    plan = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
            if key not in ("root", "output")
        },
        "reference_completion_sha256": original.sha(reference / "completion.json"),
        "source_sha256": {
            name: original.sha(config.root / name)
            for name in (
                "scripts/window_training/continue_windows.py",
                "scripts/window_training/expected_window.py",
                "scripts/window_training/report.py",
                "scripts/simulate_exact_center_exposure.py",
                "tests/test_window_continuation.py",
                str(config.protocol),
                "analysis/window_training_exploration_requirements.txt",
            )
        },
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "networkx", "matplotlib", "threadpoolctl")
        },
    }
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Nonempty output lacks a recorded plan.")
    output.mkdir(exist_ok=True, parents=True)
    path = output / "plan.json"
    if path.exists() and original.read_json(path) != json.loads(json.dumps(plan)):
        raise ValueError("Changed continuation plan; choose fresh output.")
    original.write_json(path, plan)
    if workers == 1:
        for seed in seeds:
            run_seed(config, seed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_seed, [config] * len(seeds), seeds))
    verify_hashes(config.root, plan["source_sha256"])
    if len(list(output.glob("*.sha256"))) == 50:
        original.write_json(
            output / "completion.json",
            {
                p.name: original.sha(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "completion.json"
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(Config.seeds))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if (
        not set(args.seeds) <= set(Config.seeds)
        or len(set(args.seeds)) != len(args.seeds)
        or args.workers < 1
    ):
        parser.error("Select seeds 0..4 and at least one worker.")
    run(Config(output=args.output), args.seeds, args.workers)


if __name__ == "__main__":
    main()
