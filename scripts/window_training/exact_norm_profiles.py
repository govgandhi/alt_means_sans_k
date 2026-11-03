"""Construct distinct normalized balanced profiles for one near-optimal objective.

No optimization or external data: the graph is a triangle plus one pendant node.
Inputs and illustrative profiles/tolerances are declared in Config; the proof is
in paper/current/norm_geometry_supplement.tex. Uses the existing pinned numerical
environment from analysis/window_training_exploration_requirements.txt.
Run: python3 scripts/window_training/exact_norm_profiles.py [--output DIRECTORY]
Writes exact.json and a source/runtime/output manifest. Repeated runs are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    output: Path = Path("results/norm_identifiability/exact")
    edges: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2), (0, 3))
    nodes: int = 4
    negatives: int = 5
    tolerances: tuple[float, ...] = (1e-2, 1e-6, 1e-10)
    degree_profiles: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("rising", (1, 2, 3)),
        ("falling", (3, 2, 1)),
        ("peaked", (1, 3, 1)),
    )


def weights(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adjacency = np.zeros((config.nodes, config.nodes))
    for i, j in config.edges:
        adjacency[i, j] = adjacency[j, i] = 1
    degree = adjacency.sum(axis=1)
    positive = adjacency / degree[:, None] / config.nodes
    negative = np.broadcast_to(
        config.negatives * degree[None, :] / degree.sum() / config.nodes,
        adjacency.shape,
    ).copy()
    return degree, positive, negative


def construct(
    positive: np.ndarray,
    negative: np.ndarray,
    profiles: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, Any]:
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Excess-loss tolerance must be finite and positive.")
    if not profiles or any(
        h.shape != (len(positive),) or np.any(h <= 0) or not np.isfinite(h).all()
        for h in profiles.values()
    ):
        raise ValueError("Every profile must specify a positive finite value per node.")
    zero = positive == 0
    if not np.all(np.diag(zero)) or not np.all(negative > 0):
        raise ValueError(
            "The construction requires zero-positive diagonals and positive noise."
        )
    zero_mass = float(negative[zero].sum())
    tail = max(0.0, float(np.log(zero_mass) - np.log(tolerance)))
    scores = np.full(positive.shape, -tail)
    scores[~zero] = np.log(positive[~zero] / negative[~zero])
    np.fill_diagonal(scores, 0)
    if not np.allclose(scores, scores.T, rtol=0, atol=1e-12):
        raise ValueError("The finite edge-optimum matrix must be symmetric.")
    squared = {name: h**2 / np.mean(h**2) for name, h in profiles.items()}
    # One common scale makes every matrix strictly negative definite.
    scale = max(
        float(np.max(np.maximum(tail, np.abs(scores).sum(axis=1) + 1) / a))
        for a in squared.values()
    )
    optimum = np.zeros_like(scores)
    optimum[~zero] = np.log(positive[~zero] / negative[~zero])
    free_loss = float(
        np.sum(
            positive[~zero] * np.logaddexp(0, -optimum[~zero])
            + negative[~zero] * np.logaddexp(0, optimum[~zero])
        )
    )
    examples = []
    for name, a in squared.items():
        matrix = scores.copy()
        np.fill_diagonal(matrix, -scale * a)
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        radii = np.sqrt(left**2 @ singular)
        rms = float(np.sqrt(np.mean(radii**2)))
        excess = float(np.sum(negative[zero] * np.logaddexp(0, matrix[zero])))
        direct_loss = float(
            np.sum(
                positive * np.logaddexp(0, -matrix) + negative * np.logaddexp(0, matrix)
            )
        )
        if not (
            np.linalg.eigvalsh(matrix).max() < 0
            and excess <= tolerance * (1 + 1e-12)
            and np.allclose(radii**2, scale * a, rtol=1e-12, atol=1e-12)
            and np.isclose(direct_loss, free_loss + excess, rtol=0, atol=1e-13)
        ):
            raise ArithmeticError(
                "Exact construction failed its independent numerical checks."
            )
        examples.append(
            {
                "profile": name,
                "scores": matrix.tolist(),
                "prescribed_profile": profiles[name].tolist(),
                "balanced_radius": radii.tolist(),
                "normalized_radius": (radii / rms).tolist(),
                "radius_rms": rms,
                "excess_loss": excess,
                "loss": direct_loss,
                "largest_eigenvalue": float(np.linalg.eigvalsh(matrix).max()),
            }
        )
    return {
        "tolerance": tolerance,
        "zero_pair_negative_mass": zero_mass,
        "tail": tail,
        "common_diagonal_scale": scale,
        "free_loss_infimum": free_loss,
        "examples": examples,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: Config) -> None:
    degree, positive, negative = weights(config)
    profiles = {
        name: np.asarray(h, dtype=float)[degree.astype(int) - 1]
        for name, h in config.degree_profiles
    }
    scientific_config = asdict(config)
    for key in ("root", "output"):
        scientific_config.pop(key)
    output = config.root / config.output
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "config": scientific_config,
        "degree": degree.tolist(),
        "positive_weights": positive.tolist(),
        "negative_weights": negative.tolist(),
        "runs": [
            construct(positive, negative, profiles, delta)
            for delta in config.tolerances
        ],
    }
    (output / "exact.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    source = Path(__file__).resolve()
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": {str(source.relative_to(config.root)): digest(source)},
        "outputs_sha256": {"exact.json": digest(output / "exact.json")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "constructions": len(config.tolerances) * len(profiles),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    args = parser.parse_args()
    run(Config(output=args.output))
