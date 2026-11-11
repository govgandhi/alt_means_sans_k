"""Audit exact population and realized-graph targets without training embeddings.

Inputs: the fixtures in analysis/realized_graph_audit_protocol.md and the unchanged
configuration-graph generator and integer pair allocator. Outputs: deterministic
JSON with source/runtime/graph provenance, every node's window and allocation
diagnostics, exact mathematical checks, and a figure of graphwise KL statistics.

Run with the pinned requirements in analysis/realized_graph_audit_requirements.txt:
    python3 scripts/audit_realized_graph.py --output results/realized_graph_audit
No corpus, optimizer, score smoothing, or pseudocount is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymupdf
import sympy as sp
from simulate_exact_center_exposure import configuration_graph, graph_arrays
from simulate_turnover_surface import (
    Condition,
    allocate_pairs,
    row_kl,
)
from simulate_turnover_surface import (
    Config as AllocationConfig,
)


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    output: Path = Path("results/realized_graph_audit")
    nodes: int = 600
    attachment: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    windows: tuple[int, ...] = (1, 2, 5, 10)
    events_per_node: int = 240
    negatives: int = 5
    alphas: tuple[float, ...] = (0.75, 1.0, 1.25)
    nodes_per_block: int = 8
    theta_min: float = 0.5
    theta_max: float = 1.5
    block_matrices: tuple[tuple[tuple[float, ...], ...], ...] = (
        ((0.2,),),
        ((0.18, 0.04, 0.03), (0.04, 0.16, 0.05), (0.03, 0.05, 0.20)),
    )
    tolerance: float = 1e-10


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_small(value: float, tolerance: float) -> float:
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f"Identity residual {value} exceeds {tolerance}.")
    return float(value)


def pairwise_scores(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Return finite optima, boundary +/- infinity, or NaN for zero/zero weights."""
    positive, negative = np.broadcast_arrays(positive, negative)
    if np.any(positive < 0) or np.any(negative < 0):
        raise ValueError("Logistic weights must be nonnegative.")
    scores = np.full(positive.shape, np.nan)
    both = (positive > 0) & (negative > 0)
    scores[both] = np.log(positive[both] / negative[both])
    scores[(positive == 0) & (negative > 0)] = -np.inf
    scores[(positive > 0) & (negative == 0)] = np.inf
    return scores


def uniform_windows(
    transition: np.ndarray, windows: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Average lags 1 through L; never include lag zero or just the last power."""
    if not windows or min(windows) < 1:
        raise ValueError("Windows must be positive integers.")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("Transition rows must be probability distributions.")
    power = np.eye(len(transition))
    accumulated = np.zeros_like(transition, dtype=float)
    result = {}
    for lag in range(1, max(windows) + 1):
        power = power @ transition
        accumulated += power
        if lag in windows:
            result[lag] = accumulated / lag
    return result


def balanced_factors(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimal balanced SVD factors of a finite matrix; reject support infinities."""
    if not np.isfinite(scores).all():
        raise ValueError(
            "An infinite/undefined target has no finite exact SVD factors."
        )
    left, singular, right = np.linalg.svd(scores, full_matrices=False)
    threshold = max(scores.shape) * np.finfo(float).eps * singular[0]
    retain = singular > threshold
    scale = np.sqrt(singular[retain])
    return left[:, retain] * scale, right[retain].T * scale


def symbolic_checks() -> dict[str, object]:
    p, n = sp.symbols("p n", positive=True)
    x = sp.symbols("x", real=True)
    objective = -p * sp.log(1 + sp.exp(-x)) - n * sp.log(1 + sp.exp(x))
    derivative = sp.diff(objective, x)
    curvature = sp.diff(objective, x, 2)
    residuals = {
        "derivative": sp.simplify(derivative - (p - n * sp.exp(x)) / (1 + sp.exp(x))),
        "curvature": sp.simplify(
            curvature + (p + n) * sp.exp(x) / (1 + sp.exp(x)) ** 2
        ),
        "stationary_point": sp.simplify(derivative.subs(x, sp.log(p / n))),
    }
    c, r, q, k = sp.symbols("c r q k", positive=True)
    residuals["center_cancellation"] = sp.simplify(c * r / (k * c * q) - r / (k * q))
    residuals["skipped_collision_ratio"] = sp.simplify(
        c * r / (k * q * (c - c * r)) - r / (k * q * (1 - r))
    )
    if any(value != 0 for value in residuals.values()):
        raise AssertionError("A symbolic identity failed.")
    return {
        "identity_residuals": {name: str(value) for name, value in residuals.items()},
        "interior_target": "log(R_ij / (k q_j)); c_i,R_ij,q_j,k > 0",
        "skipped_collision_target": "log(R_ij / (k q_j (1-R_ij))); 0<R_ij<1",
        "boundaries": {
            "positive=0,negative>0": "-infinity",
            "positive>0,negative=0": "+infinity",
            "both=0": "unidentified",
        },
        "positive_only_limit": str(sp.limit(objective.subs(n, 0), x, sp.oo)),
        "negative_only_limit": str(sp.limit(objective.subs(p, 0), x, -sp.oo)),
    }


def population_checks(config: Config) -> list[dict[str, object]]:
    results = []
    for values in config.block_matrices:
        block = np.asarray(values)
        communities = len(block)
        labels = np.repeat(np.arange(communities), config.nodes_per_block)
        theta = np.tile(
            np.linspace(config.theta_min, config.theta_max, config.nodes_per_block),
            communities,
        )
        omega = theta[:, None] * block[labels[:, None], labels] * theta[None, :]
        degree = omega.sum(axis=1)
        stationary = degree / degree.sum()
        transition = omega / degree[:, None]
        masses = np.bincount(labels, weights=theta)
        kappa = block @ masses
        normalized = omega / np.sqrt(degree[:, None] * degree[None, :])
        eigenvalues, vectors = np.linalg.eigh(normalized)
        vectors = vectors[:, np.argsort(np.abs(eigenvalues))[-communities:]]
        radius = np.linalg.norm(vectors, axis=1)
        kl = row_kl(transition, stationary)
        block_transition = block * masses[None, :] / kappa[:, None]
        block_stationary = masses * kappa / degree.sum()
        block_kl = row_kl(block_transition, block_stationary)
        checks = {
            "spectral_radius_error": np.max(
                abs(radius - np.sqrt(degree / (masses[labels] * kappa[labels])))
            ),
            "block_kl_error": np.max(abs(kl - block_kl[labels])),
            "row_sum_error": np.max(abs(transition.sum(axis=1) - 1)),
        }
        if communities == 1:
            checks["k1_transition_equals_stationary_error"] = np.max(
                abs(transition - stationary[None, :])
            )
            checks["k1_kl_zero_error"] = np.max(abs(kl))
        score_results = []
        block_windows = uniform_windows(block_transition, config.windows)
        for length, conditional in uniform_windows(transition, config.windows).items():
            window = block_windows[length]
            expected_contexts = (
                theta[None, :]
                / masses[labels][None, :]
                * window[labels[:, None], labels]
            )
            for alpha in config.alphas:
                normalizer = np.sum(degree**alpha)
                noise = degree**alpha / normalizer
                scores = pairwise_scores(conditional, config.negatives * noise[None, :])
                center, context = balanced_factors(scores)
                first = np.arange(communities) * config.nodes_per_block
                block_score = (
                    np.log(window)
                    - np.log(masses)[None, :]
                    - alpha * np.log(kappa)[None, :]
                    + np.log(normalizer)
                    - np.log(config.negatives)
                )
                predicted = (
                    block_score[labels[:, None], labels]
                    + (1 - alpha) * np.log(theta)[None, :]
                )
                errors = {
                    "window_endpoint_error": np.max(
                        abs(conditional - expected_contexts)
                    ),
                    "window_block_kl_error": np.max(
                        abs(
                            row_kl(conditional, stationary)
                            - row_kl(window, block_stationary)[labels]
                        )
                    ),
                    "context_log_propensity_formula_error": np.max(
                        abs(scores - predicted)
                    ),
                    "balanced_factor_reconstruction_error": np.max(
                        abs(center @ context.T - scores)
                    ),
                    "balanced_gram_error": np.max(
                        abs(center.T @ center - context.T @ context)
                    ),
                    "center_block_row_error": np.max(
                        abs(center - center[first][labels])
                    ),
                }
                slopes = [
                    float(
                        np.polyfit(
                            np.log(theta[labels == group]),
                            scores[0, labels == group],
                            1,
                        )[0]
                    )
                    for group in range(communities)
                ]
                errors["context_slope_error"] = np.max(
                    abs(np.asarray(slopes) - (1 - alpha))
                )
                if alpha == 1:
                    errors["score_block_error"] = np.max(
                        abs(
                            scores
                            - scores[first[:, None], first][labels[:, None], labels]
                        )
                    )
                    errors["context_block_row_error"] = np.max(
                        abs(context - context[first][labels])
                    )
                    if communities == 1:
                        errors["k1_constant_score_error"] = np.max(
                            abs(scores + np.log(config.negatives))
                        )
                score_results.append(
                    {
                        "window": length,
                        "alpha": alpha,
                        "rank": center.shape[1],
                        "context_slopes": slopes,
                        "checks": {
                            key: require_small(float(value), config.tolerance)
                            for key, value in errors.items()
                        },
                    }
                )
        results.append(
            {
                "communities": communities,
                "labels": labels.tolist(),
                "theta": theta.tolist(),
                "degree": degree.tolist(),
                "spectral_radius": radius.tolist(),
                "context_kl": kl.tolist(),
                "checks": {
                    key: require_small(float(value), config.tolerance)
                    for key, value in checks.items()
                },
                "scores": score_results,
            }
        )
    return results


def counterexamples(config: Config) -> dict[str, object]:
    edge = np.array([[0.0, 1.0], [1.0, 0.0]])
    edge_kl = {
        str(lag): row_kl(rows, np.full(2, 0.5)).tolist()
        for lag, rows in uniform_windows(edge, (1, 2, 3)).items()
    }
    triangle = nx.Graph([(0, 3), (1, 2), (1, 3), (2, 3)])
    adjacency = nx.to_numpy_array(triangle, nodelist=range(4))
    degree = adjacency.sum(axis=1)
    triangle_kl = {
        str(lag): row_kl(rows, degree / degree.sum()).tolist()
        for lag, rows in uniform_windows(adjacency / degree[:, None], (2, 3)).items()
    }
    star = nx.star_graph(3)
    star_kl = []
    for added in (False, True):
        if added:
            star.add_edge(1, 2)
        adjacency = nx.to_numpy_array(star, nodelist=range(4))
        degree = adjacency.sum(axis=1)
        star_kl.append(
            float(row_kl(adjacency / degree[:, None], degree / degree.sum())[3])
        )
    transition = np.array([[1 / 5, 4 / 5], [1 / 20, 19 / 20]])
    stationary = np.array([1 / 17, 16 / 17])
    kl = row_kl(transition, stationary)
    scores = pairwise_scores(transition, config.negatives * stationary[None, :])
    left, right = balanced_factors(scores)
    norms = np.linalg.norm(left, axis=1)
    eigenvalues, eigenvectors = np.linalg.eigh(scores @ scores.T)
    independent_norms = np.sqrt(eigenvectors**2 @ np.sqrt(np.maximum(eigenvalues, 0)))
    require_small(
        float(
            np.max(
                abs(
                    stationary[:, None] * transition
                    - stationary[None, :] * transition.T
                )
            )
        ),
        config.tolerance,
    )
    if not (
        edge_kl["3"][0] > edge_kl["2"][0]
        and triangle_kl["3"][3] > triangle_kl["2"][3]
        and star_kl[1] > star_kl[0]
        and kl[0] > kl[1]
        and norms[0] < norms[1]
    ):
        raise AssertionError(
            "A proposed counterexample did not exhibit the stated ordering."
        )
    return {
        "periodic_edge": {"edges": [[0, 1]], "node_kl_by_window": edge_kl},
        "aperiodic_triangle_leaf": {
            "edges": [[0, 3], [1, 2], [1, 3], [2, 3]],
            "node_kl_by_window": triangle_kl,
            "nonmonotone_node": 3,
        },
        "edge_addition": {
            "initial_edges": [[0, 1], [0, 2], [0, 3]],
            "added_edge": [1, 2],
            "untouched_node": 3,
            "kl_before_after": star_kl,
        },
        "kl_norm_reversal": {
            "transition": transition.tolist(),
            "stationary": stationary.tolist(),
            "negatives": config.negatives,
            "kl": kl.tolist(),
            "score": scores.tolist(),
            "balanced_center_norm": norms.tolist(),
            "independent_norm_error": require_small(
                float(np.max(abs(norms - independent_norms))), config.tolerance
            ),
            "reconstruction_error": require_small(
                float(np.max(abs(left @ right.T - scores))), config.tolerance
            ),
        },
    }


def graph_diagnostics(
    config: Config,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], float
]:
    windows, allocations, graphs = [], [], []
    all_degree, all_kl = [], []
    allocation_config = AllocationConfig(
        nodes=config.nodes, events_per_node=config.events_per_node
    )
    for seed in config.seeds:
        graph = configuration_graph(config.nodes, config.attachment, seed)
        adjacency, _, degree = graph_arrays(graph)
        transition = adjacency / degree[:, None]
        stationary = degree / degree.sum()
        edge_rows, edge_columns = np.nonzero(adjacency)
        edge_scores = pairwise_scores(
            transition, config.negatives * stationary[None, :]
        )
        expected_edges = np.log(
            degree.sum() / (config.negatives * degree[edge_rows] * degree[edge_columns])
        )
        graph_checks = {
            "transition_row_sum_error": np.max(abs(transition.sum(axis=1) - 1)),
            "stationarity_error": np.max(abs(stationary @ transition - stationary)),
            "edge_score_formula_error": np.max(
                abs(edge_scores[edge_rows, edge_columns] - expected_edges)
            ),
        }
        if not np.isneginf(edge_scores[adjacency == 0]).all():
            raise AssertionError("Nonedges must retain their infinite target boundary.")
        graph_record = {
            "seed": seed,
            "nodes": len(degree),
            "edges": graph.number_of_edges(),
            "adjacency_sha256": digest(adjacency.astype("<f8").tobytes()),
            "adjacency_hash_encoding": "row-major little-endian float64",
            "degree": degree.astype(int).tolist(),
            "positive_support": len(edge_rows),
            "minus_infinity_targets": int(np.sum(adjacency == 0)),
            "checks": {
                key: require_small(float(value), config.tolerance)
                for key, value in graph_checks.items()
            },
        }
        graphs.append(graph_record)
        for length, rows in uniform_windows(transition, config.windows).items():
            marginal = rows.mean(axis=0)
            kl = row_kl(rows, stationary)
            marginal_kl = row_kl(rows, marginal)
            support = (rows > 0).sum(axis=1)
            windows.append(
                {
                    "seed": seed,
                    "window": length,
                    "mean_kl_to_fixed_noise": float(np.mean(kl)),
                    "slope_kl_on_log_degree": float(
                        np.polyfit(np.log(degree), kl, 1)[0]
                    ),
                    "mean_kl_to_positive_marginal": float(np.mean(marginal_kl)),
                    "nodes": [
                        {
                            "node": node,
                            "degree": int(degree[node]),
                            "center_share": 1 / len(degree),
                            "fixed_noise": float(stationary[node]),
                            "positive_marginal": float(marginal[node]),
                            "kl_to_fixed_noise": float(kl[node]),
                            "kl_to_positive_marginal": float(marginal_kl[node]),
                            "positive_support": int(support[node]),
                            "minus_infinity_targets": int(len(degree) - support[node]),
                        }
                        for node in range(len(degree))
                    ],
                }
            )
            if length == 1:
                all_degree.extend(degree)
                all_kl.extend(kl)
                edge_rows, edge_columns = np.nonzero(adjacency)
                formula = (
                    np.bincount(
                        edge_rows,
                        weights=np.log(
                            degree.sum() / (degree[edge_rows] * degree[edge_columns])
                        ),
                        minlength=len(degree),
                    )
                    / degree
                )
                graph_record["checks"]["graph_kl_formula_error"] = require_small(
                    float(np.max(abs(formula - kl))), config.tolerance
                )
        for beta in (0.0, 1.0):
            for mixing in (0.0, 1.0):
                condition = Condition(
                    f"beta_{beta:g}_lambda_{mixing:g}", beta, mixing, None, False
                )
                (
                    centers,
                    contexts,
                    counts,
                    noise,
                    intended_kl,
                    empirical_kl,
                    row_error,
                    marginal_error,
                ) = allocate_pairs(transition, stationary, allocation_config, condition)
                intended_marginal = (counts / counts.sum()) @ transition
                intended = (1 - mixing) * transition + mixing * intended_marginal[
                    None, :
                ]
                matrix = np.zeros_like(adjacency, dtype=np.int64)
                np.add.at(matrix, (centers, contexts), 1)
                empirical = matrix / counts[:, None]
                usable = noise > 0
                direct_kl = np.sum(
                    np.where(
                        empirical[:, usable] > 0,
                        empirical[:, usable]
                        * np.log(
                            np.divide(
                                empirical[:, usable],
                                noise[None, usable],
                                out=np.ones_like(empirical[:, usable]),
                                where=empirical[:, usable] > 0,
                            )
                        ),
                        0,
                    ),
                    axis=1,
                )
                require_small(
                    float(np.max(abs(direct_kl - empirical_kl))), config.tolerance
                )
                allocations.append(
                    {
                        "seed": seed,
                        "beta": beta,
                        "lambda": mixing,
                        "events": len(centers),
                        "maximum_row_error": row_error,
                        "maximum_marginal_error": marginal_error,
                        "zero_noise_nodes": int(np.sum(~usable)),
                        "intended_positive_entries": int(np.sum(intended > 0)),
                        "realized_positive_entries": int(np.sum(matrix > 0)),
                        "lost_positive_entries": int(
                            np.sum((intended > 0) & (matrix == 0))
                        ),
                        "nodes": [
                            {
                                "node": node,
                                "center_count": int(counts[node]),
                                "intended_marginal": float(intended_marginal[node]),
                                "realized_marginal": float(noise[node]),
                                "intended_kl": float(intended_kl[node]),
                                "realized_kl": float(empirical_kl[node]),
                                "maximum_row_error": float(
                                    np.max(abs(empirical[node] - intended[node]))
                                ),
                                "intended_support": int(np.sum(intended[node] > 0)),
                                "realized_support": int(np.sum(matrix[node] > 0)),
                                "minus_infinity_targets": int(
                                    np.sum((matrix[node] == 0) & usable)
                                ),
                                "unidentified_targets": int(
                                    np.sum((matrix[node] == 0) & ~usable)
                                ),
                            }
                            for node in range(len(degree))
                        ],
                    }
                )
        print(f"Audited graph seed {seed}; no training.", flush=True)
    return (
        windows,
        allocations,
        graphs,
        float(np.polyfit(np.log(all_degree), all_kl, 1)[0]),
    )


def plot_windows(windows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in windows}):
        selected = [row for row in windows if row["seed"] == seed]
        lengths = [row["window"] for row in selected]
        for axis, field in zip(
            axes, ("mean_kl_to_fixed_noise", "slope_kl_on_log_degree"), strict=True
        ):
            axis.plot(
                lengths,
                [row[field] for row in selected],
                marker="o",
                label=f"Graph {seed}",
            )
            axis.set(xlabel="Uniform window L", xticks=lengths)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean context KL to fixed noise π")
    axes[1].set_ylabel("Slope of context KL on log degree")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exact realized-graph targets; no embedding training")
    fig.savefig(output, dpi=180, metadata={"Software": "audit_realized_graph.py"})
    plt.close(fig)


def qualify_historical_figure(config: Config, output: Path) -> dict[str, object]:
    """Replace three historical PDF labels; retain all data paths and other text."""
    source = config.root / "results/turnover_surface/turnover_surface.pdf"
    old_title = "Causal two-force test for the norm-frequency turnover"
    new_title = "Exposure and context interventions for norm turnover"
    replacements = {
        old_title: new_title,
        "B  Specificity interventions": "B  Intended context KL",
        "C  Exposure creates the rising limb": "C  Norm response to exposure",
    }
    pymupdf.TOOLS.set_small_glyph_heights(True)
    with pymupdf.open(source) as document:
        if len(document) != 1:
            raise ValueError("The source figure must be a single PDF page.")
        page = document[0]
        before_text = Counter(page.get_text().splitlines())
        drawings = page.get_drawings()
        changes = []
        for old, new in replacements.items():
            spans = [
                span
                for block in page.get_text("dict")["blocks"]
                for line in block.get("lines", [])
                for span in line["spans"]
                if span["text"] == old
            ]
            if len(spans) != 1:
                raise ValueError(f"Expected exactly one figure label: {old}")
            span = spans[0]
            page.add_redact_annot(span["bbox"], fill=None)
            changes.append((span, new))
        label_regions = [pymupdf.Rect(span["bbox"]) for span, _ in changes]

        def outside_labels(paths: list[dict]) -> list[dict]:
            # Type-3 font glyphs also appear as vector paths.
            return [
                {key: value for key, value in path.items() if key != "seqno"}
                for path in paths
                if not any(path["rect"].intersects(region) for region in label_regions)
            ]

        before_paths = outside_labels(drawings)
        page.apply_redactions(images=0, graphics=0)
        for span, new in changes:
            width = pymupdf.get_text_length(new, fontname="helv", fontsize=span["size"])
            left = (span["bbox"][0] + span["bbox"][2] - width) / 2
            page.insert_text(
                (left, span["origin"][1]),
                new,
                fontname="helv",
                fontsize=span["size"],
            )
        expected_text = before_text.copy()
        for old, new in replacements.items():
            expected_text[old] -= 1
            expected_text[new] += 1
        if +expected_text != Counter(page.get_text().splitlines()):
            raise AssertionError("Figure edit changed text outside the three labels.")
        after_paths = outside_labels(page.get_drawings())
        if before_paths != after_paths:
            raise AssertionError("Figure edit changed plotted vector paths.")
        (output / "turnover_surface_qualified.pdf").write_bytes(
            document.tobytes(garbage=4, deflate=True, no_new_id=True)
        )
    return {
        "source": str(source.relative_to(config.root)),
        "source_sha256": digest(source.read_bytes()),
        "replacements": replacements,
        "unchanged_paths_outside_labels": len(before_paths),
        "other_text_unchanged": True,
    }


def run(config: Config) -> None:
    output = config.root / config.output
    output.mkdir(parents=True, exist_ok=True)
    parameters = asdict(config)
    parameters["root"] = "."
    parameters["output"] = str(config.output)
    sources = (
        "scripts/audit_realized_graph.py",
        "tests/test_realized_graph_audit.py",
        "scripts/simulate_exact_center_exposure.py",
        "scripts/simulate_turnover_surface.py",
        "results/turnover_surface/turnover_surface.pdf",
        "analysis/realized_graph_audit_protocol.md",
        "analysis/realized_graph_audit_requirements.txt",
    )
    runtime = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "networkx",
                "matplotlib",
                "sympy",
                "mpmath",
                "pytest",
                "PyMuPDF",
            )
        },
    }
    checks = {
        "symbolic": symbolic_checks(),
        "population": population_checks(config),
        "counterexamples": counterexamples(config),
        "historical_figure": qualify_historical_figure(config, output),
    }
    windows, allocations, graphs, slope = graph_diagnostics(config)
    old_summary = config.root / "results/norm_frequency_simulation/summary.json"
    stored_slope = json.loads(old_summary.read_text())["population"][
        "pooled_kl_slope_on_log_degree"
    ]
    checks["historical_realized_graph_kl"] = {
        "recomputed_slope": slope,
        "stored_slope": stored_slope,
        "absolute_difference": require_small(
            abs(slope - stored_slope), config.tolerance
        ),
        "source": str(old_summary.relative_to(config.root)),
        "source_sha256": digest(old_summary.read_bytes()),
        "meaning": (
            "Exact one-step KL on realized simple graph adjacencies, "
            "not DCSBM population KL."
        ),
    }
    artifacts = {
        "checks.json": checks,
        "windows.json.gz": windows,
        "allocations.json.gz": allocations,
        "graphs.json": graphs,
    }
    for name, value in artifacts.items():
        data = encoded(value)
        (output / name).write_bytes(
            gzip.compress(data, mtime=0) if name.endswith(".gz") else data
        )
    figure = output / "window_kl.png"
    plot_windows(windows, figure)
    provenance = {
        "config": parameters,
        "config_sha256": digest(encoded(parameters)),
        "runtime": runtime,
        "runtime_sha256": digest(encoded(runtime)),
        "source_sha256": {
            name: digest((config.root / name).read_bytes()) for name in sources
        },
        "output_sha256": {
            name: digest((output / name).read_bytes())
            for name in (*artifacts, figure.name, "turnover_surface_qualified.pdf")
        },
        "scope": (
            "Retrospective exact-matrix and allocation audit; no training, "
            "convergence claim, density sweep, or norm prediction."
        ),
    }
    (output / "provenance.json").write_bytes(encoded(provenance))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    args = parser.parse_args()
    run(Config(output=args.output))


if __name__ == "__main__":
    main()
