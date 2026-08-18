# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "matplotlib>=3.10,<4",
#   "networkx>=3.4,<4",
#   "numpy>=2.2,<3",
#   "scipy>=1.15,<2",
# ]
# ///
"""Exactly control positive center updates in a 2-by-2 SGNS experiment.

The two factors are:

1. positive center updates proportional to stationary frequency versus exactly
   equal across nodes; and
2. Gensim-style frequent-context subsampling off versus on.

Subsampling is applied only to the positive context distribution. Final center
counts are imposed afterward. This keeps center exposure and context thinning
as distinct interventions; applying subsampling to centers would alias them.

Run the preregistered defaults with

    uv run scripts/simulate_exact_center_exposure.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr


@dataclass(frozen=True)
class Config:
    nodes: int = 600
    attachment: int = 3
    dimension: int = 32
    negatives: int = 5
    events_per_node: int = 300
    epochs: int = 5
    seeds: int = 5
    batch_size: int = 128
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0025
    subsample: float = 1e-3


@dataclass(frozen=True)
class Condition:
    name: str
    exposure: str
    context_subsampling: bool


CONDITIONS = (
    Condition("proportional_off", "proportional", False),
    Condition("proportional_on", "proportional", True),
    Condition("equalized_off", "equalized", False),
    Condition("equalized_on", "equalized", True),
)


def configuration_graph(nodes: int, attachment: int, seed: int) -> nx.Graph:
    """Return a simple configuration graph with a broad degree sequence."""
    template = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    degree_sequence = [degree for _, degree in template.degree()]
    multigraph = nx.configuration_model(degree_sequence, seed=seed)
    graph = nx.Graph(multigraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if not nx.is_connected(graph):
        components = [sorted(component) for component in nx.connected_components(graph)]
        for left, right in zip(components, components[1:], strict=False):
            graph.add_edge(left[0], right[0])
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def graph_arrays(
    graph: nx.Graph,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    n = graph.number_of_nodes()
    adjacency = np.zeros((n, n), dtype=np.float64)
    for source, target in graph.edges():
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(axis=1)
    if np.any(degree == 0):
        raise ValueError("The simulation requires a graph without isolates.")
    neighbors = [np.flatnonzero(adjacency[node]) for node in range(n)]
    return adjacency, neighbors, degree


def context_kl(adjacency: np.ndarray, degree: np.ndarray) -> np.ndarray:
    """Return D_KL(T_i || pi) for the one-step simple random walk."""
    volume = float(degree.sum())
    rows, cols = np.nonzero(adjacency)
    edge_terms = np.log(volume / (degree[rows] * degree[cols]))
    return np.bincount(rows, weights=edge_terms, minlength=len(degree)) / degree


def exact_allocate(weights: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder allocation with an exact integer total."""
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Allocation requires a nonnegative total and positive weight.")
    expected = weights / weights.sum() * total
    allocated = np.floor(expected).astype(np.int64)
    remainder = int(total - allocated.sum())
    if remainder:
        order = np.argsort(expected - allocated, kind="stable")
        allocated[order[-remainder:]] += 1
    if int(allocated.sum()) != total:
        raise AssertionError("Largest-remainder allocation changed the total.")
    return allocated


def gensim_keep_probability(
    token_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Gensim/Mikolov retention probability expressed using token shares."""
    keep = (
        (np.sqrt(token_probability / threshold) + 1.0) * threshold / token_probability
    )
    return np.minimum(keep, 1.0)


def positive_pairs(
    neighbors: list[np.ndarray],
    stationary: np.ndarray,
    config: Config,
    condition: Condition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(neighbors)
    total_events = n * config.events_per_node
    if condition.exposure == "equalized":
        center_counts = np.full(n, config.events_per_node, dtype=np.int64)
    elif condition.exposure == "proportional":
        center_counts = exact_allocate(stationary, total_events)
    else:
        raise ValueError(f"Unknown exposure condition: {condition.exposure}")

    if condition.context_subsampling:
        keep_probability = gensim_keep_probability(
            stationary,
            config.subsample,
        )
    else:
        keep_probability = np.ones(n, dtype=np.float64)

    center_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    maximum_context_error = 0.0
    for center, count in enumerate(center_counts):
        local_neighbors = neighbors[center]
        weights = keep_probability[local_neighbors]
        context_counts = exact_allocate(weights, int(count))
        center_parts.append(np.full(int(count), center, dtype=np.int64))
        context_parts.append(np.repeat(local_neighbors, context_counts))
        empirical = context_counts / max(int(count), 1)
        target = weights / weights.sum()
        maximum_context_error = max(
            maximum_context_error,
            float(np.max(np.abs(empirical - target))),
        )

    centers = np.concatenate(center_parts)
    contexts = np.concatenate(context_parts)
    realized_counts = np.bincount(centers, minlength=n)
    if not np.array_equal(realized_counts, center_counts):
        raise AssertionError("Realized center counts differ from assigned counts.")
    return (
        centers,
        contexts,
        center_counts,
        keep_probability,
        maximum_context_error,
    )


def balanced_left_norm(
    center_vectors: np.ndarray,
    context_vectors: np.ndarray,
) -> np.ndarray:
    """Return the left norm in the canonical balanced gauge of U V^T."""
    left_basis, left_triangular = np.linalg.qr(center_vectors, mode="reduced")
    _, right_triangular = np.linalg.qr(context_vectors, mode="reduced")
    core_left, singular_values, _ = np.linalg.svd(
        left_triangular @ right_triangular.T,
        full_matrices=False,
    )
    balanced = left_basis @ (
        core_left * np.sqrt(np.maximum(singular_values, 0.0))[None, :]
    )
    return np.linalg.norm(balanced, axis=1)


def train_sgns(
    centers: np.ndarray,
    contexts: np.ndarray,
    noise_distribution: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mini-batch SGNS with the standard Word2Vec initialization."""
    rng = np.random.default_rng(seed)
    n = len(noise_distribution)
    center_vectors = rng.uniform(
        -0.5 / config.dimension,
        0.5 / config.dimension,
        size=(n, config.dimension),
    )
    context_vectors = np.zeros((n, config.dimension), dtype=np.float64)
    order = np.arange(len(centers))
    batches_per_epoch = math.ceil(len(order) / config.batch_size)
    total_steps = config.epochs * batches_per_epoch
    step = 0

    for _ in range(config.epochs):
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            center_ids = centers[batch]
            context_ids = contexts[batch]
            negative_ids = rng.choice(
                n,
                size=(len(batch), config.negatives),
                p=noise_distribution,
            )

            center = center_vectors[center_ids].copy()
            positive = context_vectors[context_ids].copy()
            negative = context_vectors[negative_ids].copy()
            positive_score = np.sum(center * positive, axis=1)
            negative_score = np.einsum("bd,bkd->bk", center, negative)
            positive_weight = expit(-positive_score)
            negative_weight = expit(negative_score)

            center_gradient = positive_weight[:, None] * positive
            center_gradient -= np.einsum(
                "bk,bkd->bd",
                negative_weight,
                negative,
            )
            positive_gradient = positive_weight[:, None] * center
            negative_gradient = -negative_weight[:, :, None] * center[:, None, :]

            progress = step / max(total_steps - 1, 1)
            rate = config.learning_rate + progress * (
                config.min_learning_rate - config.learning_rate
            )
            np.add.at(center_vectors, center_ids, rate * center_gradient)
            np.add.at(context_vectors, context_ids, rate * positive_gradient)
            for negative_index in range(config.negatives):
                np.add.at(
                    context_vectors,
                    negative_ids[:, negative_index],
                    rate * negative_gradient[:, negative_index],
                )
            step += 1

    return center_vectors, context_vectors


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    tolerance_x = np.finfo(float).eps * max(1.0, float(np.max(np.abs(x))))
    tolerance_y = np.finfo(float).eps * max(1.0, float(np.max(np.abs(y))))
    if np.ptp(x) <= tolerance_x or np.ptp(y) <= tolerance_y:
        return None
    return float(spearmanr(x, y).statistic)


def binned_curve(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_x = np.log(x)
    edges = np.unique(np.quantile(log_x, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([])
    labels = np.clip(
        np.digitize(log_x, edges[1:-1], right=True),
        0,
        len(edges) - 2,
    )
    x_mean: list[float] = []
    y_mean: list[float] = []
    y_error: list[float] = []
    for label in range(len(edges) - 1):
        chosen = labels == label
        if not np.any(chosen):
            continue
        x_mean.append(float(np.exp(np.mean(log_x[chosen]))))
        y_mean.append(float(np.mean(y[chosen])))
        y_error.append(float(np.std(y[chosen]) / math.sqrt(chosen.sum())))
    return np.asarray(x_mean), np.asarray(y_mean), np.asarray(y_error)


def plot_binned(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
) -> None:
    x_mean, y_mean, y_error = binned_curve(x, y)
    axis.errorbar(
        x_mean,
        y_mean,
        yerr=y_error,
        marker="o",
        markersize=4,
        linewidth=1.7,
        capsize=2,
        label=label,
    )
    axis.set_xscale("log")


def selected(
    rows: list[dict[str, float | int | str]],
    condition: str,
    field: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row["condition"] == condition]
    )


def paired_difference(
    rows: list[dict[str, float | int | str]],
    left_condition: str,
    right_condition: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = {
        (int(row["seed"]), int(row["node"])): row
        for row in rows
        if row["condition"] == left_condition
    }
    right = {
        (int(row["seed"]), int(row["node"])): row
        for row in rows
        if row["condition"] == right_condition
    }
    if left.keys() != right.keys():
        raise AssertionError("Paired conditions contain different nodes.")
    keys = sorted(left)
    degree = np.asarray([float(left[key]["degree"]) for key in keys])
    difference = np.asarray(
        [float(left[key][field]) - float(right[key][field]) for key in keys]
    )
    return degree, difference


def curve_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    binned_x, binned_y, _ = binned_curve(x, y)
    return {
        "spearman_degree": safe_spearman(x, y),
        "lowest_degree_bin": float(binned_y[0]),
        "highest_degree_bin": float(binned_y[-1]),
        "highest_minus_lowest": float(binned_y[-1] - binned_y[0]),
        "peak": float(np.max(binned_y)),
        "highest_minus_peak": float(binned_y[-1] - np.max(binned_y)),
        "peak_degree": float(binned_x[int(np.argmax(binned_y))]),
    }


def summarize(
    rows: list[dict[str, float | int | str]],
    checks: list[dict[str, float]],
    config: Config,
) -> dict[str, object]:
    result: dict[str, object] = {
        "config": asdict(config),
        "checks": {
            "maximum_context_allocation_error": max(
                check["maximum_context_allocation_error"] for check in checks
            ),
            "equalized_updates_per_node_per_epoch": config.events_per_node,
            "equalized_updates_per_node_total": (
                config.events_per_node * config.epochs
            ),
        },
        "conditions": {},
    }
    conditions = result["conditions"]
    if not isinstance(conditions, dict):
        raise AssertionError("Conditions summary was not initialized as a mapping.")
    for condition in CONDITIONS:
        degree = selected(rows, condition.name, "degree")
        raw_norm = selected(rows, condition.name, "raw_norm")
        balanced_norm = selected(rows, condition.name, "balanced_norm")
        updates = selected(rows, condition.name, "updates_per_epoch")
        conditions[condition.name] = {
            "updates_per_epoch_min": int(np.min(updates)),
            "updates_per_epoch_max": int(np.max(updates)),
            "spearman_updates_degree": safe_spearman(degree, updates),
            "raw_norm": curve_summary(degree, raw_norm),
            "balanced_norm": curve_summary(degree, balanced_norm),
            "spearman_raw_norm_context_kl": safe_spearman(
                raw_norm,
                selected(rows, condition.name, "context_kl"),
            ),
        }

    contrasts: dict[str, object] = {}
    for subsampling in ("off", "on"):
        degree, effect = paired_difference(
            rows,
            f"proportional_{subsampling}",
            f"equalized_{subsampling}",
            "raw_norm",
        )
        contrasts[f"proportional_minus_equalized_{subsampling}"] = curve_summary(
            degree, effect
        )
    for exposure in ("proportional", "equalized"):
        degree, effect = paired_difference(
            rows,
            f"{exposure}_on",
            f"{exposure}_off",
            "raw_norm",
        )
        contrasts[f"subsampling_on_minus_off_{exposure}"] = {
            **curve_summary(degree, effect),
            "mean_absolute_effect": float(np.mean(np.abs(effect))),
        }
    result["contrasts"] = contrasts
    return result


def make_figure(
    rows: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    degree = selected(rows, "proportional_off", "degree")
    kl = selected(rows, "proportional_off", "context_kl")
    plot_binned(axes[0, 0], degree, kl, "configuration model")
    axes[0, 0].set(xlabel="degree", ylabel=r"$D_{KL}(T_i\Vert\pi)$")
    axes[0, 0].set_title("A  Population specificity")

    for condition, label in (
        ("proportional_off", "proportional"),
        ("equalized_off", "exactly equal"),
    ):
        plot_binned(
            axes[0, 1],
            selected(rows, condition, "degree"),
            selected(rows, condition, "updates_per_epoch"),
            label,
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xlabel="degree", ylabel="positive center updates / epoch")
    axes[0, 1].set_title("B  Enforced exposure")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for condition, label in (
        ("proportional_off", "proportional"),
        ("equalized_off", "exactly equal"),
    ):
        plot_binned(
            axes[0, 2],
            selected(rows, condition, "degree"),
            selected(rows, condition, "raw_norm"),
            label,
        )
    axes[0, 2].set(xlabel="degree", ylabel="raw center-vector norm")
    axes[0, 2].set_title("C  Context subsampling off")
    axes[0, 2].legend(frameon=False, fontsize=8)

    for condition, label in (
        ("proportional_on", "proportional"),
        ("equalized_on", "exactly equal"),
    ):
        plot_binned(
            axes[1, 0],
            selected(rows, condition, "degree"),
            selected(rows, condition, "raw_norm"),
            label,
        )
    axes[1, 0].set(xlabel="degree", ylabel="raw center-vector norm")
    axes[1, 0].set_title("D  Context subsampling on")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for subsampling, label in (("off", "off"), ("on", "on")):
        effect_degree, exposure_effect = paired_difference(
            rows,
            f"proportional_{subsampling}",
            f"equalized_{subsampling}",
            "raw_norm",
        )
        plot_binned(
            axes[1, 1],
            effect_degree,
            exposure_effect,
            f"context subsampling {label}",
        )
    axes[1, 1].axhline(0.0, color="0.35", linewidth=1.0)
    axes[1, 1].set(
        xlabel="degree",
        ylabel="norm(proportional) − norm(equalized)",
    )
    axes[1, 1].set_title("E  Causal exposure contrast")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for exposure in ("proportional", "equalized"):
        effect_degree, subsampling_effect = paired_difference(
            rows,
            f"{exposure}_on",
            f"{exposure}_off",
            "raw_norm",
        )
        plot_binned(
            axes[1, 2],
            effect_degree,
            subsampling_effect,
            exposure,
        )
    axes[1, 2].axhline(0.0, color="0.35", linewidth=1.0)
    axes[1, 2].set(
        xlabel="degree",
        ylabel="norm(subsampling on) − norm(off)",
    )
    axes[1, 2].set_title("F  Context-subsampling contrast")
    axes[1, 2].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle("Exact positive-center exposure intervention", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exact_center_exposure"),
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--events-per-node", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a one-seed smoke test with smaller graphs and corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        nodes=200 if args.quick else args.nodes,
        events_per_node=80 if args.quick else args.events_per_node,
        epochs=2 if args.quick else args.epochs,
        seeds=1 if args.quick else args.seeds,
        dimension=16 if args.quick else 32,
    )
    rows: list[dict[str, float | int | str]] = []
    checks: list[dict[str, float]] = []
    for seed in range(config.seeds):
        graph = configuration_graph(config.nodes, config.attachment, seed)
        adjacency, neighbors, degree = graph_arrays(graph)
        stationary = degree / degree.sum()
        kl = context_kl(adjacency, degree)

        for condition in CONDITIONS:
            centers, contexts, counts, keep, context_error = positive_pairs(
                neighbors,
                stationary,
                config,
                condition,
            )
            center_vectors, context_vectors = train_sgns(
                centers,
                contexts,
                stationary,
                config,
                seed=100_000 + seed,
            )
            raw_norm = np.linalg.norm(center_vectors, axis=1)
            balanced_norm = balanced_left_norm(center_vectors, context_vectors)
            for node in range(config.nodes):
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition.name,
                        "node": node,
                        "degree": float(degree[node]),
                        "stationary_probability": float(stationary[node]),
                        "updates_per_epoch": int(counts[node]),
                        "keep_probability": float(keep[node]),
                        "context_kl": float(kl[node]),
                        "raw_norm": float(raw_norm[node]),
                        "balanced_norm": float(balanced_norm[node]),
                    }
                )
            checks.append({"maximum_context_allocation_error": context_error})
        print(f"completed seed {seed + 1}/{config.seeds}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "node_results.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, checks, config)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure_path = args.output_dir / "exact_center_exposure.png"
    make_figure(rows, figure_path)

    print(json.dumps(summary, indent=2))
    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
