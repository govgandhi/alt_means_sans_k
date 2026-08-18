# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "gensim>=4.4,<5",
#   "matplotlib>=3.10,<4",
#   "networkx>=3.4,<4",
#   "numpy>=2.2,<3",
#   "scipy>=1.15,<2",
# ]
# ///
"""Test mechanisms behind the inverted-U of SGNS norm versus frequency.

The simulation separates four claims:

1. center frequency cancels from the population SGNS score at alpha = 1;
2. context information declines approximately as -log(degree) in a
   configuration model;
3. finite Gensim training can produce an inverted-U; and
4. raw input-vector norms are gauge-dependent even when all scores are fixed.

Run the preregistered defaults with

    uv run scripts/simulate_norm_frequency.py
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
from gensim.models import Word2Vec
from scipy.stats import spearmanr


@dataclass(frozen=True)
class Config:
    nodes: int = 600
    attachment: int = 3
    dimension: int = 64
    negatives: int = 5
    walk_length: int = 5
    walks_per_node: int = 300
    epochs: int = 5
    seeds: int = 5
    sample: float = 1e-3
    learning_rate: float = 0.025
    min_count: int = 50


@dataclass(frozen=True)
class Condition:
    name: str
    graph: str
    starts: str
    sample: float


CONDITIONS = (
    Condition("standard", "configuration", "size", 1e-3),
    Condition("uniform_starts", "configuration", "uniform", 1e-3),
    Condition("no_subsampling", "configuration", "size", 0.0),
    Condition("regular_control", "regular", "size", 1e-3),
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


def population_checks(
    adjacency: np.ndarray,
    degree: np.ndarray,
    negatives: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Compute graph KL and verify the two exact identities used in the paper."""
    n = len(degree)
    volume = float(degree.sum())
    stationary = degree / volume
    transition = adjacency / degree[:, None]

    edge_rows, edge_cols = np.nonzero(adjacency)
    direct_terms = np.log(transition[edge_rows, edge_cols] / stationary[edge_cols])
    direct_kl = np.bincount(
        edge_rows,
        weights=transition[edge_rows, edge_cols] * direct_terms,
        minlength=n,
    )
    formula_terms = np.log(volume / (degree[edge_rows] * degree[edge_cols]))
    formula_kl = (
        np.bincount(
            edge_rows,
            weights=formula_terms,
            minlength=n,
        )
        / degree
    )

    arbitrary_centers = rng.lognormal(size=n)
    arbitrary_centers /= arbitrary_centers.sum()
    positive = arbitrary_centers[edge_rows] * transition[edge_rows, edge_cols]
    negative = negatives * arbitrary_centers[edge_rows] * stationary[edge_cols]
    score_from_pairs = np.log(positive / negative)
    score_target = direct_terms - math.log(negatives)

    return {
        "stationary": stationary,
        "context_kl": direct_kl,
        "formula_error": float(np.max(np.abs(direct_kl - formula_kl))),
        "cancellation_error": float(np.max(np.abs(score_from_pairs - score_target))),
    }


def sample_walks(
    neighbors: list[np.ndarray],
    start_probabilities: np.ndarray,
    total_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], np.ndarray]:
    n = len(neighbors)
    starts = rng.choice(n, size=total_walks, p=start_probabilities)
    walks: list[list[int]] = []
    counts = np.zeros(n, dtype=np.int64)
    for start in starts:
        walk = [int(start)]
        for _ in range(walk_length - 1):
            walk.append(int(rng.choice(neighbors[walk[-1]])))
        counts[walk] += 1
        walks.append(walk)
    return walks, counts


def balanced_left_norm(
    center_vectors: np.ndarray,
    context_vectors: np.ndarray,
) -> np.ndarray:
    """Return the left norm of the balanced SVD gauge without forming U V^T."""
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


def gauge_norms(
    center_vectors: np.ndarray,
    context_vectors: np.ndarray,
    frequency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Change norms along a frequency-aligned direction while preserving scores."""
    standardized_frequency = np.log(frequency.astype(np.float64))
    standardized_frequency -= standardized_frequency.mean()
    frequency_weighted_gram = center_vectors.T @ (
        standardized_frequency[:, None] * center_vectors
    )
    _, eigenvectors = np.linalg.eigh(frequency_weighted_gram)
    direction = eigenvectors[:, -1]

    def transform(scale: float) -> tuple[np.ndarray, np.ndarray]:
        center_projection = center_vectors @ direction
        context_projection = context_vectors @ direction
        changed_center = center_vectors + (
            (scale - 1.0) * center_projection[:, None] * direction[None, :]
        )
        changed_context = context_vectors + (
            (1.0 / scale - 1.0) * context_projection[:, None] * direction[None, :]
        )
        return changed_center, changed_context

    shrunk_center, shrunk_context = transform(0.1)
    stretched_center, stretched_context = transform(10.0)
    original_scores = center_vectors @ context_vectors.T
    preservation_error = max(
        float(np.max(np.abs(shrunk_center @ shrunk_context.T - original_scores))),
        float(np.max(np.abs(stretched_center @ stretched_context.T - original_scores))),
    )
    return (
        np.linalg.norm(shrunk_center, axis=1),
        np.linalg.norm(stretched_center, axis=1),
        preservation_error,
    )


def train_word2vec(
    walks: list[list[int]],
    counts: np.ndarray,
    config: Config,
    sample: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    model = Word2Vec(
        sentences=walks,
        vector_size=config.dimension,
        window=1,
        min_count=config.min_count,
        sg=1,
        negative=config.negatives,
        ns_exponent=1.0,
        workers=1,
        epochs=config.epochs,
        seed=seed,
        sample=sample,
        alpha=config.learning_rate,
        sorted_vocab=1,
        shrink_windows=False,
    )
    if len(model.wv) != len(counts):
        missing = len(counts) - len(model.wv)
        raise ValueError(f"min_count removed {missing} nodes; increase walks_per_node.")
    indices = np.asarray([model.wv.key_to_index[node] for node in range(len(counts))])
    center_vectors = model.wv.vectors[indices].astype(np.float64)
    context_vectors = model.syn1neg[indices].astype(np.float64)
    raw_norm = np.linalg.norm(center_vectors, axis=1)
    balanced_norm = balanced_left_norm(center_vectors, context_vectors)
    shrunk_norm, stretched_norm, preservation_error = gauge_norms(
        center_vectors,
        context_vectors,
        counts,
    )
    return {
        "raw_norm": raw_norm,
        "balanced_norm": balanced_norm,
        "gauge_shrunk_norm": shrunk_norm,
        "gauge_stretched_norm": stretched_norm,
        "gauge_score_error": preservation_error,
    }


def regular_sizes(reference_degree: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sizes = reference_degree.copy()
    rng.shuffle(sizes)
    return sizes


def append_rows(
    rows: list[dict[str, float | int | str]],
    *,
    seed: int,
    condition: Condition,
    degree: np.ndarray,
    size: np.ndarray,
    counts: np.ndarray,
    context_kl: np.ndarray,
    embeddings: dict[str, np.ndarray | float],
) -> None:
    for node in range(len(degree)):
        rows.append(
            {
                "seed": seed,
                "condition": condition.name,
                "graph": condition.graph,
                "node": node,
                "degree": float(degree[node]),
                "size": float(size[node]),
                "frequency": int(counts[node]),
                "context_kl": float(context_kl[node]),
                "raw_norm": float(np.asarray(embeddings["raw_norm"])[node]),
                "balanced_norm": float(np.asarray(embeddings["balanced_norm"])[node]),
                "gauge_shrunk_norm": float(
                    np.asarray(embeddings["gauge_shrunk_norm"])[node]
                ),
                "gauge_stretched_norm": float(
                    np.asarray(embeddings["gauge_stretched_norm"])[node]
                ),
            }
        )


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    tolerance_x = np.finfo(float).eps * max(1.0, float(np.max(np.abs(x))))
    tolerance_y = np.finfo(float).eps * max(1.0, float(np.max(np.abs(y))))
    if np.ptp(x) <= tolerance_x or np.ptp(y) <= tolerance_y:
        return None
    return float(spearmanr(x, y).statistic)


def selected(
    rows: list[dict[str, float | int | str]],
    condition: str,
    field: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row["condition"] == condition]
    )


def binned_curve(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([])
    labels = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
    x_mean: list[float] = []
    y_mean: list[float] = []
    y_error: list[float] = []
    for label in range(len(edges) - 1):
        chosen = labels == label
        if not np.any(chosen):
            continue
        x_mean.append(float(np.mean(x[chosen])))
        y_mean.append(float(np.mean(y[chosen])))
        y_error.append(float(np.std(y[chosen]) / math.sqrt(chosen.sum())))
    return np.asarray(x_mean), np.asarray(y_mean), np.asarray(y_error)


def curve_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | bool | None]:
    log_x = np.log(x)
    z = (log_x - log_x.mean()) / log_x.std()
    standardized_y = (y - y.mean()) / y.std()
    quadratic, linear, _ = np.polyfit(z, standardized_y, 2)
    peak_z = -linear / (2.0 * quadratic) if quadratic < 0 else None
    binned_x, binned_y, _ = binned_curve(log_x, y)
    binned_z = (binned_x - log_x.mean()) / log_x.std()
    binned_target = (binned_y - binned_y.mean()) / binned_y.std()
    binned_quadratic, binned_linear, _ = np.polyfit(
        binned_z,
        binned_target,
        2,
    )
    binned_peak_z = (
        -binned_linear / (2.0 * binned_quadratic) if binned_quadratic < 0 else None
    )
    peak_bin = int(np.argmax(binned_y))
    last_drop = float(binned_y[-1] - np.max(binned_y))
    return {
        "spearman": safe_spearman(x, y),
        "quadratic": float(quadratic),
        "peak_z": float(peak_z) if peak_z is not None else None,
        "binned_quadratic": float(binned_quadratic),
        "binned_peak_z": (float(binned_peak_z) if binned_peak_z is not None else None),
        "observed_max_z": float(np.max(z)),
        "peak_bin": peak_bin,
        "bins": int(len(binned_y)),
        "last_bin_minus_peak": last_drop,
        "inverted_u_by_bins": bool(
            binned_quadratic < 0
            and binned_peak_z is not None
            and np.min(binned_z) < binned_peak_z < np.max(binned_z)
            and last_drop < -0.02 * float(np.mean(binned_y))
        ),
    }


def summarize(
    rows: list[dict[str, float | int | str]],
    checks: list[dict[str, float]],
    config: Config,
) -> dict[str, object]:
    standard_degree = selected(rows, "standard", "degree")
    standard_kl = selected(rows, "standard", "context_kl")
    population_slope = float(np.polyfit(np.log(standard_degree), standard_kl, 1)[0])
    condition_summaries: dict[str, object] = {}
    for condition in CONDITIONS:
        frequency = selected(rows, condition.name, "frequency")
        raw_norm = selected(rows, condition.name, "raw_norm")
        condition_summaries[condition.name] = {
            "norm_on_frequency": curve_metrics(frequency, raw_norm),
            "spearman_norm_context_kl": safe_spearman(
                raw_norm,
                selected(rows, condition.name, "context_kl"),
            ),
        }

    standard_frequency = selected(rows, "standard", "frequency")
    return {
        "config": asdict(config),
        "exact_checks": {
            "maximum_center_frequency_cancellation_error": max(
                check["cancellation_error"] for check in checks
            ),
            "maximum_graph_kl_formula_error": max(
                check["formula_error"] for check in checks
            ),
            "maximum_gauge_score_preservation_error": max(
                check["gauge_score_error"] for check in checks
            ),
        },
        "population": {
            "pooled_kl_slope_on_log_degree": population_slope,
            "spearman_kl_degree": safe_spearman(standard_kl, standard_degree),
        },
        "conditions": condition_summaries,
        "gauge": {
            "raw_norm_frequency_spearman": safe_spearman(
                standard_frequency,
                selected(rows, "standard", "raw_norm"),
            ),
            "shrunk_direction_norm_frequency_spearman": safe_spearman(
                standard_frequency,
                selected(rows, "standard", "gauge_shrunk_norm"),
            ),
            "stretched_direction_norm_frequency_spearman": safe_spearman(
                standard_frequency,
                selected(rows, "standard", "gauge_stretched_norm"),
            ),
            "balanced_norm_frequency_spearman": safe_spearman(
                standard_frequency,
                selected(rows, "standard", "balanced_norm"),
            ),
        },
    }


def plot_binned(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    *,
    log_x: bool = True,
) -> None:
    transformed_x = np.log(x) if log_x else x
    x_mean, y_mean, y_error = binned_curve(transformed_x, y)
    display_x = np.exp(x_mean) if log_x else x_mean
    axis.errorbar(
        display_x,
        y_mean,
        yerr=y_error,
        marker="o",
        markersize=4,
        linewidth=1.7,
        capsize=2,
        label=label,
    )
    if log_x:
        axis.set_xscale("log")


def make_figure(
    rows: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)

    degree = selected(rows, "standard", "degree")
    context_kl = selected(rows, "standard", "context_kl")
    plot_binned(axes[0, 0], degree, context_kl, "configuration model")
    regular_kl = selected(rows, "regular_control", "context_kl")
    axes[0, 0].axhline(
        np.mean(regular_kl),
        color="0.35",
        linestyle="--",
        label="regular graph",
    )
    axes[0, 0].set(xlabel="degree", ylabel=r"$D_{KL}(T_i\Vert\pi)$")
    axes[0, 0].set_title("A  Population context specificity")
    axes[0, 0].legend(frameon=False, fontsize=8)

    standard_frequency = selected(rows, "standard", "frequency")
    standard_norm = selected(rows, "standard", "raw_norm")
    plot_binned(
        axes[0, 1],
        selected(rows, "standard", "size"),
        standard_norm,
        "standard",
    )
    axes[0, 1].set(xlabel="node size (degree)", ylabel="raw input-vector norm")
    axes[0, 1].set_title("B  Standard SGNS simulation")

    for condition, label in (
        ("standard", "size-proportional starts"),
        ("uniform_starts", "uniform starts"),
    ):
        plot_binned(
            axes[0, 2],
            selected(rows, condition, "size"),
            selected(rows, condition, "raw_norm"),
            label,
        )
    axes[0, 2].set(xlabel="node size (degree)", ylabel="raw input-vector norm")
    axes[0, 2].set_title("C  Starting-exposure intervention")
    axes[0, 2].legend(frameon=False, fontsize=8)

    plot_binned(
        axes[1, 0],
        selected(rows, "standard", "size"),
        selected(rows, "standard", "raw_norm"),
        "configuration model",
    )
    regular_norm = selected(rows, "regular_control", "raw_norm")
    axes[1, 0].axhline(
        np.mean(regular_norm),
        color="tab:orange",
        linestyle="--",
        label="regular graph (mean ± SD)",
    )
    standard_sizes = selected(rows, "standard", "size")
    axes[1, 0].fill_between(
        [
            np.min(standard_sizes),
            np.max(standard_sizes),
        ],
        np.mean(regular_norm) - np.std(regular_norm),
        np.mean(regular_norm) + np.std(regular_norm),
        color="tab:orange",
        alpha=0.12,
    )
    axes[1, 0].set_xlim(np.min(standard_sizes), np.max(standard_sizes))
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel="node size (degree)", ylabel="raw input-vector norm")
    axes[1, 0].set_title("D  Structural-heterogeneity control")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for condition, label in (
        ("standard", "Gensim subsampling"),
        ("no_subsampling", "no subsampling"),
    ):
        plot_binned(
            axes[1, 1],
            selected(rows, condition, "frequency"),
            selected(rows, condition, "raw_norm"),
            label,
        )
    axes[1, 1].set(xlabel="token frequency", ylabel="raw input-vector norm")
    axes[1, 1].set_title("E  Frequent-token subsampling control")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for field, label in (
        ("gauge_shrunk_norm", "frequency direction ×0.1"),
        ("raw_norm", "raw gauge"),
        ("gauge_stretched_norm", "frequency direction ×10"),
    ):
        plot_binned(
            axes[1, 2],
            standard_frequency,
            selected(rows, "standard", field),
            label,
        )
    axes[1, 2].set(xlabel="token frequency", ylabel="input-vector norm")
    axes[1, 2].set_title("F  Same scores, different gauges")
    axes[1, 2].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle("Mechanism tests for the norm-frequency inverted-U", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/norm_frequency_simulation"),
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--walks-per-node", type=int, default=300)
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
        dimension=32 if args.quick else args.dimension,
        walks_per_node=80 if args.quick else args.walks_per_node,
        seeds=1 if args.quick else args.seeds,
        min_count=10 if args.quick else 50,
    )
    if config.nodes * (2 * config.attachment) % 2 != 0:
        raise ValueError("nodes * regular degree must be even.")

    rows: list[dict[str, float | int | str]] = []
    checks: list[dict[str, float]] = []
    for seed in range(config.seeds):
        graph_rng = np.random.default_rng(seed + 1_000)
        configuration = configuration_graph(
            config.nodes,
            config.attachment,
            seed,
        )
        regular = nx.random_regular_graph(
            2 * config.attachment,
            config.nodes,
            seed=seed,
        )
        regular = nx.convert_node_labels_to_integers(regular, ordering="sorted")
        graph_data: dict[
            str,
            tuple[list[np.ndarray], np.ndarray, dict[str, object]],
        ] = {}
        for name, graph in (("configuration", configuration), ("regular", regular)):
            adjacency, neighbors, degree = graph_arrays(graph)
            population = population_checks(
                adjacency,
                degree,
                config.negatives,
                graph_rng,
            )
            graph_data[name] = (neighbors, degree, population)

        configuration_degree = graph_data["configuration"][1]
        assigned_sizes = {
            "configuration": configuration_degree,
            "regular": regular_sizes(configuration_degree, graph_rng),
        }
        walk_cache: dict[tuple[str, str], tuple[list[list[int]], np.ndarray]] = {}

        for condition_index, condition in enumerate(CONDITIONS):
            neighbors, degree, population = graph_data[condition.graph]
            size = assigned_sizes[condition.graph]
            if condition.starts == "uniform":
                start_probabilities = np.full(config.nodes, 1.0 / config.nodes)
            else:
                start_probabilities = size / size.sum()
            cache_key = (condition.graph, condition.starts)
            if cache_key not in walk_cache:
                walk_rng = np.random.default_rng(seed + 10_000 + condition_index)
                walk_cache[cache_key] = sample_walks(
                    neighbors,
                    start_probabilities,
                    config.nodes * config.walks_per_node,
                    config.walk_length,
                    walk_rng,
                )
            walks, counts = walk_cache[cache_key]
            training_offset = {
                "standard": 0,
                "no_subsampling": 0,
                "uniform_starts": 1,
                "regular_control": 2,
            }[condition.name]
            embeddings = train_word2vec(
                walks,
                counts,
                config,
                condition.sample,
                seed + 100_000 + training_offset,
            )
            append_rows(
                rows,
                seed=seed,
                condition=condition,
                degree=degree,
                size=size,
                counts=counts,
                context_kl=np.asarray(population["context_kl"]),
                embeddings=embeddings,
            )
            checks.append(
                {
                    "cancellation_error": float(population["cancellation_error"]),
                    "formula_error": float(population["formula_error"]),
                    "gauge_score_error": float(embeddings["gauge_score_error"]),
                }
            )
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
    figure_path = args.output_dir / "mechanism_tests.png"
    make_figure(rows, figure_path)

    print(json.dumps(summary, indent=2))
    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
