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
"""Validate the controlled turnover prediction in literal Gensim SGNS.

The graph, random-walk corpus, initialization, and training seed are paired
across three Gensim subsampling thresholds. The fixed-epoch sweep measures the
literal Word2Vec behavior; a second sweep approximately equalizes expected
retained center-token counts by changing only the epoch count. No threshold
result is used to fit the controlled exposure/specificity experiment in
simulate_turnover_surface.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulate_exact_center_exposure import gensim_keep_probability, plot_binned
from simulate_norm_frequency import (
    configuration_graph,
    graph_arrays,
    sample_walks,
    train_word2vec,
)
from simulate_turnover_surface import direct_curve_metrics

THRESHOLDS = (3e-4, 1e-3, 3e-3)
SCHEDULES = ("fixed_epochs", "budget_matched")


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
    learning_rate: float = 0.025
    min_count: int = 50


def threshold_name(threshold: float) -> str:
    labels = {3e-4: "t3e-4", 1e-3: "t1e-3", 3e-3: "t3e-3"}
    return labels[threshold]


def condition_name(schedule: str, threshold: float) -> str:
    return f"{schedule}_{threshold_name(threshold)}"


def binned_values(
    rows: list[dict[str, object]],
    condition: str,
    field: str,
    bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    local = [row for row in rows if row["condition"] == condition]
    degree = np.asarray([float(row["degree"]) for row in local])
    values = np.asarray([float(row[field]) for row in local])
    log_degree = np.log(degree)
    edges = np.unique(np.quantile(log_degree, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(
        np.digitize(log_degree, edges[1:-1], right=True),
        0,
        len(edges) - 2,
    )
    degree_mean: list[float] = []
    value_mean: list[float] = []
    for label in range(len(edges) - 1):
        chosen = labels == label
        if not np.any(chosen):
            continue
        degree_mean.append(float(np.exp(np.mean(log_degree[chosen]))))
        value_mean.append(float(np.mean(values[chosen])))
    return np.asarray(degree_mean), np.asarray(value_mean)


def controlled_peaks(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    predicted = summary["threshold_peak_shift"]["predicted_peak_degrees"]
    return [float(value) for value in predicted]


def summarize(
    rows: list[dict[str, object]],
    config: Config,
    prediction_path: Path,
) -> dict[str, object]:
    conditions: dict[str, object] = {schedule: {} for schedule in SCHEDULES}
    actual_peaks: dict[str, list[float]] = {schedule: [] for schedule in SCHEDULES}
    for schedule in SCHEDULES:
        schedule_conditions = conditions[schedule]
        if not isinstance(schedule_conditions, dict):
            raise AssertionError("Schedule conditions are not a mapping.")
        for threshold in THRESHOLDS:
            name = condition_name(schedule, threshold)
            degree, raw = binned_values(rows, name, "raw_norm")
            _, balanced = binned_values(rows, name, "balanced_norm")
            epochs = selected(rows, name, "epochs")
            total_exposure = selected(
                rows,
                name,
                "expected_total_retained_centers",
            )
            raw_curve = direct_curve_metrics(degree, raw)
            balanced_curve = direct_curve_metrics(degree, balanced)
            schedule_conditions[threshold_name(threshold)] = {
                "subsample_threshold": threshold,
                "epochs_min": int(np.min(epochs)),
                "epochs_max": int(np.max(epochs)),
                "mean_expected_total_retained_centers": float(
                    np.sum(total_exposure) / config.seeds
                ),
                "raw_norm": raw_curve,
                "balanced_norm": balanced_curve,
            }
            actual_peaks[schedule].append(float(raw_curve["peak_degree"]))
    predicted_peaks = controlled_peaks(prediction_path)
    comparison: dict[str, object] = {
        "fixed_epochs_actual_peak_degrees": actual_peaks["fixed_epochs"],
        "budget_matched_actual_peak_degrees": actual_peaks["budget_matched"],
        "controlled_prediction_available": predicted_peaks is not None,
        "controlled_prediction_status": (
            "exploratory; its interpolation rule was selected after the global "
            "quadratic surface failed"
        ),
    }
    if predicted_peaks is not None:
        comparison["controlled_predicted_peak_degrees"] = predicted_peaks
        comparison["fixed_epochs_mean_absolute_peak_degree_error"] = float(
            np.mean(
                np.abs(
                    np.asarray(actual_peaks["fixed_epochs"])
                    - np.asarray(predicted_peaks)
                )
            )
        )
        comparison["budget_matched_mean_absolute_peak_degree_error"] = float(
            np.mean(
                np.abs(
                    np.asarray(actual_peaks["budget_matched"])
                    - np.asarray(predicted_peaks)
                )
            )
        )
    return {
        "config": asdict(config),
        "paired_across_thresholds": True,
        "workers": 1,
        "window": 1,
        "negative_sampling_exponent": 1.0,
        "conditions": conditions,
        "peak_comparison": comparison,
    }


def selected(
    rows: list[dict[str, object]],
    condition: str,
    field: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row["condition"] == condition]
    )


def make_figure(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for threshold in THRESHOLDS:
        fixed_name = condition_name("fixed_epochs", threshold)
        matched_name = condition_name("budget_matched", threshold)
        label = rf"$t={threshold:.0e}$"
        plot_binned(
            axes[0, 0],
            selected(rows, fixed_name, "degree"),
            selected(rows, fixed_name, "expected_retained_centers"),
            label,
        )
        plot_binned(
            axes[0, 1],
            selected(rows, fixed_name, "degree"),
            selected(rows, fixed_name, "raw_norm"),
            label,
        )
        epochs = int(np.median(selected(rows, matched_name, "epochs")))
        plot_binned(
            axes[1, 0],
            selected(rows, matched_name, "degree"),
            selected(rows, matched_name, "raw_norm"),
            rf"$t={threshold:.0e}$ ({epochs} epochs)",
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(
        xlabel="degree",
        ylabel="expected retained center tokens",
        title="A  Literal retention exposure",
    )
    axes[0, 1].set(
        xlabel="degree",
        ylabel="raw center-vector norm",
        title="B  Fixed five epochs",
    )
    axes[1, 0].set(
        xlabel="degree",
        ylabel="raw center-vector norm",
        title="C  Retained-center budget matched",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)

    comparison = summary["peak_comparison"]
    if not isinstance(comparison, dict):
        raise AssertionError("Peak comparison is not a mapping.")
    fixed = np.asarray(comparison["fixed_epochs_actual_peak_degrees"])
    matched = np.asarray(comparison["budget_matched_actual_peak_degrees"])
    axes[1, 1].plot(THRESHOLDS, fixed, marker="o", label="fixed epochs")
    axes[1, 1].plot(THRESHOLDS, matched, marker="o", label="budget matched")
    if comparison["controlled_prediction_available"]:
        axes[1, 1].plot(
            THRESHOLDS,
            comparison["controlled_predicted_peak_degrees"],
            marker="o",
            label="exploratory controlled fit",
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(
        xlabel="subsampling threshold $t$",
        ylabel="degree at peak norm",
        title="D  External peak check",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle("Literal Word2Vec validation of the turnover prediction", fontsize=13)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gensim_threshold_sweep"),
    )
    parser.add_argument(
        "--controlled-summary",
        type=Path,
        default=Path("results/turnover_surface/summary.json"),
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--walks-per-node", type=int, default=300)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a one-seed smoke test with a smaller corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        nodes=200 if args.quick else args.nodes,
        dimension=32 if args.quick else 64,
        walks_per_node=80 if args.quick else args.walks_per_node,
        seeds=1 if args.quick else args.seeds,
        min_count=10 if args.quick else 50,
    )
    rows: list[dict[str, object]] = []
    for seed in range(config.seeds):
        graph = configuration_graph(config.nodes, config.attachment, seed)
        _, neighbors, degree = graph_arrays(graph)
        stationary = degree / degree.sum()
        walk_rng = np.random.default_rng(seed + 10_000)
        walks, counts = sample_walks(
            neighbors,
            stationary,
            config.nodes * config.walks_per_node,
            config.walk_length,
            walk_rng,
        )
        token_probability = counts / counts.sum()
        keep_by_threshold = {
            threshold: gensim_keep_probability(token_probability, threshold)
            for threshold in THRESHOLDS
        }
        retained_per_epoch = {
            threshold: float(np.sum(counts * keep_by_threshold[threshold]))
            for threshold in THRESHOLDS
        }
        target_retained_centers = config.epochs * retained_per_epoch[max(THRESHOLDS)]
        schedule_epochs = {
            "fixed_epochs": {threshold: config.epochs for threshold in THRESHOLDS},
            "budget_matched": {
                threshold: max(
                    1,
                    int(
                        np.rint(target_retained_centers / retained_per_epoch[threshold])
                    ),
                )
                for threshold in THRESHOLDS
            },
        }
        embedding_cache: dict[tuple[float, int], dict[str, np.ndarray | float]] = {}
        for schedule in SCHEDULES:
            for threshold in THRESHOLDS:
                epochs = schedule_epochs[schedule][threshold]
                cache_key = (threshold, epochs)
                if cache_key not in embedding_cache:
                    training_config = replace(config, epochs=epochs)
                    embedding_cache[cache_key] = train_word2vec(
                        walks,
                        counts,
                        training_config,
                        threshold,
                        seed + 100_000,
                    )
                embeddings = embedding_cache[cache_key]
                keep = keep_by_threshold[threshold]
                for node in range(config.nodes):
                    retained_centers = float(counts[node] * keep[node])
                    rows.append(
                        {
                            "seed": seed,
                            "schedule": schedule,
                            "condition": condition_name(schedule, threshold),
                            "subsample_threshold": threshold,
                            "epochs": epochs,
                            "node": node,
                            "degree": float(degree[node]),
                            "corpus_frequency": int(counts[node]),
                            "keep_probability": float(keep[node]),
                            "expected_retained_centers": retained_centers,
                            "expected_total_retained_centers": (
                                retained_centers * epochs
                            ),
                            "raw_norm": float(np.asarray(embeddings["raw_norm"])[node]),
                            "balanced_norm": float(
                                np.asarray(embeddings["balanced_norm"])[node]
                            ),
                        }
                    )
        print(f"completed seed {seed + 1}/{config.seeds}", flush=True)

    summary = summarize(rows, config, args.controlled_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "node_results.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure_path = args.output_dir / "gensim_threshold_sweep.png"
    make_figure(rows, summary, figure_path)
    print(json.dumps(summary, indent=2))
    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
