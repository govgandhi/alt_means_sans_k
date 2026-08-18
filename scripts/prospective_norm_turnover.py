# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "gensim>=4.4,<5",
#   "matplotlib>=3.10,<4",
#   "networkx>=3.4,<4",
#   "numpy>=2.2,<3",
#   "scikit-learn>=1.6,<2",
#   "scipy>=1.15,<2",
# ]
# ///
"""Run the frozen prospective validation of the norm-turnover mechanism.

The workflow has three deliberately separate stages:

1. ``calibrate`` trains only on the original configuration-model family;
2. ``predict`` records and hashes target predictions without training targets;
3. ``evaluate`` verifies the hashes before training any target embedding.

Target subsampling happens outside Gensim. Gensim then receives the retained
walks with internal subsampling disabled, which makes every positive pair and
conditional context distribution observable before target training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from simulate_exact_center_exposure import (
    balanced_left_norm,
    configuration_graph,
    gensim_keep_probability,
    graph_arrays,
)
from simulate_norm_frequency import sample_walks
from sklearn.ensemble import ExtraTreesRegressor

THRESHOLDS = (3e-4, 1e-3, 3e-3)
CALIBRATION_SEEDS = tuple(range(5))
TARGET_REPLICATES = {
    "barabasi_albert": tuple(range(101, 106)),
    "powerlaw_cluster": tuple(range(201, 206)),
    "netscience": tuple(range(301, 306)),
}
RESPONSES = ("raw_norm", "balanced_norm")
MODEL_PARAMETERS = {
    "n_estimators": 400,
    "min_samples_leaf": 40,
    "max_features": 2,
    "random_state": 20_260_813,
    "n_jobs": 1,
}
SUCCESS_CRITERIA = {
    "maximum_median_absolute_peak_bin_error": 1.0,
    "minimum_fraction_peaks_within_one_bin": 2.0 / 3.0,
    "minimum_pooled_spearman": 0.5,
    "minimum_families_with_nondecreasing_observed_peaks": 2,
}


@dataclass(frozen=True)
class Config:
    nodes: int = 600
    attachment: int = 3
    triangle_probability: float = 0.35
    dimension: int = 64
    negatives: int = 5
    walk_length: int = 5
    walks_per_node: int = 300
    epochs: int = 5
    learning_rate: float = 0.025
    min_count: int = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    result = spearmanr(x, y)
    return float(result.statistic)


def largest_simple_component(graph: nx.Graph) -> nx.Graph:
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def read_netscience(path: Path) -> nx.Graph:
    return largest_simple_component(nx.read_gml(path))


def validation_graph(
    family: str,
    replicate: int,
    config: Config,
    netscience_path: Path,
) -> nx.Graph:
    if family == "barabasi_albert":
        graph = nx.barabasi_albert_graph(
            config.nodes,
            config.attachment,
            seed=replicate,
        )
    elif family == "powerlaw_cluster":
        graph = nx.powerlaw_cluster_graph(
            config.nodes,
            config.attachment,
            config.triangle_probability,
            seed=replicate,
        )
    elif family == "netscience":
        graph = read_netscience(netscience_path)
    else:
        raise ValueError(f"Unknown graph family: {family}")
    return largest_simple_component(graph)


def walk_seed(family: str, replicate: int) -> int:
    offsets = {
        "calibration": 10_000,
        "barabasi_albert": 20_000,
        "powerlaw_cluster": 30_000,
        "netscience": 40_000,
    }
    return offsets[family] + replicate


def subsample_seed(family: str, replicate: int, threshold_index: int) -> int:
    offsets = {
        "calibration": 50_000,
        "barabasi_albert": 60_000,
        "powerlaw_cluster": 70_000,
        "netscience": 80_000,
    }
    return offsets[family] + 10 * replicate + threshold_index


def optimizer_seed(family: str, replicate: int) -> int:
    offsets = {
        "calibration": 100_000,
        "barabasi_albert": 200_000,
        "powerlaw_cluster": 300_000,
        "netscience": 400_000,
    }
    return offsets[family] + replicate


def prepare_walks(
    graph: nx.Graph,
    config: Config,
    family: str,
    replicate: int,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    _, neighbors, degree = graph_arrays(graph)
    stationary = degree / degree.sum()
    rng = np.random.default_rng(walk_seed(family, replicate))
    walks, token_counts = sample_walks(
        neighbors,
        stationary,
        graph.number_of_nodes() * config.walks_per_node,
        config.walk_length,
        rng,
    )
    if np.any(token_counts == 0):
        raise AssertionError("A node never appeared in the random-walk corpus.")
    return degree, walks, token_counts


def externally_subsample(
    walks: list[list[int]],
    keep_probability: np.ndarray,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    retained: list[list[int]] = []
    for walk in walks:
        tokens = np.asarray(walk, dtype=np.int64)
        chosen = rng.random(len(tokens)) < keep_probability[tokens]
        filtered = tokens[chosen].tolist()
        if filtered:
            retained.append(filtered)
    return retained


def pair_statistics(
    walks: list[list[int]],
    nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_counts = np.zeros((nodes, nodes), dtype=np.int64)
    retained_tokens = np.zeros(nodes, dtype=np.int64)
    for walk in walks:
        retained_tokens += np.bincount(walk, minlength=nodes)
        for left, right in zip(walk[:-1], walk[1:], strict=False):
            pair_counts[left, right] += 1
            pair_counts[right, left] += 1

    positive_pairs = pair_counts.sum(axis=1)
    if np.any(positive_pairs == 0):
        missing = int(np.sum(positive_pairs == 0))
        raise ValueError(f"External subsampling left {missing} nodes without pairs.")
    total_pairs = int(positive_pairs.sum())
    context_marginal = pair_counts.sum(axis=0) / total_pairs
    if np.any(context_marginal == 0):
        raise ValueError("External subsampling produced an unseen context node.")

    conditional = pair_counts / positive_pairs[:, None]
    positive = conditional > 0
    context_kl = np.sum(
        np.where(
            positive,
            conditional
            * np.log(
                np.divide(
                    conditional,
                    context_marginal[None, :],
                    out=np.ones_like(conditional),
                    where=positive,
                )
            ),
            0.0,
        ),
        axis=1,
    )
    return positive_pairs, context_kl, retained_tokens


def prepare_condition(
    walks: list[list[int]],
    token_counts: np.ndarray,
    threshold: float,
    threshold_index: int,
    family: str,
    replicate: int,
) -> tuple[list[list[int]], np.ndarray, np.ndarray, np.ndarray]:
    token_probability = token_counts / token_counts.sum()
    keep = gensim_keep_probability(token_probability, threshold)
    retained_walks = externally_subsample(
        walks,
        keep,
        subsample_seed(family, replicate, threshold_index),
    )
    positive_pairs, context_kl, retained_tokens = pair_statistics(
        retained_walks,
        len(token_counts),
    )
    return retained_walks, positive_pairs, context_kl, retained_tokens


def train_embedding(
    walks: list[list[int]],
    nodes: int,
    config: Config,
    seed: int,
) -> dict[str, np.ndarray]:
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
        sample=0.0,
        alpha=config.learning_rate,
        sorted_vocab=1,
        shrink_windows=False,
    )
    if len(model.wv) != nodes:
        raise ValueError(f"Training retained {len(model.wv)} of {nodes} nodes.")
    indices = np.asarray([model.wv.key_to_index[node] for node in range(nodes)])
    center_vectors = model.wv.vectors[indices].astype(np.float64)
    context_vectors = model.syn1neg[indices].astype(np.float64)
    return {
        "raw_norm": np.linalg.norm(center_vectors, axis=1),
        "balanced_norm": balanced_left_norm(center_vectors, context_vectors),
    }


def append_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    family: str,
    replicate: int,
    threshold: float,
    degree: np.ndarray,
    positive_pairs: np.ndarray,
    context_kl: np.ndarray,
    retained_tokens: np.ndarray,
    outcomes: dict[str, np.ndarray] | None = None,
    predictions: dict[str, np.ndarray] | None = None,
) -> None:
    for node in range(len(degree)):
        row: dict[str, Any] = {
            "dataset": dataset,
            "family": family,
            "replicate": replicate,
            "threshold": threshold,
            "node": node,
            "nodes": len(degree),
            "degree": float(degree[node]),
            "realized_positive_pairs": int(positive_pairs[node]),
            "context_kl": float(context_kl[node]),
            "retained_tokens": int(retained_tokens[node]),
        }
        if predictions is not None:
            for response in RESPONSES:
                row[f"predicted_{response}"] = float(predictions[response][node])
        if outcomes is not None:
            for response in RESPONSES:
                row[response] = float(outcomes[response][node])
        rows.append(row)


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    pairs = np.asarray(
        [float(row["realized_positive_pairs"]) for row in rows],
        dtype=np.float64,
    )
    context_kl = np.asarray(
        [float(row["context_kl"]) for row in rows],
        dtype=np.float64,
    )
    return np.column_stack((np.log(pairs), context_kl))


def fit_models(rows: list[dict[str, Any]]) -> dict[str, ExtraTreesRegressor]:
    x = feature_matrix(rows)
    models: dict[str, ExtraTreesRegressor] = {}
    for response in RESPONSES:
        y = np.asarray([float(row[response]) for row in rows])
        model = ExtraTreesRegressor(**MODEL_PARAMETERS)
        model.fit(x, y)
        models[response] = model
    return models


def predict_models(
    models: dict[str, ExtraTreesRegressor],
    positive_pairs: np.ndarray,
    context_kl: np.ndarray,
) -> dict[str, np.ndarray]:
    x = np.column_stack((np.log(positive_pairs), context_kl))
    return {response: models[response].predict(x) for response in RESPONSES}


def calibration_cross_validation(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    predicted: dict[str, list[float]] = {response: [] for response in RESPONSES}
    actual: dict[str, list[float]] = {response: [] for response in RESPONSES}
    for seed in CALIBRATION_SEEDS:
        training = [row for row in rows if int(row["replicate"]) != seed]
        target = [row for row in rows if int(row["replicate"]) == seed]
        models = fit_models(training)
        x = feature_matrix(target)
        for response in RESPONSES:
            predicted[response].extend(models[response].predict(x).tolist())
            actual[response].extend(float(row[response]) for row in target)

    summary: dict[str, dict[str, float | None]] = {}
    for response in RESPONSES:
        observed = np.asarray(actual[response])
        estimate = np.asarray(predicted[response])
        residual = observed - estimate
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        summary[response] = {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r_squared": 1.0 - float(np.sum(residual**2)) / denominator,
            "spearman": safe_spearman(observed, estimate),
        }
    return summary


def binned_curve(
    degree: np.ndarray,
    values: np.ndarray,
    bins: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_degree = np.log(degree)
    edges = np.unique(np.quantile(log_degree, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(
        np.digitize(log_degree, edges[1:-1], right=True),
        0,
        len(edges) - 2,
    )
    x: list[float] = []
    y: list[float] = []
    error: list[float] = []
    for label in range(len(edges) - 1):
        chosen = labels == label
        if not np.any(chosen):
            continue
        x.append(float(np.exp(np.mean(log_degree[chosen]))))
        y.append(float(np.mean(values[chosen])))
        error.append(float(np.std(values[chosen]) / math.sqrt(np.sum(chosen))))
    return np.asarray(x), np.asarray(y), np.asarray(error)


def peak_metrics(degree: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    x, y, _ = binned_curve(degree, values)
    peak = int(np.argmax(y))
    return {
        "peak_bin": peak,
        "peak_degree": float(x[peak]),
        "peak_norm": float(y[peak]),
        "peak_minus_lowest": float(y[peak] - y[0]),
        "peak_minus_highest": float(y[peak] - y[-1]),
        "inverted_u": bool(
            0 < peak < len(y) - 1 and y[peak] > y[0] and y[peak] > y[-1]
        ),
    }


def grouped_curve_summary(
    rows: list[dict[str, Any]],
    predicted_field: str,
    actual_field: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in TARGET_REPLICATES:
        for threshold in THRESHOLDS:
            local = [
                row
                for row in rows
                if row["family"] == family
                and math.isclose(float(row["threshold"]), threshold)
            ]
            degree = np.asarray([float(row["degree"]) for row in local])
            predicted = np.asarray([float(row[predicted_field]) for row in local])
            record: dict[str, Any] = {
                "family": family,
                "threshold": threshold,
                "predicted": peak_metrics(degree, predicted),
            }
            if actual_field is not None:
                actual = np.asarray([float(row[actual_field]) for row in local])
                residual = actual - predicted
                record["actual"] = peak_metrics(degree, actual)
                record["absolute_peak_bin_error"] = abs(
                    int(record["actual"]["peak_bin"])
                    - int(record["predicted"]["peak_bin"])
                )
                record["node_rmse"] = float(np.sqrt(np.mean(residual**2)))
                record["node_spearman"] = safe_spearman(actual, predicted)
            output.append(record)
    return output


def calibrate(config: Config, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "calibration.jsonl"
    if calibration_path.exists() and not force:
        raise FileExistsError(f"Calibration already exists: {calibration_path}")

    rows: list[dict[str, Any]] = []
    for seed in CALIBRATION_SEEDS:
        graph = configuration_graph(config.nodes, config.attachment, seed)
        degree, walks, token_counts = prepare_walks(
            graph,
            config,
            "calibration",
            seed,
        )
        for threshold_index, threshold in enumerate(THRESHOLDS):
            retained, positive_pairs, context_kl, retained_tokens = prepare_condition(
                walks,
                token_counts,
                threshold,
                threshold_index,
                "calibration",
                seed,
            )
            outcomes = train_embedding(
                retained,
                graph.number_of_nodes(),
                config,
                optimizer_seed("calibration", seed),
            )
            append_rows(
                rows,
                dataset="calibration",
                family="configuration_model",
                replicate=seed,
                threshold=threshold,
                degree=degree,
                positive_pairs=positive_pairs,
                context_kl=context_kl,
                retained_tokens=retained_tokens,
                outcomes=outcomes,
            )
            print(
                f"calibration seed {seed}: threshold {threshold:g}",
                flush=True,
            )

    write_jsonl(calibration_path, rows)
    write_json(
        output_dir / "calibration_summary.json",
        {
            "config": asdict(config),
            "seeds": list(CALIBRATION_SEEDS),
            "thresholds": list(THRESHOLDS),
            "rows": len(rows),
            "sha256": sha256_file(calibration_path),
            "leave_one_seed_out": calibration_cross_validation(rows),
        },
    )
    print(f"wrote {calibration_path}")


def predict(
    config: Config,
    output_dir: Path,
    netscience_path: Path,
    human_protocol_path: Path,
) -> None:
    calibration_path = output_dir / "calibration.jsonl"
    protocol_path = output_dir / "protocol.json"
    predictions_path = output_dir / "frozen_predictions.jsonl"
    manifest_path = output_dir / "freeze_manifest.json"
    outcome_path = output_dir / "outcomes.jsonl"
    for path in (protocol_path, predictions_path, manifest_path, outcome_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace frozen state: {path}")
    if not calibration_path.exists():
        raise FileNotFoundError("Run the calibration stage first.")
    if not human_protocol_path.exists():
        raise FileNotFoundError(human_protocol_path)

    calibration = load_jsonl(calibration_path)
    models = fit_models(calibration)
    calibration_cv = calibration_cross_validation(calibration)
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "description": (
            "Target predictions frozen before any target embedding is trained."
        ),
        "config": asdict(config),
        "thresholds": list(THRESHOLDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "target_replicates": {
            family: list(replicates) for family, replicates in TARGET_REPLICATES.items()
        },
        "model": {
            "class": "sklearn.ensemble.ExtraTreesRegressor",
            "parameters": MODEL_PARAMETERS,
            "features": ["log(realized_positive_pairs)", "context_kl"],
            "responses": list(RESPONSES),
        },
        "success_criteria": SUCCESS_CRITERIA,
        "calibration_sha256": sha256_file(calibration_path),
        "human_protocol_sha256": sha256_file(human_protocol_path),
        "netscience_sha256": sha256_file(netscience_path),
        "calibration_leave_one_seed_out": calibration_cv,
    }
    write_json(protocol_path, protocol)
    protocol_sha = sha256_file(protocol_path)

    rows: list[dict[str, Any]] = []
    for family, replicates in TARGET_REPLICATES.items():
        for replicate in replicates:
            graph = validation_graph(family, replicate, config, netscience_path)
            degree, walks, token_counts = prepare_walks(
                graph,
                config,
                family,
                replicate,
            )
            for threshold_index, threshold in enumerate(THRESHOLDS):
                _, positive_pairs, context_kl, retained_tokens = prepare_condition(
                    walks,
                    token_counts,
                    threshold,
                    threshold_index,
                    family,
                    replicate,
                )
                predictions = predict_models(models, positive_pairs, context_kl)
                append_rows(
                    rows,
                    dataset="prospective_target",
                    family=family,
                    replicate=replicate,
                    threshold=threshold,
                    degree=degree,
                    positive_pairs=positive_pairs,
                    context_kl=context_kl,
                    retained_tokens=retained_tokens,
                    predictions=predictions,
                )
            print(f"predicted {family} replicate {replicate}", flush=True)

    write_jsonl(predictions_path, rows)
    prediction_summary = {
        "raw_norm_curves": grouped_curve_summary(
            rows,
            predicted_field="predicted_raw_norm",
        ),
        "balanced_norm_curves": grouped_curve_summary(
            rows,
            predicted_field="predicted_balanced_norm",
        ),
    }
    write_json(output_dir / "frozen_prediction_summary.json", prediction_summary)
    write_json(
        manifest_path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": protocol_sha,
            "human_protocol_sha256": sha256_file(human_protocol_path),
            "calibration_sha256": sha256_file(calibration_path),
            "predictions_sha256": sha256_file(predictions_path),
            "target_outcomes_present_at_freeze": False,
        },
    )
    print(f"froze {len(rows)} target predictions")
    print(f"protocol sha256: {protocol_sha}")
    print(f"predictions sha256: {sha256_file(predictions_path)}")


def prediction_key(row: dict[str, Any]) -> tuple[str, int, float, int]:
    return (
        str(row["family"]),
        int(row["replicate"]),
        float(row["threshold"]),
        int(row["node"]),
    )


def verify_frozen_state(output_dir: Path, human_protocol_path: Path) -> None:
    manifest = load_json(output_dir / "freeze_manifest.json")
    paths = {
        "protocol_sha256": output_dir / "protocol.json",
        "human_protocol_sha256": human_protocol_path,
        "calibration_sha256": output_dir / "calibration.jsonl",
        "predictions_sha256": output_dir / "frozen_predictions.jsonl",
    }
    for field, path in paths.items():
        observed = sha256_file(path)
        expected = str(manifest[field])
        if observed != expected:
            raise RuntimeError(f"Frozen hash mismatch for {path}: {observed}")


def evaluation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    curves = grouped_curve_summary(
        rows,
        predicted_field="predicted_raw_norm",
        actual_field="raw_norm",
    )
    peak_errors = np.asarray(
        [float(record["absolute_peak_bin_error"]) for record in curves]
    )
    actual = np.asarray([float(row["raw_norm"]) for row in rows])
    predicted = np.asarray([float(row["predicted_raw_norm"]) for row in rows])

    ordering: dict[str, dict[str, Any]] = {}
    for family in TARGET_REPLICATES:
        local = sorted(
            [record for record in curves if record["family"] == family],
            key=lambda record: float(record["threshold"]),
        )
        actual_peaks = [float(record["actual"]["peak_degree"]) for record in local]
        predicted_peaks = [
            float(record["predicted"]["peak_degree"]) for record in local
        ]
        ordering[family] = {
            "actual_peak_degrees": actual_peaks,
            "predicted_peak_degrees": predicted_peaks,
            "actual_nondecreasing": all(
                right >= left
                for left, right in zip(actual_peaks[:-1], actual_peaks[1:])
            ),
            "predicted_nondecreasing": all(
                right >= left
                for left, right in zip(predicted_peaks[:-1], predicted_peaks[1:])
            ),
        }

    families_nondecreasing = sum(
        bool(record["actual_nondecreasing"]) for record in ordering.values()
    )
    pooled_spearman = safe_spearman(actual, predicted)
    criteria = {
        "median_absolute_peak_bin_error": float(np.median(peak_errors)),
        "fraction_peaks_within_one_bin": float(np.mean(peak_errors <= 1)),
        "pooled_spearman": pooled_spearman,
        "families_with_nondecreasing_observed_peaks": families_nondecreasing,
    }
    passed = {
        "median_peak_error": (
            criteria["median_absolute_peak_bin_error"]
            <= SUCCESS_CRITERIA["maximum_median_absolute_peak_bin_error"]
        ),
        "fraction_peaks": (
            criteria["fraction_peaks_within_one_bin"]
            >= SUCCESS_CRITERIA["minimum_fraction_peaks_within_one_bin"]
        ),
        "pooled_spearman": (
            pooled_spearman is not None
            and pooled_spearman >= SUCCESS_CRITERIA["minimum_pooled_spearman"]
        ),
        "peak_order": (
            families_nondecreasing
            >= SUCCESS_CRITERIA["minimum_families_with_nondecreasing_observed_peaks"]
        ),
    }
    return {
        "primary_response": "raw_norm",
        "criteria": criteria,
        "passed": passed,
        "all_primary_criteria_passed": all(passed.values()),
        "peak_order": ordering,
        "curves": curves,
        "balanced_norm_curves": grouped_curve_summary(
            rows,
            predicted_field="predicted_balanced_norm",
            actual_field="balanced_norm",
        ),
    }


def make_figure(rows: list[dict[str, Any]], path: Path) -> None:
    colors = dict(zip(THRESHOLDS, ("#0072B2", "#E69F00", "#009E73"), strict=True))
    labels = {
        "barabasi_albert": "Direct BA",
        "powerlaw_cluster": "Clustered power law",
        "netscience": "NetScience",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    for ax, family in zip(axes.flat[:3], TARGET_REPLICATES, strict=True):
        for threshold in THRESHOLDS:
            local = [
                row
                for row in rows
                if row["family"] == family
                and math.isclose(float(row["threshold"]), threshold)
            ]
            degree = np.asarray([float(row["degree"]) for row in local])
            actual = np.asarray([float(row["raw_norm"]) for row in local])
            predicted = np.asarray([float(row["predicted_raw_norm"]) for row in local])
            actual_x, actual_y, actual_error = binned_curve(degree, actual)
            predicted_x, predicted_y, _ = binned_curve(degree, predicted)
            color = colors[threshold]
            ax.errorbar(
                actual_x,
                actual_y,
                yerr=actual_error,
                color=color,
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=rf"$t={threshold:g}$ observed",
            )
            ax.plot(
                predicted_x,
                predicted_y,
                color=color,
                linestyle="--",
                linewidth=1.3,
                label=rf"$t={threshold:g}$ frozen",
            )
        ax.set_xscale("log")
        ax.set_xlabel("degree")
        ax.set_ylabel("raw center-vector norm")
        ax.set_title(labels[family])
        ax.grid(alpha=0.25)

    curves = grouped_curve_summary(
        rows,
        predicted_field="predicted_raw_norm",
        actual_field="raw_norm",
    )
    peak_ax = axes.flat[3]
    family_markers = {
        "barabasi_albert": "o",
        "powerlaw_cluster": "s",
        "netscience": "^",
    }
    for record in curves:
        peak_ax.scatter(
            float(record["predicted"]["peak_degree"]),
            float(record["actual"]["peak_degree"]),
            color=colors[float(record["threshold"])],
            marker=family_markers[str(record["family"])],
            s=48,
        )
    limits = np.asarray(peak_ax.get_xlim() + peak_ax.get_ylim())
    lower = max(1.0, 0.9 * float(np.min(limits)))
    upper = 1.12 * float(np.max(limits))
    peak_ax.plot([lower, upper], [lower, upper], color="0.4", linestyle=":")
    peak_ax.set_xlim(lower, upper)
    peak_ax.set_ylim(lower, upper)
    peak_ax.set_xscale("log")
    peak_ax.set_yscale("log")
    peak_ax.set_xlabel("frozen predicted peak degree")
    peak_ax.set_ylabel("observed peak degree")
    peak_ax.set_title("Peak prediction")
    peak_ax.grid(alpha=0.25)
    peak_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.25",
                marker=family_markers[family],
                linestyle="none",
                label=labels[family],
            )
            for family in TARGET_REPLICATES
        ],
        loc="lower right",
        fontsize=8,
        frameon=False,
    )

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def evaluate(
    config: Config,
    output_dir: Path,
    netscience_path: Path,
    human_protocol_path: Path,
) -> None:
    outcome_path = output_dir / "outcomes.jsonl"
    summary_path = output_dir / "summary.json"
    if outcome_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to replace prospective target outcomes.")
    verify_frozen_state(output_dir, human_protocol_path)
    frozen_rows = load_jsonl(output_dir / "frozen_predictions.jsonl")
    frozen = {prediction_key(row): row for row in frozen_rows}

    rows: list[dict[str, Any]] = []
    for family, replicates in TARGET_REPLICATES.items():
        for replicate in replicates:
            graph = validation_graph(family, replicate, config, netscience_path)
            degree, walks, token_counts = prepare_walks(
                graph,
                config,
                family,
                replicate,
            )
            for threshold_index, threshold in enumerate(THRESHOLDS):
                (
                    retained,
                    positive_pairs,
                    context_kl,
                    retained_tokens,
                ) = prepare_condition(
                    walks,
                    token_counts,
                    threshold,
                    threshold_index,
                    family,
                    replicate,
                )
                predictions: dict[str, np.ndarray] = {}
                for response in RESPONSES:
                    predictions[response] = np.asarray(
                        [
                            float(
                                frozen[(family, replicate, threshold, node)][
                                    f"predicted_{response}"
                                ]
                            )
                            for node in range(graph.number_of_nodes())
                        ]
                    )
                for node in range(graph.number_of_nodes()):
                    expected = frozen[(family, replicate, threshold, node)]
                    if int(expected["realized_positive_pairs"]) != int(
                        positive_pairs[node]
                    ):
                        raise RuntimeError("Frozen pair count changed before training.")
                    if not math.isclose(
                        float(expected["context_kl"]),
                        float(context_kl[node]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError("Frozen context KL changed before training.")

                outcomes = train_embedding(
                    retained,
                    graph.number_of_nodes(),
                    config,
                    optimizer_seed(family, replicate),
                )
                append_rows(
                    rows,
                    dataset="prospective_target",
                    family=family,
                    replicate=replicate,
                    threshold=threshold,
                    degree=degree,
                    positive_pairs=positive_pairs,
                    context_kl=context_kl,
                    retained_tokens=retained_tokens,
                    outcomes=outcomes,
                    predictions=predictions,
                )
                print(
                    f"evaluated {family} replicate {replicate}: "
                    f"threshold {threshold:g}",
                    flush=True,
                )

    write_jsonl(outcome_path, rows)
    summary = evaluation_summary(rows)
    manifest = load_json(output_dir / "freeze_manifest.json")
    summary["freeze_manifest"] = manifest
    summary["outcomes_sha256"] = sha256_file(outcome_path)
    summary["config"] = asdict(config)
    write_json(summary_path, summary)
    figure_path = output_dir / "prospective_validation.png"
    make_figure(rows, figure_path)
    print(json.dumps(summary["criteria"], indent=2))
    print(json.dumps(summary["passed"], indent=2))
    print(f"all primary criteria passed: {summary['all_primary_criteria_passed']}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("calibrate", "predict", "evaluate", "plot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prospective_validation"),
    )
    parser.add_argument(
        "--netscience",
        type=Path,
        default=Path("are_angles_important/netscience/netscience.gml"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("analysis/prospective_validation_protocol.md"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace calibration only; frozen predictions and outcomes are immutable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.stage == "calibrate":
        calibrate(config, args.output_dir, args.force)
    elif args.stage == "predict":
        if args.force:
            raise ValueError("--force is available only for calibration.")
        predict(config, args.output_dir, args.netscience, args.protocol)
    elif args.stage == "evaluate":
        if args.force:
            raise ValueError("--force is available only for calibration.")
        evaluate(config, args.output_dir, args.netscience, args.protocol)
    else:
        if args.force:
            raise ValueError("--force is available only for calibration.")
        rows = load_jsonl(args.output_dir / "outcomes.jsonl")
        figure_path = args.output_dir / "prospective_validation.png"
        make_figure(rows, figure_path)
        print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
