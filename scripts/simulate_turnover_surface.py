# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "matplotlib>=3.10,<4",
#   "networkx>=3.4,<4",
#   "numpy>=2.2,<3",
#   "scipy>=1.15,<2",
# ]
# ///
"""Causally separate exposure and context specificity in finite SGNS.

The calibration experiment crosses two interventions:

1. exact positive-center counts proportional to stationary frequency raised to
   beta; and
2. conditional contexts mixed toward their own marginal by lambda.

For realized center shares c and one-step transition matrix T, the context
marginal is q = c T and the intervention is

    T(lambda) = (1 - lambda) T + lambda 1 q.

Thus c T(lambda) = q: lambda changes row specificity while preserving the
positive-context marginal within every exposure condition. Negative contexts
are sampled from the realized positive-context marginal, so the unigram
exponent is exactly one after integer pair allocation.

Three conditions are held out from calibration. Their center counts follow the
actual Gensim/Mikolov retention law at three thresholds. A quadratic response
surface declared before the run tests whether log exposure and scalar context
KL suffice; it fails at the strongest subsampling threshold. A narrower
within-degree exposure interpolation, added after that failure, is reported as
an exploratory secondary analysis. For the full run, both analyses predict
every held-out graph seed using only calibration conditions from the other
graph seeds.

The total pair budget is fixed. The held-out conditions therefore isolate the
redistribution of exposure caused by subsampling, not its reduction in runtime.
Run the fixed defaults with

    uv run scripts/simulate_turnover_surface.py
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
import numpy as np
from simulate_exact_center_exposure import (
    balanced_left_norm,
    binned_curve,
    configuration_graph,
    exact_allocate,
    gensim_keep_probability,
    graph_arrays,
    plot_binned,
    safe_spearman,
    train_sgns,
)

CALIBRATION_BETAS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
CALIBRATION_LAMBDAS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
TARGET_THRESHOLDS = (3e-4, 1e-3, 3e-3)
REFERENCE_BETA = 0.0
REFERENCE_THRESHOLD = 3e-4


@dataclass(frozen=True)
class Config:
    nodes: int = 600
    attachment: int = 3
    dimension: int = 24
    negatives: int = 5
    events_per_node: int = 240
    epochs: int = 4
    seeds: int = 5
    batch_size: int = 256
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0025


@dataclass(frozen=True)
class Condition:
    name: str
    beta: float | None
    context_mix: float
    subsample_threshold: float | None
    held_out: bool


@dataclass(frozen=True)
class SurfaceScale:
    exposure_mean: float
    exposure_scale: float
    specificity_mean: float
    specificity_scale: float


def value_slug(value: float) -> str:
    """Return a stable three-digit slug for a value in [0, 1]."""
    return f"{round(100 * value):03d}"


def calibration_name(beta: float, context_mix: float) -> str:
    return f"beta_{value_slug(beta)}_lambda_{value_slug(context_mix)}"


def threshold_name(threshold: float) -> str:
    labels = {3e-4: "holdout_t3e-4", 1e-3: "holdout_t1e-3", 3e-3: "holdout_t3e-3"}
    return labels[threshold]


def conditions() -> tuple[Condition, ...]:
    calibration = tuple(
        Condition(
            name=calibration_name(beta, context_mix),
            beta=beta,
            context_mix=context_mix,
            subsample_threshold=None,
            held_out=False,
        )
        for beta in CALIBRATION_BETAS
        for context_mix in CALIBRATION_LAMBDAS
    )
    targets = tuple(
        Condition(
            name=threshold_name(threshold),
            beta=None,
            context_mix=0.0,
            subsample_threshold=threshold,
            held_out=True,
        )
        for threshold in TARGET_THRESHOLDS
    )
    return calibration + targets


CONDITIONS = conditions()


def transition_matrix(adjacency: np.ndarray, degree: np.ndarray) -> np.ndarray:
    return adjacency / degree[:, None]


def exposure_weights(
    stationary: np.ndarray,
    condition: Condition,
) -> np.ndarray:
    if condition.beta is not None:
        return stationary**condition.beta
    if condition.subsample_threshold is None:
        raise AssertionError("Held-out exposure law lacks a subsampling threshold.")
    return stationary * gensim_keep_probability(
        stationary,
        condition.subsample_threshold,
    )


def row_kl(conditional: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """Return row-wise KL divergence from a strictly positive marginal."""
    if np.any(marginal <= 0):
        raise ValueError("Context marginal must be strictly positive.")
    positive = conditional > 0
    log_ratio = np.zeros_like(conditional)
    log_ratio[positive] = (
        np.log(conditional[positive])
        - np.broadcast_to(np.log(marginal), conditional.shape)[positive]
    )
    return np.sum(conditional * log_ratio, axis=1)


def allocate_pairs(
    transition: np.ndarray,
    stationary: np.ndarray,
    config: Config,
    condition: Condition,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Allocate an exact pair budget for one intervention condition."""
    n = len(stationary)
    total_events = n * config.events_per_node
    center_counts = exact_allocate(
        exposure_weights(stationary, condition),
        total_events,
    )
    center_share = center_counts / total_events
    intended_context_marginal = center_share @ transition
    mixed_transition = (
        1.0 - condition.context_mix
    ) * transition + condition.context_mix * intended_context_marginal[None, :]
    intended_kl = row_kl(mixed_transition, intended_context_marginal)

    center_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    empirical_kl = np.empty(n, dtype=np.float64)
    maximum_row_error = 0.0
    context_count_matrix = np.zeros((n, n), dtype=np.int64)
    for center, count in enumerate(center_counts):
        context_counts = exact_allocate(mixed_transition[center], int(count))
        context_count_matrix[center] = context_counts
        center_parts.append(np.full(int(count), center, dtype=np.int64))
        context_parts.append(np.repeat(np.arange(n), context_counts))
        empirical = context_counts / int(count)
        maximum_row_error = max(
            maximum_row_error,
            float(np.max(np.abs(empirical - mixed_transition[center]))),
        )

    centers = np.concatenate(center_parts)
    contexts = np.concatenate(context_parts)
    realized_center_counts = np.bincount(centers, minlength=n)
    if not np.array_equal(realized_center_counts, center_counts):
        raise AssertionError("Realized center counts changed during allocation.")
    realized_context_counts = np.bincount(contexts, minlength=n)
    realized_context_marginal = realized_context_counts / total_events
    maximum_marginal_error = float(
        np.max(np.abs(realized_context_marginal - intended_context_marginal))
    )
    for center, count in enumerate(center_counts):
        empirical = context_count_matrix[center] / int(count)
        positive = empirical > 0
        empirical_kl[center] = float(
            np.sum(
                empirical[positive]
                * np.log(empirical[positive] / realized_context_marginal[positive])
            )
        )

    if len(centers) != total_events or len(contexts) != total_events:
        raise AssertionError("Pair allocation changed the total event budget.")
    if not math.isclose(float(realized_context_marginal.sum()), 1.0):
        raise AssertionError("Realized context marginal is not normalized.")
    return (
        centers,
        contexts,
        center_counts,
        realized_context_marginal,
        intended_kl,
        empirical_kl,
        maximum_row_error,
        maximum_marginal_error,
    )


def centered_score_norms(
    center_vectors: np.ndarray,
    context_vectors: np.ndarray,
    context_marginal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and balanced norms after removing the shared score intercept."""
    context_mean = context_marginal @ context_vectors
    centered_context = context_vectors - context_mean[None, :]
    raw = np.linalg.norm(center_vectors @ centered_context.T, axis=1)
    balanced = balanced_left_norm(center_vectors, centered_context)
    return raw, balanced


def selected(
    rows: list[dict[str, object]],
    condition: str,
    field: str,
    seed: int | None = None,
) -> np.ndarray:
    return np.asarray(
        [
            float(row[field])
            for row in rows
            if row["condition"] == condition
            and (seed is None or int(row["seed"]) == seed)
        ]
    )


def selected_rows(
    rows: list[dict[str, object]],
    *,
    held_out: bool | None = None,
    condition: str | None = None,
    excluded_seed: int | None = None,
    seed: int | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if (held_out is None or bool(row["held_out"]) is held_out)
        and (condition is None or row["condition"] == condition)
        and (excluded_seed is None or int(row["seed"]) != excluded_seed)
        and (seed is None or int(row["seed"]) == seed)
    ]


def response_variables(
    rows: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    exposure = np.log(np.asarray([float(row["updates_per_epoch"]) for row in rows]))
    specificity = np.asarray([float(row["context_kl"]) for row in rows])
    return exposure, specificity


def surface_matrix(
    exposure: np.ndarray,
    specificity: np.ndarray,
    scale: SurfaceScale,
) -> np.ndarray:
    x = (exposure - scale.exposure_mean) / scale.exposure_scale
    z = (specificity - scale.specificity_mean) / scale.specificity_scale
    return np.column_stack(
        (
            np.ones(len(x)),
            x,
            z,
            x**2,
            z**2,
            x * z,
        )
    )


def fit_surface(
    rows: list[dict[str, object]],
    response: str,
) -> tuple[np.ndarray, SurfaceScale]:
    exposure, specificity = response_variables(rows)
    scale = SurfaceScale(
        exposure_mean=float(np.mean(exposure)),
        exposure_scale=max(float(np.std(exposure)), np.finfo(float).eps),
        specificity_mean=float(np.mean(specificity)),
        specificity_scale=max(float(np.std(specificity)), np.finfo(float).eps),
    )
    design = surface_matrix(exposure, specificity, scale)
    outcome = np.asarray([float(row[response]) for row in rows])
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return coefficients, scale


def predict_surface(
    rows: list[dict[str, object]],
    coefficients: np.ndarray,
    scale: SurfaceScale,
) -> np.ndarray:
    exposure, specificity = response_variables(rows)
    return surface_matrix(exposure, specificity, scale) @ coefficients


def leave_one_seed_out_predictions(
    rows: list[dict[str, object]],
    config: Config,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for condition in CONDITIONS:
        if not condition.held_out:
            continue
        for seed in range(config.seeds):
            if config.seeds > 1:
                training = selected_rows(
                    rows,
                    held_out=False,
                    excluded_seed=seed,
                )
                validation_mode = "held-out condition and graph seed"
            else:
                training = selected_rows(rows, held_out=False)
                validation_mode = "held-out condition only (quick run)"
            target = selected_rows(
                rows,
                condition=condition.name,
                seed=seed,
            )
            raw_coefficients, raw_scale = fit_surface(training, "raw_norm")
            balanced_coefficients, balanced_scale = fit_surface(
                training,
                "balanced_norm",
            )
            centered_coefficients, centered_scale = fit_surface(
                training,
                "centered_score_norm",
            )
            raw_prediction = predict_surface(
                target,
                raw_coefficients,
                raw_scale,
            )
            balanced_prediction = predict_surface(
                target,
                balanced_coefficients,
                balanced_scale,
            )
            centered_prediction = predict_surface(
                target,
                centered_coefficients,
                centered_scale,
            )
            for row, predicted_raw, predicted_balanced, predicted_centered in zip(
                target,
                raw_prediction,
                balanced_prediction,
                centered_prediction,
                strict=True,
            ):
                predictions.append(
                    {
                        "seed": seed,
                        "condition": condition.name,
                        "subsample_threshold": float(
                            condition.subsample_threshold or 0.0
                        ),
                        "node": int(row["node"]),
                        "degree": float(row["degree"]),
                        "updates_per_epoch": int(row["updates_per_epoch"]),
                        "context_kl": float(row["context_kl"]),
                        "actual_raw_norm": float(row["raw_norm"]),
                        "predicted_raw_norm": float(predicted_raw),
                        "actual_balanced_norm": float(row["balanced_norm"]),
                        "predicted_balanced_norm": float(predicted_balanced),
                        "actual_centered_score_norm": float(row["centered_score_norm"]),
                        "predicted_centered_score_norm": float(predicted_centered),
                        "validation_mode": validation_mode,
                    }
                )
    return predictions


def degree_bin_edges(degree: np.ndarray, bins: int = 8) -> np.ndarray:
    """Return log-degree quantile edges with open tails."""
    edges = np.unique(np.quantile(np.log(degree), np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Degree distribution does not define at least two bins.")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def degree_bin_labels(degree: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(
        np.digitize(np.log(degree), edges[1:-1], right=True),
        0,
        len(edges) - 2,
    )


def exposure_matched_predictions(
    rows: list[dict[str, object]],
    config: Config,
) -> list[dict[str, object]]:
    """Interpolate exposure within degree bins while holding contexts fixed."""
    response_fields = ("raw_norm", "balanced_norm", "centered_score_norm")
    predictions: list[dict[str, object]] = []
    for condition in CONDITIONS:
        if not condition.held_out:
            continue
        for seed in range(config.seeds):
            excluded_seed = seed if config.seeds > 1 else None
            reference = selected_rows(
                rows,
                condition=calibration_name(0.0, 0.0),
                excluded_seed=excluded_seed,
            )
            edges = degree_bin_edges(
                np.asarray([float(row["degree"]) for row in reference])
            )
            target = selected_rows(rows, condition=condition.name, seed=seed)
            target_degree = np.asarray([float(row["degree"]) for row in target])
            target_labels = degree_bin_labels(target_degree, edges)

            calibration: dict[float, list[dict[str, object]]] = {}
            calibration_labels: dict[float, np.ndarray] = {}
            for beta in CALIBRATION_BETAS:
                beta_rows = selected_rows(
                    rows,
                    condition=calibration_name(beta, 0.0),
                    excluded_seed=excluded_seed,
                )
                calibration[beta] = beta_rows
                calibration_labels[beta] = degree_bin_labels(
                    np.asarray([float(row["degree"]) for row in beta_rows]),
                    edges,
                )

            for label in range(len(edges) - 1):
                target_chosen = target_labels == label
                if not np.any(target_chosen):
                    continue
                target_rows = [
                    row
                    for row, chosen in zip(target, target_chosen, strict=True)
                    if chosen
                ]
                target_log_updates = float(
                    np.mean(
                        np.log([float(row["updates_per_epoch"]) for row in target_rows])
                    )
                )
                record: dict[str, object] = {
                    "seed": seed,
                    "condition": condition.name,
                    "subsample_threshold": float(condition.subsample_threshold or 0.0),
                    "degree_bin": label,
                    "degree": float(
                        np.exp(
                            np.mean(
                                np.log([float(row["degree"]) for row in target_rows])
                            )
                        )
                    ),
                    "updates_per_epoch": float(np.exp(target_log_updates)),
                    "validation_mode": (
                        "held-out condition and graph seed; lambda=0 exposure "
                        "interpolation within degree bin"
                        if config.seeds > 1
                        else "held-out condition; lambda=0 exposure interpolation "
                        "within degree bin (quick run)"
                    ),
                }
                calibration_updates: list[float] = []
                calibration_outcomes: dict[str, list[float]] = {
                    field: [] for field in response_fields
                }
                for beta in CALIBRATION_BETAS:
                    beta_rows = calibration[beta]
                    chosen = calibration_labels[beta] == label
                    local = [
                        row
                        for row, include in zip(beta_rows, chosen, strict=True)
                        if include
                    ]
                    calibration_updates.append(
                        float(
                            np.mean(
                                np.log(
                                    [float(row["updates_per_epoch"]) for row in local]
                                )
                            )
                        )
                    )
                    for field in response_fields:
                        calibration_outcomes[field].append(
                            float(np.mean([float(row[field]) for row in local]))
                        )
                order = np.argsort(calibration_updates)
                ordered_updates = np.asarray(calibration_updates)[order]
                record["exposure_bracketed"] = bool(
                    ordered_updates[0] <= target_log_updates <= ordered_updates[-1]
                )
                for field in response_fields:
                    actual = float(np.mean([float(row[field]) for row in target_rows]))
                    ordered_outcome = np.asarray(calibration_outcomes[field])[order]
                    predicted = float(
                        np.interp(
                            target_log_updates,
                            ordered_updates,
                            ordered_outcome,
                        )
                    )
                    record[f"actual_{field}"] = actual
                    record[f"predicted_{field}"] = predicted
                predictions.append(record)
    return predictions


def direct_curve_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Describe an already aggregated, degree-ordered curve."""
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    peak_index = int(np.argmax(y))
    return {
        "spearman_degree": safe_spearman(x, y),
        "lowest_degree_bin": float(y[0]),
        "highest_degree_bin": float(y[-1]),
        "peak": float(y[peak_index]),
        "peak_bin": peak_index,
        "bins": len(y),
        "peak_degree": float(x[peak_index]),
        "peak_minus_lowest": float(y[peak_index] - y[0]),
        "highest_minus_peak": float(y[-1] - y[peak_index]),
        "inverted_u": bool(
            0 < peak_index < len(y) - 1
            and y[peak_index] > y[0]
            and y[peak_index] > y[-1]
        ),
    }


def exposure_prediction_metrics(
    prediction_rows: list[dict[str, object]],
    response: str,
) -> dict[str, object]:
    actual = np.asarray([float(row[f"actual_{response}"]) for row in prediction_rows])
    predicted = np.asarray(
        [float(row[f"predicted_{response}"]) for row in prediction_rows]
    )
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    aggregate: list[tuple[float, float, float]] = []
    for label in sorted({int(row["degree_bin"]) for row in prediction_rows}):
        local = [row for row in prediction_rows if int(row["degree_bin"]) == label]
        aggregate.append(
            (
                float(np.mean([float(row["degree"]) for row in local])),
                float(np.mean([float(row[f"actual_{response}"]) for row in local])),
                float(np.mean([float(row[f"predicted_{response}"]) for row in local])),
            )
        )
    degree = np.asarray([value[0] for value in aggregate])
    actual_curve = np.asarray([value[1] for value in aggregate])
    predicted_curve = np.asarray([value[2] for value in aggregate])
    return {
        "rmse_across_seed_bins": float(np.sqrt(np.mean(residual**2))),
        "mae_across_seed_bins": float(np.mean(np.abs(residual))),
        "r_squared_across_seed_bins": (1.0 - float(np.sum(residual**2)) / denominator),
        "spearman_actual_predicted_across_seed_bins": safe_spearman(
            actual,
            predicted,
        ),
        "rmse_of_pooled_curve": float(
            np.sqrt(np.mean((actual_curve - predicted_curve) ** 2))
        ),
        "fraction_exposure_bracketed": float(
            np.mean([bool(row["exposure_bracketed"]) for row in prediction_rows])
        ),
        "actual_curve": direct_curve_metrics(degree, actual_curve),
        "predicted_curve": direct_curve_metrics(degree, predicted_curve),
    }


def curve_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    binned_x, binned_y, _ = binned_curve(x, y)
    peak_index = int(np.argmax(binned_y))
    return {
        "spearman_degree": safe_spearman(x, y),
        "lowest_degree_bin": float(binned_y[0]),
        "highest_degree_bin": float(binned_y[-1]),
        "peak": float(binned_y[peak_index]),
        "peak_bin": peak_index,
        "bins": len(binned_y),
        "peak_degree": float(binned_x[peak_index]),
        "peak_minus_lowest": float(binned_y[peak_index] - binned_y[0]),
        "highest_minus_peak": float(binned_y[-1] - binned_y[peak_index]),
        "inverted_u": bool(
            0 < peak_index < len(binned_y) - 1
            and binned_y[peak_index] > binned_y[0]
            and binned_y[peak_index] > binned_y[-1]
        ),
    }


def prediction_metrics(
    prediction_rows: list[dict[str, object]],
    actual_field: str,
    predicted_field: str,
) -> dict[str, object]:
    degree = np.asarray([float(row["degree"]) for row in prediction_rows])
    actual = np.asarray([float(row[actual_field]) for row in prediction_rows])
    predicted = np.asarray([float(row[predicted_field]) for row in prediction_rows])
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r_squared": r_squared,
        "spearman_actual_predicted": safe_spearman(actual, predicted),
        "actual_curve": curve_metrics(degree, actual),
        "predicted_curve": curve_metrics(degree, predicted),
    }


def surface_derivatives(
    rows: list[dict[str, object]],
    response: str,
) -> dict[str, object]:
    coefficients, scale = fit_surface(rows, response)
    exposure, specificity = response_variables(rows)
    x = (exposure - scale.exposure_mean) / scale.exposure_scale
    z = (specificity - scale.specificity_mean) / scale.specificity_scale
    exposure_derivative = (
        coefficients[1] + 2.0 * coefficients[3] * x + coefficients[5] * z
    ) / scale.exposure_scale
    specificity_derivative = (
        coefficients[2] + 2.0 * coefficients[4] * z + coefficients[5] * x
    ) / scale.specificity_scale
    return {
        "model": (
            "norm ~ 1 + log_updates + KL + log_updates^2 + KL^2 + log_updates:KL"
        ),
        "median_partial_norm_per_log_update": float(np.median(exposure_derivative)),
        "fraction_positive_exposure_derivative": float(
            np.mean(exposure_derivative > 0)
        ),
        "median_partial_norm_per_kl": float(np.median(specificity_derivative)),
        "fraction_positive_specificity_derivative": float(
            np.mean(specificity_derivative > 0)
        ),
    }


def summarize(
    rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    exposure_predictions: list[dict[str, object]],
    checks: list[dict[str, float]],
    config: Config,
) -> dict[str, object]:
    calibration_rows = selected_rows(rows, held_out=False)
    result: dict[str, object] = {
        "analysis_plan": {
            "calibration_betas": list(CALIBRATION_BETAS),
            "calibration_context_mix": list(CALIBRATION_LAMBDAS),
            "held_out_subsample_thresholds": list(TARGET_THRESHOLDS),
            "held_out_target_excluded_from_surface_fit": True,
            "leave_one_graph_seed_out": config.seeds > 1,
            "fixed_total_pair_budget": True,
            "declared_before_run_prediction": (
                "global quadratic surface in log exposure and scalar context KL"
            ),
            "exploratory_secondary_prediction": (
                "linear interpolation across beta within degree bins at lambda=0; "
                "chosen after inspecting the global surface failure"
            ),
            "negative_distribution": (
                "realized positive-context marginal within each condition"
            ),
        },
        "config": asdict(config),
        "checks": {
            "maximum_row_allocation_error": max(
                check["maximum_row_allocation_error"] for check in checks
            ),
            "maximum_context_marginal_error": max(
                check["maximum_context_marginal_error"] for check in checks
            ),
            "events_per_condition_per_epoch": (config.nodes * config.events_per_node),
        },
        "response_surface": {
            "raw_norm": surface_derivatives(calibration_rows, "raw_norm"),
            "balanced_norm": surface_derivatives(
                calibration_rows,
                "balanced_norm",
            ),
            "centered_score_norm": surface_derivatives(
                calibration_rows,
                "centered_score_norm",
            ),
        },
        "calibration_curves": {},
        "held_out_targets": {},
    }

    calibration_curves = result["calibration_curves"]
    if not isinstance(calibration_curves, dict):
        raise AssertionError("Calibration summary is not a mapping.")
    for beta in CALIBRATION_BETAS:
        for context_mix in CALIBRATION_LAMBDAS:
            name = calibration_name(beta, context_mix)
            calibration_curves[name] = {
                "beta": beta,
                "context_mix": context_mix,
                "raw_norm": curve_metrics(
                    selected(rows, name, "degree"),
                    selected(rows, name, "raw_norm"),
                ),
                "balanced_norm": curve_metrics(
                    selected(rows, name, "degree"),
                    selected(rows, name, "balanced_norm"),
                ),
                "centered_score_norm": curve_metrics(
                    selected(rows, name, "degree"),
                    selected(rows, name, "centered_score_norm"),
                ),
            }

    target_summary = result["held_out_targets"]
    if not isinstance(target_summary, dict):
        raise AssertionError("Target summary is not a mapping.")
    actual_peak_degree: list[float] = []
    predicted_peak_degree: list[float] = []
    global_actual_peak_degree: list[float] = []
    global_predicted_peak_degree: list[float] = []
    for threshold in TARGET_THRESHOLDS:
        name = threshold_name(threshold)
        target_predictions = [row for row in predictions if row["condition"] == name]
        raw_metrics = prediction_metrics(
            target_predictions,
            "actual_raw_norm",
            "predicted_raw_norm",
        )
        balanced_metrics = prediction_metrics(
            target_predictions,
            "actual_balanced_norm",
            "predicted_balanced_norm",
        )
        centered_metrics = prediction_metrics(
            target_predictions,
            "actual_centered_score_norm",
            "predicted_centered_score_norm",
        )
        target_exposure_predictions = [
            row for row in exposure_predictions if row["condition"] == name
        ]
        exposure_raw_metrics = exposure_prediction_metrics(
            target_exposure_predictions,
            "raw_norm",
        )
        exposure_balanced_metrics = exposure_prediction_metrics(
            target_exposure_predictions,
            "balanced_norm",
        )
        exposure_score_metrics = exposure_prediction_metrics(
            target_exposure_predictions,
            "centered_score_norm",
        )
        raw = prediction_values(predictions, name, "actual_raw_norm")
        centered_score = prediction_values(
            predictions,
            name,
            "actual_centered_score_norm",
        )
        target_summary[name] = {
            "subsample_threshold": threshold,
            "updates_per_epoch_min": int(
                np.min(selected(rows, name, "updates_per_epoch"))
            ),
            "updates_per_epoch_max": int(
                np.max(selected(rows, name, "updates_per_epoch"))
            ),
            "raw_norm_prediction": raw_metrics,
            "balanced_norm_prediction": balanced_metrics,
            "centered_score_norm_prediction": centered_metrics,
            "exposure_matched_prediction": {
                "raw_norm": exposure_raw_metrics,
                "balanced_norm": exposure_balanced_metrics,
                "centered_score_norm": exposure_score_metrics,
            },
            "spearman_raw_norm_centered_score_norm": safe_spearman(
                raw,
                centered_score,
            ),
        }
        actual_curve = exposure_raw_metrics["actual_curve"]
        predicted_curve = exposure_raw_metrics["predicted_curve"]
        if not isinstance(actual_curve, dict) or not isinstance(
            predicted_curve,
            dict,
        ):
            raise AssertionError("Target curve summary is not a mapping.")
        actual_peak_degree.append(float(actual_curve["peak_degree"]))
        predicted_peak_degree.append(float(predicted_curve["peak_degree"]))
        global_actual = raw_metrics["actual_curve"]
        global_predicted = raw_metrics["predicted_curve"]
        if not isinstance(global_actual, dict) or not isinstance(
            global_predicted,
            dict,
        ):
            raise AssertionError("Global curve summary is not a mapping.")
        global_actual_peak_degree.append(float(global_actual["peak_degree"]))
        global_predicted_peak_degree.append(float(global_predicted["peak_degree"]))

    result["threshold_peak_shift"] = {
        "spearman_threshold_actual_peak_degree": safe_spearman(
            np.asarray(TARGET_THRESHOLDS),
            np.asarray(actual_peak_degree),
        ),
        "spearman_threshold_predicted_peak_degree": safe_spearman(
            np.asarray(TARGET_THRESHOLDS),
            np.asarray(predicted_peak_degree),
        ),
        "actual_peak_degrees": actual_peak_degree,
        "predicted_peak_degrees": predicted_peak_degree,
    }
    result["global_quadratic_threshold_peak_shift"] = {
        "spearman_threshold_actual_peak_degree": safe_spearman(
            np.asarray(TARGET_THRESHOLDS),
            np.asarray(global_actual_peak_degree),
        ),
        "spearman_threshold_predicted_peak_degree": safe_spearman(
            np.asarray(TARGET_THRESHOLDS),
            np.asarray(global_predicted_peak_degree),
        ),
        "actual_peak_degrees": global_actual_peak_degree,
        "predicted_peak_degrees": global_predicted_peak_degree,
    }
    return result


def prediction_values(
    predictions: list[dict[str, object]],
    condition: str,
    field: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in predictions if row["condition"] == condition]
    )


def exposure_prediction_curve(
    predictions: list[dict[str, object]],
    condition: str,
    response: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = [row for row in predictions if row["condition"] == condition]
    aggregate: list[tuple[float, float, float]] = []
    for label in sorted({int(row["degree_bin"]) for row in local}):
        bin_rows = [row for row in local if int(row["degree_bin"]) == label]
        aggregate.append(
            (
                float(np.mean([float(row["degree"]) for row in bin_rows])),
                float(np.mean([float(row[f"actual_{response}"]) for row in bin_rows])),
                float(
                    np.mean([float(row[f"predicted_{response}"]) for row in bin_rows])
                ),
            )
        )
    return (
        np.asarray([value[0] for value in aggregate]),
        np.asarray([value[1] for value in aggregate]),
        np.asarray([value[2] for value in aggregate]),
    )


def make_figure(
    rows: list[dict[str, object]],
    exposure_predictions: list[dict[str, object]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)

    for beta in CALIBRATION_BETAS:
        name = calibration_name(beta, 0.0)
        plot_binned(
            axes[0, 0],
            selected(rows, name, "degree"),
            selected(rows, name, "updates_per_epoch"),
            rf"$\beta={beta:.2g}$",
        )
    target_name = threshold_name(REFERENCE_THRESHOLD)
    plot_binned(
        axes[0, 0],
        selected(rows, target_name, "degree"),
        selected(rows, target_name, "updates_per_epoch"),
        r"held out: $t=3\times10^{-4}$",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(
        xlabel="degree",
        ylabel="positive center updates / epoch",
    )
    axes[0, 0].set_title("A  Exposure interventions")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    for context_mix in CALIBRATION_LAMBDAS:
        name = calibration_name(REFERENCE_BETA, context_mix)
        plot_binned(
            axes[0, 1],
            selected(rows, name, "degree"),
            selected(rows, name, "context_kl"),
            rf"$\lambda={context_mix:.2g}$",
        )
    axes[0, 1].set(
        xlabel="degree",
        ylabel=r"$D_{KL}(T_i^{(\lambda)}\Vert q)$",
    )
    axes[0, 1].set_title("B  Specificity interventions")
    axes[0, 1].legend(frameon=False, fontsize=7)

    for beta in CALIBRATION_BETAS:
        name = calibration_name(beta, 0.0)
        plot_binned(
            axes[0, 2],
            selected(rows, name, "degree"),
            selected(rows, name, "raw_norm"),
            rf"$\beta={beta:.2g}$",
        )
    plot_binned(
        axes[0, 2],
        selected(rows, target_name, "degree"),
        selected(rows, target_name, "raw_norm"),
        r"held out: $t=3\times10^{-4}$",
    )
    axes[0, 2].set(xlabel="degree", ylabel="raw center-vector norm")
    axes[0, 2].set_title("C  Exposure creates the rising limb")
    axes[0, 2].legend(frameon=False, fontsize=7, ncol=2)

    for context_mix in CALIBRATION_LAMBDAS:
        name = calibration_name(REFERENCE_BETA, context_mix)
        plot_binned(
            axes[1, 0],
            selected(rows, name, "degree"),
            selected(rows, name, "raw_norm"),
            rf"$\lambda={context_mix:.2g}$",
        )
    axes[1, 0].set(xlabel="degree", ylabel="raw center-vector norm")
    axes[1, 0].set_title("D  Mixing removes the falling limb")
    axes[1, 0].legend(frameon=False, fontsize=7)

    target_degree, target_actual, target_predicted = exposure_prediction_curve(
        exposure_predictions,
        target_name,
        "raw_norm",
    )
    axes[1, 1].plot(
        target_degree,
        target_actual,
        marker="o",
        label="actual held-out curve",
    )
    axes[1, 1].plot(
        target_degree,
        target_predicted,
        marker="o",
        label="exploratory exposure-matched fit",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(xlabel="degree", ylabel="raw center-vector norm")
    axes[1, 1].set_title("E  Exploratory held-out fit")
    axes[1, 1].legend(frameon=False, fontsize=7)

    actual_peaks: list[float] = []
    predicted_peaks: list[float] = []
    for threshold in TARGET_THRESHOLDS:
        name = threshold_name(threshold)
        degree, actual_curve, predicted_curve = exposure_prediction_curve(
            exposure_predictions,
            name,
            "raw_norm",
        )
        actual_peaks.append(float(degree[int(np.argmax(actual_curve))]))
        predicted_peaks.append(float(degree[int(np.argmax(predicted_curve))]))
    axes[1, 2].plot(
        TARGET_THRESHOLDS,
        actual_peaks,
        marker="o",
        label="actual",
    )
    axes[1, 2].plot(
        TARGET_THRESHOLDS,
        predicted_peaks,
        marker="o",
        label="exploratory fit",
    )
    axes[1, 2].set_xscale("log")
    axes[1, 2].set(
        xlabel="subsampling threshold $t$",
        ylabel="degree at peak norm",
    )
    axes[1, 2].set_title("F  Exploratory peak shift")
    axes[1, 2].legend(frameon=False, fontsize=7)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle("Causal two-force test for the norm-frequency turnover", fontsize=13)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_experiment(
    config: Config,
) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    rows: list[dict[str, object]] = []
    checks: list[dict[str, float]] = []
    for seed in range(config.seeds):
        graph = configuration_graph(config.nodes, config.attachment, seed)
        adjacency, _, degree = graph_arrays(graph)
        stationary = degree / degree.sum()
        transition = transition_matrix(adjacency, degree)

        for condition_index, condition in enumerate(CONDITIONS):
            (
                centers,
                contexts,
                center_counts,
                noise_distribution,
                intended_kl,
                empirical_kl,
                maximum_row_error,
                maximum_marginal_error,
            ) = allocate_pairs(
                transition,
                stationary,
                config,
                condition,
            )
            center_vectors, context_vectors = train_sgns(
                centers,
                contexts,
                noise_distribution,
                config,
                seed=100_000 + seed,
            )
            raw_norm = np.linalg.norm(center_vectors, axis=1)
            balanced_norm = balanced_left_norm(center_vectors, context_vectors)
            centered_score_norm, contrast_norm = centered_score_norms(
                center_vectors,
                context_vectors,
                noise_distribution,
            )
            for node in range(config.nodes):
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition.name,
                        "held_out": condition.held_out,
                        "beta": (
                            float(condition.beta) if condition.beta is not None else ""
                        ),
                        "context_mix": condition.context_mix,
                        "subsample_threshold": (
                            float(condition.subsample_threshold)
                            if condition.subsample_threshold is not None
                            else ""
                        ),
                        "node": node,
                        "degree": float(degree[node]),
                        "stationary_probability": float(stationary[node]),
                        "updates_per_epoch": int(center_counts[node]),
                        "context_kl": float(intended_kl[node]),
                        "empirical_context_kl": float(empirical_kl[node]),
                        "raw_norm": float(raw_norm[node]),
                        "balanced_norm": float(balanced_norm[node]),
                        "centered_score_norm": float(centered_score_norm[node]),
                        "contrast_balanced_norm": float(contrast_norm[node]),
                    }
                )
            checks.append(
                {
                    "maximum_row_allocation_error": maximum_row_error,
                    "maximum_context_marginal_error": maximum_marginal_error,
                }
            )
            print(
                f"seed {seed + 1}/{config.seeds}: "
                f"condition {condition_index + 1}/{len(CONDITIONS)}",
                flush=True,
            )
    return rows, checks


def load_node_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for row in rows:
        row["held_out"] = str(row["held_out"]).lower() == "true"
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/turnover_surface"),
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--events-per-node", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Recompute summaries and figures from an existing node_results.csv.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a one-seed smoke test with a smaller graph and pair budget.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_path = args.output_dir / "node_results.csv"
    summary_path = args.output_dir / "summary.json"
    if args.reuse:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        config = Config(**previous["config"])
        rows = load_node_rows(node_path)
        checks = [
            {
                "maximum_row_allocation_error": previous["checks"][
                    "maximum_row_allocation_error"
                ],
                "maximum_context_marginal_error": previous["checks"][
                    "maximum_context_marginal_error"
                ],
            }
        ]
    else:
        config = Config(
            nodes=180 if args.quick else args.nodes,
            dimension=12 if args.quick else 24,
            events_per_node=60 if args.quick else args.events_per_node,
            epochs=2 if args.quick else args.epochs,
            seeds=1 if args.quick else args.seeds,
        )
        rows, checks = run_experiment(config)
        with node_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    predictions = leave_one_seed_out_predictions(rows, config)
    exposure_predictions = exposure_matched_predictions(rows, config)
    summary = summarize(
        rows,
        predictions,
        exposure_predictions,
        checks,
        config,
    )

    prediction_path = args.output_dir / "heldout_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    exposure_prediction_path = args.output_dir / "exposure_matched_predictions.csv"
    with exposure_prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exposure_predictions[0]))
        writer.writeheader()
        writer.writerows(exposure_predictions)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    figure_path = args.output_dir / "turnover_surface.png"
    make_figure(rows, exposure_predictions, figure_path)

    print(json.dumps(summary, indent=2))
    print(f"wrote {node_path}")
    print(f"wrote {prediction_path}")
    print(f"wrote {exposure_prediction_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
