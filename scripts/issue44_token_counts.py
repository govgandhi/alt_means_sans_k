"""Rerun the historical Gensim sweep and norm-frequency simulation with token counts.

Issue 44: the original `sample_walks` counted walks visiting a node, not its token
occurrences (`counts[walk] += 1` adds once per repeated index). PR 35 fixed the
counter. This script reruns the two unchanged experiment scripts with that fix and
the corrected Gensim build, then audits, without training, what the walk-visit
counter changed in every corpus that fed the historical sweep, the norm-frequency
simulation and the frozen prospective external retention law.

Inputs: deterministic synthetic graphs, the NetScience GML, the original summaries
(read only) and the pinned PR 35 runtime
(analysis/negative_sampling_control_requirements.txt, scripts/build_control_gensim.py).
Outputs (default results/issue44_token_counts/): plan.json, completion.json, both
reruns, count_audit.json, count_audit_nodes.csv, comparison.json and README.md.
Run: python3 scripts/issue44_token_counts.py [--output DIR]
     [--stage all|run|audit|report]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from check_control_gensim import blas_check
from prospective_norm_turnover import TARGET_REPLICATES, validation_graph, walk_seed
from prospective_norm_turnover import Config as ProspectiveConfig
from simulate_exact_center_exposure import gensim_keep_probability
from simulate_gensim_threshold_sweep import (
    SCHEDULES,
    THRESHOLDS,
    binned_values,
    condition_name,
    threshold_name,
)
from simulate_norm_frequency import (
    CONDITIONS,
    binned_curve,
    configuration_graph,
    curve_metrics,
    graph_arrays,
    population_checks,
    regular_sizes,
    sample_walks,
)
from simulate_norm_frequency import Config as NormConfig

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("results/issue44_token_counts")
REQUIREMENTS = Path("analysis/negative_sampling_control_requirements.txt")
NETSCIENCE = Path("are_angles_important/netscience/netscience.gml")
CONTROLLED_SUMMARY = Path("results/turnover_surface/summary.json")
HISTORICAL = {
    "gensim_threshold_sweep": Path("results/gensim_threshold_sweep"),
    "norm_frequency_simulation": Path("results/norm_frequency_simulation"),
}
RERUNS = {
    "gensim_threshold_sweep": "scripts/simulate_gensim_threshold_sweep.py",
    "norm_frequency_simulation": "scripts/simulate_norm_frequency.py",
}
PROTECTED = (
    *map(str, HISTORICAL.values()),
    "results/prospective_validation",
    "results/subsampling_origin",
    "results/negative_sampling_control",
    "results/negative_sampling_revalidation",
    "archive",
)
SOURCES = (
    *RERUNS.values(),
    "scripts/simulate_exact_center_exposure.py",
    "scripts/simulate_turnover_surface.py",
    "scripts/gensim_training.py",
    "scripts/check_control_gensim.py",
    "scripts/prospective_norm_turnover.py",
    str(REQUIREMENTS),
    str(CONTROLLED_SUMMARY),
    str(NETSCIENCE),
)
SINGLE_THREAD = {
    name: "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
}
BUDGET_EPOCHS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def legacy_visits(walks: list[list[int]], nodes: int) -> np.ndarray:
    """Reproduce the historical counter: a repeated index adds once per walk."""
    counts = np.zeros(nodes, dtype=np.int64)
    for walk in walks:
        counts[walk] += 1
    return counts


def display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check_output(output: Path) -> None:
    for name in PROTECTED:
        path = (ROOT / name).resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError(f"Output overlaps protected historical path: {name}")


def pinned_packages() -> dict[str, str]:
    packages = {}
    for line in (ROOT / REQUIREMENTS).read_text().splitlines():
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
        packages[name] = actual
    return packages


def run(output: Path) -> None:
    """Rerun both unchanged historical scripts into fresh subdirectories."""
    env = {**os.environ, **SINGLE_THREAD, "SOURCE_DATE_EPOCH": "0"}
    commands = {
        name: [script, "--output-dir", display(output / name)]
        for name, script in RERUNS.items()
    }
    plan = {
        "scope": "retrospective rerun of historical outputs with token counts",
        "commands": commands,
        "environment": {**SINGLE_THREAD, "SOURCE_DATE_EPOCH": "0"},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pinned_packages(),
        "blas_check": blas_check(),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
    }
    plan = json.loads(json.dumps(plan))
    plan_path, completion_path = output / "plan.json", output / "completion.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("Plan differs; use a fresh output directory.")
        if completion_path.exists():
            for name, expected in read_json(completion_path).items():
                if sha256(output / name) != expected:
                    raise ValueError(f"Changed completed artifact: {name}")
            print("Verified existing reruns; no files changed.", flush=True)
            return
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty output has no recorded plan.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    processes = {}
    for name, command in commands.items():
        log = (output / f"{name}.log").open("w")
        processes[name] = (
            subprocess.Popen(
                [sys.executable, *command],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ),
            log,
        )
    for name, (process, log) in processes.items():
        if process.wait() != 0:
            raise RuntimeError(f"{name} failed; see {log.name}")
        log.close()
    files = sorted(
        path.relative_to(output)
        for name in RERUNS
        for path in (output / name).iterdir()
        if path.suffix != ".pdf"  # PDF bytes depend on the matplotlib backend build.
    )
    write_json(completion_path, {str(path): sha256(output / path) for path in files})


def degree_labels(degree: np.ndarray, bins: int = 8) -> np.ndarray:
    """Pooled log-degree quantile bins, as in simulate_gensim_threshold_sweep."""
    log_degree = np.log(degree)
    edges = np.unique(np.quantile(log_degree, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(
        np.digitize(log_degree, edges[1:-1], right=True), 0, len(edges) - 2
    )
    return np.unique(labels, return_inverse=True)[1]


def budget_epochs(counts: np.ndarray) -> dict[str, Any]:
    """Epoch rule of simulate_gensim_threshold_sweep, from the given counts."""
    probability = counts / counts.sum()
    retained = {
        t: float(counts @ gensim_keep_probability(probability, t)) for t in THRESHOLDS
    }
    ratio = {
        t: BUDGET_EPOCHS * retained[max(THRESHOLDS)] / retained[t] for t in THRESHOLDS
    }
    return {
        threshold_name(t): {
            "retained_per_epoch": retained[t],
            "unrounded_epochs": ratio[t],
            "epochs": max(1, int(np.rint(ratio[t]))),
        }
        for t in THRESHOLDS
    }


def slope(degree: np.ndarray, counts: np.ndarray) -> float | None:
    """Log-log slope on degree; undefined on a regular graph."""
    if np.ptp(degree) == 0:
        return None
    return float(np.polyfit(np.log(degree), np.log(counts), 1)[0])


def corpus_audit(
    degree: np.ndarray, token: np.ndarray, legacy: np.ndarray
) -> dict[str, Any]:
    labels = degree_labels(degree)
    by_bin = [
        {
            "nodes": int(np.sum(labels == b)),
            "geometric_mean_degree": float(
                np.exp(np.mean(np.log(degree[labels == b])))
            ),
            "token_count": int(token[labels == b].sum()),
            "walk_visit_count": int(legacy[labels == b].sum()),
            "visit_over_token": float(
                legacy[labels == b].sum() / token[labels == b].sum()
            ),
        }
        for b in range(labels.max() + 1)
    ]
    thresholds = {}
    for t in THRESHOLDS:
        keep_token = gensim_keep_probability(token / token.sum(), t)
        keep_legacy = gensim_keep_probability(legacy / legacy.sum(), t)
        relative = keep_legacy / keep_token - 1
        thresholds[threshold_name(t)] = {
            "downsampled_nodes_token_law": int(np.sum(keep_token < 1)),
            "downsampled_nodes_visit_law": int(np.sum(keep_legacy < 1)),
            "max_relative_keep_difference": float(np.max(np.abs(relative))),
            "min_relative_keep_difference": float(np.min(relative)),
            "expected_retained_tokens_token_law": float(token @ keep_token),
            "expected_retained_tokens_visit_law": float(token @ keep_legacy),
            "visit_count_times_visit_keep": float(legacy @ keep_legacy),
        }
    ratio = legacy / token
    return {
        "nodes": len(degree),
        "token_total": int(token.sum()),
        "walk_visit_total": int(legacy.sum()),
        "visit_over_token": float(legacy.sum() / token.sum()),
        "node_ratio_min": float(ratio.min()),
        "node_ratio_max": float(ratio.max()),
        "share_ratio_min": float((ratio / (legacy.sum() / token.sum())).min()),
        "share_ratio_max": float((ratio / (legacy.sum() / token.sum())).max()),
        "log_log_slope_on_degree_token": slope(degree, token),
        "log_log_slope_on_degree_visit": slope(degree, legacy),
        "degree_bins": by_bin,
        "thresholds": thresholds,
    }


def norm_frequency_corpora(seed: int, config: NormConfig) -> dict[str, Any]:
    """Regenerate simulate_norm_frequency.main's walk corpora for one seed."""
    graph_rng = np.random.default_rng(seed + 1_000)
    configuration = configuration_graph(config.nodes, config.attachment, seed)
    regular = nx.convert_node_labels_to_integers(
        nx.random_regular_graph(2 * config.attachment, config.nodes, seed=seed),
        ordering="sorted",
    )
    graph_data = {}
    for name, graph in (("configuration", configuration), ("regular", regular)):
        adjacency, neighbors, degree = graph_arrays(graph)
        population_checks(adjacency, degree, config.negatives, graph_rng)
        graph_data[name] = (neighbors, degree)
    sizes = {
        "configuration": graph_data["configuration"][1],
        "regular": regular_sizes(graph_data["configuration"][1], graph_rng),
    }
    corpora: dict[tuple[str, str], Any] = {}
    result = {}
    for index, condition in enumerate(CONDITIONS):
        neighbors, degree = graph_data[condition.graph]
        size = sizes[condition.graph]
        key = (condition.graph, condition.starts)
        if key not in corpora:
            starts = (
                np.full(config.nodes, 1.0 / config.nodes)
                if condition.starts == "uniform"
                else size / size.sum()
            )
            corpora[key] = sample_walks(
                neighbors,
                starts,
                config.nodes * config.walks_per_node,
                config.walk_length,
                np.random.default_rng(seed + 10_000 + index),
            )
        result[condition.name] = (degree, *corpora[key])
    return result


def audit(output: Path) -> None:
    """Compare walk-visit and token counts in every affected corpus; no training."""
    norm_config = NormConfig()
    prospective = ProspectiveConfig()
    corpora: list[tuple[str, int, np.ndarray, list[list[int]], np.ndarray]] = []
    for seed in range(norm_config.seeds):
        for name, (degree, walks, token) in norm_frequency_corpora(
            seed, norm_config
        ).items():
            corpora.append((f"norm_frequency/{name}", seed, degree, walks, token))
    for family, replicates in TARGET_REPLICATES.items():
        for replicate in replicates:
            graph = validation_graph(family, replicate, prospective, ROOT / NETSCIENCE)
            _, neighbors, degree = graph_arrays(graph)
            walks, token = sample_walks(
                neighbors,
                degree / degree.sum(),
                graph.number_of_nodes() * prospective.walks_per_node,
                prospective.walk_length,
                np.random.default_rng(walk_seed(family, replicate)),
            )
            corpora.append((f"prospective/{family}", replicate, degree, walks, token))

    records, node_rows = [], []
    for name, seed, degree, walks, token in corpora:
        n = len(degree)
        assert np.array_equal(token, np.bincount(np.concatenate(walks), minlength=n))
        assert token.sum() == len(walks) * len(walks[0]) == sum(map(len, walks))
        legacy = legacy_visits(walks, n)
        record = {"corpus": name, "seed": seed, **corpus_audit(degree, token, legacy)}
        if name == "norm_frequency/standard":
            record["budget_matched_epochs"] = {
                "walk_visit_counts": budget_epochs(legacy),
                "token_counts": budget_epochs(token),
            }
        records.append(record)
        node_rows += [
            {
                "corpus": name,
                "seed": seed,
                "node": node,
                "degree": int(degree[node]),
                "token_count": int(token[node]),
                "walk_visit_count": int(legacy[node]),
            }
            for node in range(n)
        ]
        print(f"audited {name} seed {seed}", flush=True)

    frozen = read_json(ROOT / "results/negative_sampling_control/checks.json")
    calibration = {
        r["seed"]: r for r in records if r["corpus"] == "norm_frequency/standard"
    }
    frozen_check = [
        {
            "seed": row["seed"],
            "threshold": row["threshold"],
            "frozen_legacy_count_total": row["legacy_count_total"],
            "audit_walk_visit_total": calibration[row["seed"]]["walk_visit_total"],
            "frozen_realized_retained_tokens": row["retained_tokens"],
            "audit_expected_retained_tokens_visit_law": calibration[row["seed"]][
                "thresholds"
            ][threshold_name(row["threshold"])]["expected_retained_tokens_visit_law"],
            "audit_expected_retained_tokens_token_law": calibration[row["seed"]][
                "thresholds"
            ][threshold_name(row["threshold"])]["expected_retained_tokens_token_law"],
        }
        for row in frozen
    ]
    if any(
        r["frozen_legacy_count_total"] != r["audit_walk_visit_total"]
        for r in frozen_check
    ):
        raise AssertionError("Audit does not reproduce the frozen calibration counter.")
    write_json(
        output / "count_audit.json",
        {
            "note": (
                "norm_frequency/standard and no_subsampling share one corpus per seed; "
                "it is also the threshold-sweep, subsampling-origin and prospective "
                "calibration corpus (configuration graph seed s, walk seed 10000 + s)."
            ),
            "corpora": records,
            "frozen_calibration_check": frozen_check,
        },
    )
    with (output / "count_audit_nodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(node_rows[0]))
        writer.writeheader()
        writer.writerows(node_rows)


CURVE_FIELDS = (
    "bins",
    "peak_bin",
    "peak_degree",
    "lowest_degree_bin",
    "peak",
    "highest_degree_bin",
    "peak_minus_lowest",
    "highest_minus_peak",
    "inverted_u",
    "spearman_degree",
)


def nearest_bin(degrees: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.log(degrees) - np.log(value))))


def sweep_comparison(
    output: Path, audit_records: list[dict[str, Any]]
) -> dict[str, Any]:
    rerun_dir = output / "gensim_threshold_sweep"
    original = read_json(ROOT / HISTORICAL["gensim_threshold_sweep"] / "summary.json")
    corrected = read_json(rerun_dir / "summary.json")
    rows = read_csv(rerun_dir / "node_results.csv")
    seeds = sorted({int(r["seed"]) for r in rows})
    calibration = {
        r["seed"]: r for r in audit_records if r["corpus"] == "norm_frequency/standard"
    }
    nodes = {
        (r["seed"], r["node"]): r
        for r in read_csv(output / "count_audit_nodes.csv")
        if r["corpus"] == "norm_frequency/standard"
    }
    conditions = []
    for schedule in SCHEDULES:
        for t in THRESHOLDS:
            name = condition_name(schedule, t)
            local = [r for r in rows if r["condition"] == name]
            epochs = {int(r["seed"]): int(r["epochs"]) for r in local}
            visit_epochs = {
                s: (
                    BUDGET_EPOCHS
                    if schedule == "fixed_epochs"
                    else calibration[s]["budget_matched_epochs"]["walk_visit_counts"][
                        threshold_name(t)
                    ]["epochs"]
                )
                for s in seeds
            }
            if any(
                int(nodes[(r["seed"], r["node"])]["token_count"])
                != int(r["corpus_frequency"])
                for r in local
            ):
                raise AssertionError("Rerun counts differ from regenerated corpora.")
            visit_total = 0.0
            for s in seeds:
                legacy = np.array(
                    [
                        int(nodes[(str(s), str(i))]["walk_visit_count"])
                        for i in range(NormConfig().nodes)
                    ]
                )
                keep = gensim_keep_probability(legacy / legacy.sum(), t)
                visit_total += float(legacy @ keep) * visit_epochs[s]
            before = original["conditions"][schedule][threshold_name(t)]
            after = corrected["conditions"][schedule][threshold_name(t)]
            entry = {
                "schedule": schedule,
                "threshold": t,
                "original_epochs": [before["epochs_min"], before["epochs_max"]],
                "visit_count_epochs_by_seed": visit_epochs,
                "corrected_epochs_by_seed": epochs,
                "original_mean_expected_total_retained_centers": before[
                    "mean_expected_total_retained_centers"
                ],
                "audit_visit_mean_expected_total_retained_centers": visit_total
                / len(seeds),
                "corrected_mean_expected_total_retained_centers": after[
                    "mean_expected_total_retained_centers"
                ],
            }
            for response in ("raw_norm", "balanced_norm"):
                degree, values = binned_values(rows, name, response)
                entry[response] = {
                    "original": {f: before[response][f] for f in CURVE_FIELDS},
                    "corrected": {f: after[response][f] for f in CURVE_FIELDS},
                    "max_abs_summary_point_change": max(
                        abs(after[response][f] - before[response][f])
                        for f in ("lowest_degree_bin", "peak", "highest_degree_bin")
                    ),
                    "corrected_curve": {
                        "degree": degree.tolist(),
                        "mean": values.tolist(),
                    },
                }
            conditions.append(entry)

    degrees, _ = binned_values(rows, condition_name("fixed_epochs", 3e-3), "raw_norm")
    labels = degree_labels(
        np.array(
            [
                float(r["degree"])
                for r in rows
                if r["condition"] == condition_name("fixed_epochs", 3e-3)
            ]
        )
    )
    predicted = original["peak_comparison"]["controlled_predicted_peak_degrees"]
    if corrected["peak_comparison"]["controlled_predicted_peak_degrees"] != predicted:
        raise AssertionError("Controlled prediction changed between runs.")
    peaks = []
    for index, t in enumerate(THRESHOLDS):
        predicted_bin = nearest_bin(degrees, predicted[index])
        for schedule in SCHEDULES:
            before = original["conditions"][schedule][threshold_name(t)]["raw_norm"]
            after = corrected["conditions"][schedule][threshold_name(t)]["raw_norm"]
            peaks.append(
                {
                    "threshold": t,
                    "schedule": schedule,
                    "predicted_peak_degree": predicted[index],
                    "predicted_peak_bin": predicted_bin,
                    "original_peak_bin": before["peak_bin"],
                    "corrected_peak_bin": after["peak_bin"],
                    "original_bin_error": predicted_bin - before["peak_bin"],
                    "corrected_bin_error": predicted_bin - after["peak_bin"],
                    "original_peak_degree": before["peak_degree"],
                    "corrected_peak_degree": after["peak_degree"],
                }
            )

    reference = {
        (int(r["seed"]), float(r["threshold"]), int(r["node"])): r
        for r in read_csv(ROOT / "results/subsampling_origin/outcomes.csv")
    }
    reproduction = {}
    for t in THRESHOLDS:
        local = [r for r in rows if r["condition"] == condition_name("fixed_epochs", t)]
        reproduction[threshold_name(t)] = {
            response: float(
                max(
                    abs(
                        float(r[response])
                        - float(
                            reference[(int(r["seed"]), t, int(r["node"]))][response]
                        )
                    )
                    for r in local
                )
            )
            for response in ("raw_norm", "balanced_norm")
        }
    return {
        "degree_bins": {
            "mean_degree": degrees.tolist(),
            "node_rows_per_bin": np.bincount(labels).tolist(),
        },
        "conditions": conditions,
        "predicted_versus_actual_peaks": peaks,
        "original_peak_comparison": original["peak_comparison"],
        "corrected_peak_comparison": corrected["peak_comparison"],
        "max_abs_difference_from_subsampling_origin": reproduction,
    }


def norm_frequency_comparison(output: Path) -> dict[str, Any]:
    rerun_dir = output / "norm_frequency_simulation"
    original = read_json(
        ROOT / HISTORICAL["norm_frequency_simulation"] / "summary.json"
    )
    corrected = read_json(rerun_dir / "summary.json")
    rows = read_csv(rerun_dir / "node_results.csv")
    visits = {
        (r["corpus"].split("/")[1], r["seed"], r["node"]): r
        for r in read_csv(output / "count_audit_nodes.csv")
        if r["corpus"].startswith("norm_frequency/")
    }
    conditions = {}
    for condition in CONDITIONS:
        local = [r for r in rows if r["condition"] == condition.name]
        audit_rows = [visits[(condition.name, r["seed"], r["node"])] for r in local]
        if any(
            int(a["token_count"]) != int(r["frequency"])
            for a, r in zip(audit_rows, local, strict=True)
        ):
            raise AssertionError("Rerun frequencies differ from regenerated corpora.")
        token = np.array([float(r["frequency"]) for r in local])
        visit = np.array([float(a["walk_visit_count"]) for a in audit_rows])
        norm = np.array([float(r["raw_norm"]) for r in local])
        curves = {}
        for label, frequency in (("token_axis", token), ("walk_visit_axis", visit)):
            x, y, error = binned_curve(np.log(frequency), norm)
            curves[label] = {
                "mean_frequency": np.exp(x).tolist(),
                "mean_raw_norm": y.tolist(),
                "node_standard_error": error.tolist(),
            }
        conditions[condition.name] = {
            "original": original["conditions"][condition.name],
            "corrected": corrected["conditions"][condition.name],
            "corrected_with_walk_visit_axis": curve_metrics(visit, norm),
            "corrected_curves": curves,
        }
    reference = {
        (int(r["seed"]), float(r["threshold"]), int(r["node"])): r
        for r in read_csv(ROOT / "results/subsampling_origin/outcomes.csv")
    }
    reproduction = {
        name: {
            response: float(
                max(
                    abs(
                        float(r[response])
                        - float(
                            reference[(int(r["seed"]), t, int(r["node"]))][response]
                        )
                    )
                    for r in rows
                    if r["condition"] == name
                )
            )
            for response in ("raw_norm", "balanced_norm")
        }
        for name, t in (("standard", 1e-3), ("no_subsampling", 0.0))
    }
    return {
        "conditions": conditions,
        "population": {
            "original": original["population"],
            "corrected": corrected["population"],
        },
        "exact_checks": {
            "original": original["exact_checks"],
            "corrected": corrected["exact_checks"],
        },
        "gauge": {"original": original["gauge"], "corrected": corrected["gauge"]},
        "max_abs_difference_from_subsampling_origin": reproduction,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" for _ in header) + "|",
        *("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows),
        "",
    ]


def readme(audit_data: dict[str, Any], comparison: dict[str, Any]) -> str:
    records = audit_data["corpora"]
    lines = [
        "# Issue 44: token counts versus walk-visit counts",
        "",
        "Retrospective. The historical `results/gensim_threshold_sweep/` and",
        "`results/norm_frequency_simulation/` outputs are unchanged. The reruns",
        "here use the unchanged experiment scripts on `main`, which count every",
        "token occurrence (PR 35) and train with the corrected negative table and",
        "checked BLAS build.",
        "Differences from the historical numbers therefore combine all PR 35",
        "corrections. The count audit isolates what the counter alone changed.",
        "",
        "## Count audit (no training)",
        "",
        "Walk-visit totals undercount tokens in every corpus. Shares are normalized,",
        "so the retention law sees only the node-to-node variation in the undercount.",
        "",
    ]
    lines += table(
        [
            "corpus",
            "seed",
            "tokens",
            "walk visits",
            "visits/tokens",
            "node ratio min",
            "node ratio max",
            "slope token",
            "slope visit",
        ],
        [
            [
                r["corpus"],
                r["seed"],
                r["token_total"],
                r["walk_visit_total"],
                r["visit_over_token"],
                r["node_ratio_min"],
                r["node_ratio_max"],
                r["log_log_slope_on_degree_token"],
                r["log_log_slope_on_degree_visit"],
            ]
            for r in records
            if r["corpus"] != "norm_frequency/no_subsampling"
        ],
    )
    lines += [
        "Retention law at each threshold: nodes downsampled (keep < 1) under each",
        "count, the largest relative keep-probability difference, and expected",
        "retained tokens per epoch when each law is applied to the true tokens.",
        "",
    ]
    lines += table(
        [
            "corpus",
            "seed",
            "t",
            "downsampled: token / visit",
            "max abs rel keep diff",
            "retained: token law",
            "retained: visit law",
        ],
        [
            [
                r["corpus"],
                r["seed"],
                t,
                f"{v['downsampled_nodes_token_law']} / "
                f"{v['downsampled_nodes_visit_law']}",
                v["max_relative_keep_difference"],
                round(v["expected_retained_tokens_token_law"]),
                round(v["expected_retained_tokens_visit_law"]),
            ]
            for r in records
            if r["corpus"] in ("norm_frequency/standard",)
            or r["corpus"].startswith("prospective/")
            for t, v in r["thresholds"].items()
        ],
    )
    lines += [
        "`norm_frequency/standard` is also the threshold-sweep, subsampling-origin and",
        "prospective calibration corpus. The audit reproduces the frozen calibration",
        "control's walk-visit totals for all 15 calibration conditions",
        "(`frozen_calibration_check` in count_audit.json).",
        "",
        "Visit/token ratio by pooled degree bin, calibration graph seed 0:",
        "",
    ]
    seed0 = next(
        r
        for r in records
        if r["corpus"] == "norm_frequency/standard" and r["seed"] == 0
    )
    lines += table(
        [
            "bin",
            "nodes",
            "geometric mean degree",
            "tokens",
            "walk visits",
            "visits/tokens",
        ],
        [
            [
                i,
                b["nodes"],
                b["geometric_mean_degree"],
                b["token_count"],
                b["walk_visit_count"],
                b["visit_over_token"],
            ]
            for i, b in enumerate(seed0["degree_bins"])
        ],
    )

    sweep = comparison["gensim_threshold_sweep"]
    lines += [
        "## Gensim threshold sweep",
        "",
        "Training uses Gensim's own (correct) counts for subsampling and negatives,",
        "so `counts` reaches the trained norms only through the budget-matched epoch",
        "rule. Epochs and expected retained-center totals (mean over five graphs):",
        "",
    ]
    lines += table(
        [
            "schedule",
            "t",
            "original epochs",
            "visit-count epochs",
            "corrected epochs",
            "original total",
            "audit visit total",
            "corrected total",
        ],
        [
            [
                c["schedule"],
                c["threshold"],
                "–".join(map(str, sorted(set(c["original_epochs"])))),
                "–".join(
                    map(str, sorted(set(c["visit_count_epochs_by_seed"].values())))
                ),
                "–".join(map(str, sorted(set(c["corrected_epochs_by_seed"].values())))),
                round(c["original_mean_expected_total_retained_centers"]),
                round(c["audit_visit_mean_expected_total_retained_centers"]),
                round(c["corrected_mean_expected_total_retained_centers"]),
            ]
            for c in sweep["conditions"]
        ],
    )
    lines += [
        "Pooled curves, historical versus corrected (raw and balanced norms).",
        "Peak bins are zero-based indices of the occupied pooled log-degree bins.",
        "The last column is the largest change among the three saved bin means.",
        "",
    ]
    rows = []
    for c in sweep["conditions"]:
        for response in ("raw_norm", "balanced_norm"):
            o, n = c[response]["original"], c[response]["corrected"]
            rows.append(
                [
                    c["schedule"],
                    c["threshold"],
                    response,
                    f"{o['peak_bin']} / {n['peak_bin']}",
                    f"{o['peak_degree']:.3f} / {n['peak_degree']:.3f}",
                    f"{o['lowest_degree_bin']:.3f} / {n['lowest_degree_bin']:.3f}",
                    f"{o['peak']:.3f} / {n['peak']:.3f}",
                    f"{o['highest_degree_bin']:.3f} / {n['highest_degree_bin']:.3f}",
                    f"{o['peak_minus_lowest']:.3f} / {n['peak_minus_lowest']:.3f}",
                    f"{o['highest_minus_peak']:.3f} / {n['highest_minus_peak']:.3f}",
                    f"{o['inverted_u']} / {n['inverted_u']}",
                    c[response]["max_abs_summary_point_change"],
                ]
            )
    lines += table(
        [
            "schedule",
            "t",
            "response",
            "peak bin",
            "peak degree",
            "lowest bin",
            "peak",
            "highest bin",
            "peak − lowest",
            "highest − peak",
            "inverted-U",
            "max abs change",
        ],
        rows,
    )
    bin_degrees = ", ".join(f"{d:.3f}" for d in sweep["degree_bins"]["mean_degree"])
    bin_rows = ", ".join(map(str, sweep["degree_bins"]["node_rows_per_bin"]))
    lines += [
        "Every corrected pooled curve (raw norm; degree means are common to all",
        f"conditions: {bin_degrees}; node rows per bin: {bin_rows}).",
        "The historical node rows were not committed, so only the historical summary",
        "points above are available for the historical curves.",
        "",
    ]
    lines += table(
        [
            "schedule",
            "t",
            *[f"bin {i}" for i in range(len(sweep["degree_bins"]["mean_degree"]))],
        ],
        [
            [c["schedule"], c["threshold"], *c["raw_norm"]["corrected_curve"]["mean"]]
            for c in sweep["conditions"]
        ],
    )
    lines += [
        "Exploratory controlled prediction versus observed raw-norm peak bins",
        "(prediction mapped to the nearest occupied bin;",
        "error = predicted − observed):",
        "",
    ]
    lines += table(
        [
            "t",
            "schedule",
            "predicted degree",
            "predicted bin",
            "observed bin: original / corrected",
            "bin error: original / corrected",
        ],
        [
            [
                p["threshold"],
                p["schedule"],
                p["predicted_peak_degree"],
                p["predicted_peak_bin"],
                f"{p['original_peak_bin']} / {p['corrected_peak_bin']}",
                f"{p['original_bin_error']} / {p['corrected_bin_error']}",
            ]
            for p in sweep["predicted_versus_actual_peaks"]
        ],
    )
    lines += [
        "Largest absolute node-norm difference between the corrected fixed-epoch",
        "sweep and `results/subsampling_origin/outcomes.csv` (same graphs, corpora and",
        "seeds; built independently):",
        "",
    ]
    lines += table(
        ["t", "raw_norm", "balanced_norm"],
        [
            [t, v["raw_norm"], v["balanced_norm"]]
            for t, v in sweep["max_abs_difference_from_subsampling_origin"].items()
        ],
    )

    norm = comparison["norm_frequency_simulation"]
    lines += [
        "## Norm-frequency simulation",
        "",
        "Training does not use `counts`; the counter changes the frequency axis and",
        "the gauge direction. Raw norm on frequency: historical / corrected /",
        "corrected norms on the walk-visit axis. Corrected versus visit axis",
        "isolates the counter (same norms); historical versus visit axis isolates",
        "the training corrections (same axis). The gauge norms need the trained",
        "factors, which are not saved, so their changes below are not separated.",
        "",
    ]
    rows = []
    for name, c in norm["conditions"].items():
        o = c["original"]["norm_on_frequency"]
        n = c["corrected"]["norm_on_frequency"]
        v = c["corrected_with_walk_visit_axis"]
        for field in (
            "spearman",
            "peak_bin",
            "bins",
            "binned_peak_z",
            "last_bin_minus_peak",
            "inverted_u_by_bins",
        ):
            rows.append([name, field, o[field], n[field], v[field]])
        rows.append(
            [
                name,
                "spearman_norm_context_kl",
                c["original"]["spearman_norm_context_kl"],
                c["corrected"]["spearman_norm_context_kl"],
                "",
            ]
        )
    lines += table(
        ["condition", "metric", "historical", "corrected", "corrected, visit axis"],
        rows,
    )
    lines += ["Gauge and graph checks (historical / corrected):", ""]
    lines += table(
        ["quantity", "historical", "corrected"],
        [
            [f"gauge {k}", norm["gauge"]["original"][k], norm["gauge"]["corrected"][k]]
            for k in norm["gauge"]["original"]
        ]
        + [
            [
                f"population {k}",
                norm["population"]["original"][k],
                norm["population"]["corrected"][k],
            ]
            for k in norm["population"]["original"]
        ],
    )
    lines += [
        "Corrected binned raw-norm curves on each frequency axis (eight quantile",
        "bins of log frequency; standard errors are node-based, as in the",
        "historical plot routine):",
        "",
    ]
    rows = []
    for name, c in norm["conditions"].items():
        for axis, curve in c["corrected_curves"].items():
            rows.append(
                [
                    name,
                    axis,
                    ", ".join(f"{x:.0f}" for x in curve["mean_frequency"]),
                    ", ".join(f"{y:.3f}" for y in curve["mean_raw_norm"]),
                ]
            )
    lines += table(
        ["condition", "axis", "bin mean frequency", "bin mean raw norm"], rows
    )
    lines += [
        "Largest absolute node-norm difference from `results/subsampling_origin/`:",
        "",
    ]
    lines += table(
        ["condition", "raw_norm", "balanced_norm"],
        [
            [k, v["raw_norm"], v["balanced_norm"]]
            for k, v in norm["max_abs_difference_from_subsampling_origin"].items()
        ],
    )
    lines += [
        "## Reproduce",
        "",
        "Set up the pinned runtime from",
        "`analysis/negative_sampling_revalidation_protocol.md` (Python 3.13.12,",
        "`analysis/negative_sampling_control_requirements.txt`,",
        "`python3 scripts/build_control_gensim.py --install`), then run",
        "`python3 scripts/issue44_token_counts.py --output /fresh/path`. `plan.json`",
        "records the environment and source hashes; `completion.json` hashes the",
        "rerun outputs (PDFs excluded). The `run` stage verifies an existing output",
        "against both before doing anything.",
        "",
    ]
    return "\n".join(lines)


def report(output: Path) -> None:
    audit_data = read_json(output / "count_audit.json")
    comparison = {
        "gensim_threshold_sweep": sweep_comparison(output, audit_data["corpora"]),
        "norm_frequency_simulation": norm_frequency_comparison(output),
    }
    write_json(output / "comparison.json", comparison)
    (output / "README.md").write_text(readme(audit_data, comparison))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--stage", choices=("all", "run", "audit", "report"), default="all"
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    check_output(output)
    if args.stage in ("all", "run"):
        run(output)
    if args.stage in ("all", "audit"):
        audit(output)
    if args.stage in ("all", "report"):
        report(output)


if __name__ == "__main__":
    main()
