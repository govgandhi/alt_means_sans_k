"""Test whether degree–norm turnover persists without Gensim token filtering.

Inputs: deterministic synthetic graphs, original sweep settings, pinned PR 35
runtime. Outputs: node outcomes, pairing checks, graph-level curves/SEs and hashes.
Run: python3 scripts/check_subsampling_origin.py [--output /fresh/directory]
See analysis/subsampling_origin_protocol.md for the fixed design and limitations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from check_control_gensim import blas_check
from gensim.models import Word2Vec
from gensim_training import set_negative_table, train_checked
from negative_sampling_control import curve, digest_array, digest_walks, token_counts
from simulate_norm_frequency import (
    Config as TrainingConfig,
)
from simulate_norm_frequency import (
    balanced_left_norm,
    configuration_graph,
    graph_arrays,
    sample_walks,
)
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    output: Path = Path("results/subsampling_origin")
    training: TrainingConfig = field(default_factory=TrainingConfig)
    thresholds: tuple[float, ...] = (0.0, 3e-4, 1e-3, 3e-3)
    bins: int = 8
    min_alpha: float = 0.0001
    walk_seed_offset: int = 10_000
    optimizer_seed_offset: int = 100_000
    responses: tuple[str, ...] = ("raw_norm", "balanced_norm")
    protocol: str = "analysis/subsampling_origin_protocol.md"
    requirements: str = "analysis/negative_sampling_control_requirements.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def keep_probability(counts: np.ndarray, threshold: float) -> np.ndarray:
    if threshold == 0:
        return np.ones(len(counts))
    frequency = counts / counts.sum()
    return np.minimum(1, (np.sqrt(frequency / threshold) + 1) * threshold / frequency)


def make_model(
    walks: list[list[int]], config: Config, threshold: float, seed: int
) -> Word2Vec:
    training = config.training
    model = Word2Vec(
        vector_size=training.dimension,
        window=1,
        min_count=training.min_count,
        sg=1,
        negative=training.negatives,
        ns_exponent=1.0,
        workers=1,
        epochs=training.epochs,
        seed=seed + config.optimizer_seed_offset,
        sample=threshold,
        alpha=training.learning_rate,
        min_alpha=config.min_alpha,
        sorted_vocab=1,
        shrink_windows=False,
    )
    model.build_vocab(walks)
    if set(model.wv.index_to_key) != set(range(training.nodes)):
        raise ValueError("Every graph node must survive the vocabulary cutoff.")
    set_negative_table(model)
    return model


def signature(model: Word2Vec) -> dict[str, Any]:
    state = model.random.get_state()
    return {
        "vocabulary": list(model.wv.index_to_key),
        "initial_input_sha256": digest_array(model.wv.vectors),
        "initial_output_sha256": digest_array(model.syn1neg),
        "slot_counts_sha256": digest_array(model.wv.expandos["count"]),
        "negative_cdf_sha256": digest_array(model.cum_table),
        "rng_array_sha256": digest_array(state[1]),
        "rng_metadata": [state[0], *state[2:]],
    }


def summarize(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    per_graph, pooled, across_graphs = [], [], []
    for threshold in config.thresholds:
        local = [r for r in rows if r["threshold"] == threshold]
        degree = np.array([r["degree"] for r in local])
        seeds = np.array([r["seed"] for r in local])
        log_degree = np.log(degree)
        edges = np.unique(np.quantile(log_degree, np.linspace(0, 1, config.bins + 1)))
        labels = np.digitize(log_degree, edges[1:-1], right=True)
        for response in config.responses:
            values = np.array([r[response] for r in local])
            for seed in range(config.training.seeds):
                selected = seeds == seed
                per_graph.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "response": response,
                        **curve(degree[selected], values[selected], config.bins),
                    }
                )
            pooled.append(
                {
                    "threshold": threshold,
                    "response": response,
                    **curve(degree, values, config.bins),
                }
            )
            common_bins = []
            for label in np.unique(labels):
                selected = labels == label
                counts, means = [], []
                for seed in range(config.training.seeds):
                    chosen = selected & (seeds == seed)
                    counts.append(int(chosen.sum()))
                    means.append(float(values[chosen].mean()) if chosen.any() else None)
                observed = [m for m in means if m is not None]
                common_bins.append(
                    {
                        "degree": float(np.exp(log_degree[selected].mean())),
                        "node_counts_by_seed": counts,
                        "mean_norm_by_seed": means,
                        "graphs": len(observed),
                        "mean_norm": float(np.mean(observed)),
                        "standard_error": (
                            float(np.std(observed, ddof=1) / np.sqrt(len(observed)))
                            if len(observed) > 1
                            else None
                        ),
                    }
                )
            across_graphs.append(
                {
                    "threshold": threshold,
                    "response": response,
                    "log_degree_edges": edges.tolist(),
                    "bins": common_bins,
                }
            )
    return {"per_graph": per_graph, "pooled": pooled, "across_graphs": across_graphs}


def plot(summary: dict[str, Any], output: Path, config: Config) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for i, response in enumerate(config.responses):
        for j, threshold in enumerate(config.thresholds):
            ax = axes[i, j]
            for result in summary["per_graph"]:
                if result["threshold"] == threshold and result["response"] == response:
                    ax.plot(result["degree"], result["mean_norm"], alpha=0.5, lw=1)
            aggregate = next(
                r
                for r in summary["across_graphs"]
                if r["threshold"] == threshold and r["response"] == response
            )
            bins = aggregate["bins"]
            ax.errorbar(
                [b["degree"] for b in bins],
                [b["mean_norm"] for b in bins],
                yerr=[b["standard_error"] or 0 for b in bins],
                color="black",
                marker="o",
                markersize=3,
                lw=1.8,
                capsize=3,
                label="Equal graph mean ± SE",
            )
            ax.set_xscale("log")
            ax.set_title(
                "Filtering off" if threshold == 0 else f"sample = {threshold:g}"
            )
            ax.set_xlabel("Degree (geometric bin mean)")
            ax.set_ylabel(response.replace("_", " "))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Five graphs, five epochs each; colored lines are individual graphs")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=180)
    plt.close(fig)


def report(summary: dict[str, Any], checks: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Subsampling-origin diagnostic (issue 21)",
        "",
        "Corrected Gensim, five configuration graphs, five epochs each. All four",
        "settings share each graph's raw walks, initialization and negative CDF.",
        "This is a retrospective, fixed-epoch test; update budgets are not matched.",
        "",
        "## Primary outcome: raw input norm",
        "",
        "An interior peak is a descriptive maximum of binned means, not a test of",
        "unimodality. Peak gaps are selected from the observed curve.",
        "",
        "| sample | graph seed | occupied bins | peak bin | peak degree | "
        "peak − low | peak − high | interior |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in summary["per_graph"]:
        if result["response"] == "raw_norm":
            peak = result["peak_bin"]
            lines.append(
                f"| {result['threshold']:g} | {result['seed']} | "
                f"{len(result['degree'])} | {peak} | "
                f"{result['degree'][peak - 1]:.4f} | "
                f"{result['peak_minus_lowest']:.6f} | "
                f"{result['peak_minus_highest']:.6f} | {result['interior_peak']} |"
            )
    lines += [
        "",
        "## Pooled curves (comparison with the original sweep)",
        "",
        "| sample | response | peak degree | peak − high | interior |",
        "|---:|:---|---:|---:|:---:|",
    ]
    for result in summary["pooled"]:
        lines.append(
            f"| {result['threshold']:g} | {result['response']} | "
            f"{result['degree'][result['peak_bin'] - 1]:.4f} | "
            f"{result['peak_minus_highest']:.6f} | {result['interior_peak']} |"
        )
    lines += [
        "",
        "## Training accounting",
        "",
        "Effective tokens are Gensim's retained-token accounting, not positive",
        "pairs or gradient updates. Raw tokens include repeated epoch passes.",
        "",
        "| sample | seed | raw tokens | effective tokens | downsampled nodes |",
        "|---:|---:|---:|---:|---:|",
    ]
    for check in checks:
        lines.append(
            f"| {check['threshold']:g} | {check['seed']} | {check['raw_tokens']} | "
            f"{check['effective_tokens']} | {check['downsampled_nodes']} |"
        )
    lines += [
        "",
        "## Evidence and limitations",
        "",
        "`outcomes.csv` contains every node and both norm readouts. `summary.json`",
        "contains all individual and pooled curves, bin counts, and graph means/SEs.",
        "Standard errors average graph means equally within common degree bins;",
        "they do not treat individual nodes as independent replicates. Colored",
        "curves use each graph's quantiles; black curves use common boundaries.",
        "`checks.json`, `plan.json` and `completion.json` record the pairing, runtime,",
        "source and output hashes. The plan precedes training.",
        "",
        "Filtering also changes context formation and the training budget. This",
        "test can establish that filtering is unnecessary for turnover in these",
        "settings if unfiltered turnover persists. It cannot establish the cause,",
        "convergence, generality, or an exposure-only effect. No manuscript claims",
        "or frozen historical outputs were modified.",
        "",
        "Reproduce with "
        "`python3 scripts/check_subsampling_origin.py --output /fresh/path`",
        "after following `analysis/subsampling_origin_protocol.md`.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def run(config: Config) -> None:
    output = (config.root / config.output).resolve()
    protected = [
        "results/norm_frequency_simulation",
        "results/gensim_threshold_sweep",
        "results/prospective_validation",
        "results/negative_sampling_control",
        "results/negative_sampling_revalidation",
        "archive",
    ]
    for name in protected:
        path = (config.root / name).resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError(f"Output overlaps protected historical path: {name}")
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Nonempty output has no matching recorded plan.")
    paths = [
        *sorted((config.root / "scripts").glob("*.py")),
        config.root / config.protocol,
        config.root / config.requirements,
    ]
    packages = {}
    for line in (config.root / config.requirements).read_text().splitlines():
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
        packages[name] = actual
    plan = {
        "scope": "retrospective no-filter control; fixed epochs, not matched updates",
        "config": {
            k: v for k, v in asdict(config).items() if k not in {"root", "output"}
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "blas_check": blas_check(),
        "source_sha256": {str(p.relative_to(config.root)): sha256(p) for p in paths},
    }
    plan = json.loads(json.dumps(plan))
    plan_path, completion = output / "plan.json", output / "completion.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Plan differs; use a fresh output directory.")
        if completion.exists():
            for name, expected in json.loads(completion.read_text()).items():
                if sha256(output / name) != expected:
                    raise ValueError(f"Changed completed artifact: {name}")
            print("Verified existing complete results; no files changed.", flush=True)
            return
    output.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    rows, checks = [], []
    training = config.training
    for seed in range(training.seeds):
        graph = configuration_graph(training.nodes, training.attachment, seed)
        adjacency, neighbors, degree = graph_arrays(graph)
        walks, counts = sample_walks(
            neighbors,
            degree / degree.sum(),
            training.nodes * training.walks_per_node,
            training.walk_length,
            np.random.default_rng(seed + config.walk_seed_offset),
        )
        np.testing.assert_array_equal(counts, token_counts(walks, training.nodes))
        common = {
            "graph_sha256": digest_array(adjacency),
            "walks_sha256": digest_walks(walks),
            "counts_sha256": digest_array(counts),
        }
        reference = None
        for threshold in config.thresholds:
            model = make_model(walks, config, threshold, seed)
            initial = signature(model)
            if reference is None:
                reference = initial
            if initial != reference:
                raise AssertionError(
                    "Initialization, counts or noise differ across arms."
                )
            slots = np.array([model.wv.key_to_index[n] for n in range(training.nodes)])
            np.testing.assert_array_equal(model.wv.expandos["count"][slots], counts)
            keep = keep_probability(counts, threshold)
            stored_keep = model.wv.expandos["sample_int"][slots] / (2**32 - 1)
            np.testing.assert_allclose(stored_keep, keep, rtol=0, atol=1 / (2**32 - 1))
            print(f"Training graph {seed}, sample={threshold:g}", flush=True)
            effective, raw = train_checked(model, walks)
            if raw != counts.sum() * training.epochs or not 0 < effective <= raw:
                raise AssertionError("Unexpected Gensim training-token accounting.")
            if threshold == 0 and effective != raw:
                raise AssertionError("Unfiltered training dropped tokens.")
            center = model.wv.vectors[slots].astype(np.float64)
            context = model.syn1neg[slots].astype(np.float64)
            raw_norm = np.linalg.norm(center, axis=1)
            balanced_norm = balanced_left_norm(center, context)
            if not np.all(np.isfinite([raw_norm, balanced_norm])):
                raise ValueError("Nonfinite embedding norms.")
            for node in range(training.nodes):
                rows.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "node": node,
                        "degree": int(degree[node]),
                        "tokens": int(counts[node]),
                        "keep_probability": float(keep[node]),
                        "raw_norm": float(raw_norm[node]),
                        "balanced_norm": float(balanced_norm[node]),
                    }
                )
            checks.append(
                {
                    "seed": seed,
                    "threshold": threshold,
                    **common,
                    "initial": initial,
                    "sample_int_sha256": digest_array(model.wv.expandos["sample_int"]),
                    "raw_tokens": int(raw),
                    "effective_tokens": int(effective),
                    "downsampled_nodes": int((keep < 1).sum()),
                    "input_sha256": digest_array(center),
                    "output_sha256": digest_array(context),
                    "expected_kept_tokens_per_epoch": float(counts @ keep),
                }
            )
            write_json(output / "checks.json", checks)
    with (output / "outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, config)
    write_json(output / "summary.json", summary)
    plot(summary, output, config)
    report(summary, checks, output)
    names = [
        "plan.json",
        "checks.json",
        "outcomes.csv",
        "summary.json",
        "curves.png",
        "README.md",
    ]
    write_json(completion, {name: sha256(output / name) for name in names})
    print(
        f"Completed {len(checks)} models, {len(rows)} node outcomes: {output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(Config(output=args.output))


if __name__ == "__main__":
    main()
