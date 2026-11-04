"""Report all frozen trajectories, without selecting conditions or fitting models.

Inputs: complete results from run.py. Outputs: summary.json, README and five plots,
with a separate report manifest. Run: python3 scripts/training_trajectory/report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run as experiment
from matplotlib.ticker import NullFormatter
from negative_sampling_control import curve

base = experiment.baseline
FIELDS = ("raw_norm", "balanced_norm", "node_loss", "score_change_rms_by_node")


def aggregate(
    graphs: list[dict[str, Any]], snapshots: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    degrees = np.concatenate([g["degree"] for g in graphs])
    log_edges = np.unique(np.quantile(np.log(degrees), np.linspace(0, 1, 9)))
    labels = [
        np.digitize(np.log(g["degree"]), log_edges[1:-1], right=True) for g in graphs
    ]
    bins = []
    for label in np.unique(np.concatenate(labels)):
        means, counts, all_degrees = [], [], []
        for graph, snapshot, group in zip(graphs, snapshots, labels, strict=True):
            chosen = group == label
            values = np.array(snapshot[field])
            counts.append(int(chosen.sum()))
            means.append(float(values[chosen].mean()) if chosen.any() else None)
            all_degrees.extend(np.array(graph["degree"])[chosen])
        observed = [v for v in means if v is not None]
        bins.append(
            {
                "degree": float(np.exp(np.log(all_degrees).mean())),
                "node_counts_by_seed": counts,
                "values_by_seed": means,
                "graphs": len(observed),
                "mean": float(np.mean(observed)),
                "standard_error": float(
                    np.std(observed, ddof=1) / np.sqrt(len(observed))
                )
                if len(observed) > 1
                else None,
            }
        )
    return bins


def summarize(runs: list[dict[str, Any]], graphs: dict[str, Any]) -> dict[str, Any]:
    per_graph, pooled, binned, progress = [], [], [], []
    config = experiment.Config()
    for schedule in config.schedules:
        for threshold in base.Config().thresholds:
            local = sorted(
                [
                    r
                    for r in runs
                    if r["schedule"] == schedule and r["threshold"] == threshold
                ],
                key=lambda r: r["seed"],
            )
            if [r["seed"] for r in local] != list(range(base.Config().training.seeds)):
                raise ValueError("Incomplete or duplicated seed set.")
            group_graphs = [graphs[r["graph_file"]] for r in local]
            for epoch in config.checkpoints:
                snapshots = [
                    next(s for s in r["snapshots"] if s["epoch"] == epoch)
                    for r in local
                ]
                common = {"schedule": schedule, "threshold": threshold, "epoch": epoch}
                for run, graph, snapshot in zip(
                    local, group_graphs, snapshots, strict=True
                ):
                    progress.append(
                        {
                            **common,
                            "seed": run["seed"],
                            **{
                                k: v
                                for k, v in snapshot.items()
                                if isinstance(v, (int, float)) and k != "epoch"
                            },
                        }
                    )
                    for response in FIELDS[:2]:
                        per_graph.append(
                            {
                                **common,
                                "seed": run["seed"],
                                "response": response,
                                **curve(
                                    np.array(graph["degree"]),
                                    np.array(snapshot[response]),
                                    8,
                                ),
                            }
                        )
                for field in FIELDS:
                    binned.append(
                        {
                            **common,
                            "field": field,
                            "bins": aggregate(group_graphs, snapshots, field),
                        }
                    )
                for response in FIELDS[:2]:
                    pooled.append(
                        {
                            **common,
                            "response": response,
                            **curve(
                                np.concatenate([g["degree"] for g in group_graphs]),
                                np.concatenate([s[response] for s in snapshots]),
                                8,
                            ),
                        }
                    )
    return {
        "per_graph": per_graph,
        "pooled": pooled,
        "binned": binned,
        "progress": progress,
    }


def mean_se(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def plot_curves(summary: dict[str, Any], output: Path, schedule: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, 5))
    for row, response in enumerate(FIELDS[:2]):
        for col, threshold in enumerate(base.Config().thresholds):
            ax = axes[row, col]
            for curve_result in summary["per_graph"]:
                if (
                    curve_result["schedule"] == schedule
                    and curve_result["threshold"] == threshold
                    and curve_result["response"] == response
                    and curve_result["epoch"] == 20
                ):
                    ax.plot(
                        curve_result["degree"],
                        curve_result["mean_norm"],
                        color="grey",
                        alpha=0.35,
                        lw=0.6,
                    )
            for color, epoch in zip(
                colors, experiment.Config().checkpoints[1:], strict=True
            ):
                record = next(
                    r
                    for r in summary["binned"]
                    if r["schedule"] == schedule
                    and r["threshold"] == threshold
                    and r["field"] == response
                    and r["epoch"] == epoch
                )
                bins = record["bins"]
                ax.errorbar(
                    [b["degree"] for b in bins],
                    [b["mean"] for b in bins],
                    yerr=[b["standard_error"] for b in bins],
                    label=f"Epoch {epoch}",
                    color=color,
                    marker="o",
                    ms=3,
                    capsize=2,
                    lw=1.2,
                )
            ax.set_xscale("log")
            ax.set_xlim(2.7, 20)
            ax.set_xticks([3, 4, 6, 8, 12, 16], labels=[3, 4, 6, 8, 12, 16])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(
                "Filtering off" if threshold == 0 else f"sample = {threshold:g}"
            )
            ax.set_xlabel("Degree (geometric bin mean)")
            ax.set_ylabel(response.replace("_", " "))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{schedule}: graph means ± SE; faint lines are individual graphs at epoch 20"
    )
    fig.tight_layout()
    fig.savefig(output / f"curves_{schedule}.png", dpi=170)
    plt.close(fig)


def plot_unfiltered_comparison(summary: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharex=True)
    schedules = (
        ("baseline_then_floor", "Original schedule + continuation", "#2563a6"),
        ("constant", "Constant learning rate 0.0025", "#c9681a"),
    )
    for ax, response in zip(axes, FIELDS[:2], strict=True):
        for schedule, label, color in schedules:
            for record in summary["per_graph"]:
                if (
                    record["schedule"] == schedule
                    and record["threshold"] == 0
                    and record["response"] == response
                    and record["epoch"] == 20
                ):
                    ax.plot(
                        record["degree"],
                        record["mean_norm"],
                        color=color,
                        alpha=0.2,
                        lw=0.7,
                    )
            bins = next(
                r["bins"]
                for r in summary["binned"]
                if r["schedule"] == schedule
                and r["threshold"] == 0
                and r["field"] == response
                and r["epoch"] == 20
            )
            ax.errorbar(
                [b["degree"] for b in bins],
                [b["mean"] for b in bins],
                yerr=[b["standard_error"] for b in bins],
                color=color,
                label=label,
                marker="o",
                ms=3,
                capsize=3,
            )
        ax.set_xscale("log")
        ax.set_xlim(2.7, 20)
        ax.set_xticks([3, 4, 6, 8, 12, 16], labels=[3, 4, 6, 8, 12, 16])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Degree (geometric bin mean)")
        ax.set_ylabel(response.replace("_", " "))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Filtering off, epoch 20: five paired graphs; means ± SE")
    fig.tight_layout()
    fig.savefig(output / "unfiltered_schedule_comparison.png", dpi=200)
    fig.savefig(
        output / "unfiltered_schedule_comparison.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_progress(
    summary: dict[str, Any], output: Path, field: str, filename: str
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    epochs = experiment.Config().checkpoints[1:]
    for row, schedule in enumerate(experiment.Config().schedules):
        for col, threshold in enumerate(base.Config().thresholds):
            ax = axes[row, col]
            for seed in range(5):
                records = sorted(
                    [
                        r
                        for r in summary["progress"]
                        if r["schedule"] == schedule
                        and r["threshold"] == threshold
                        and r["seed"] == seed
                        and r["epoch"] > 0
                    ],
                    key=lambda r: r["epoch"],
                )
                ax.plot(epochs, [r[field] for r in records], alpha=0.45, lw=0.8)
            means, errors = [], []
            for epoch in epochs:
                values = [
                    r[field]
                    for r in summary["progress"]
                    if r["schedule"] == schedule
                    and r["threshold"] == threshold
                    and r["epoch"] == epoch
                ]
                mean, error = mean_se(values)
                means.append(mean)
                errors.append(error)
            ax.errorbar(
                epochs, means, yerr=errors, color="black", marker="o", ms=3, capsize=3
            )
            ax.set_xscale("log")
            ax.set_xticks(epochs, labels=epochs)
            ax.set_title(f"{schedule}\nsample = {threshold:g}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(field.replace("_", " "))
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=170)
    plt.close(fig)


def plot_degree_loss(summary: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    epochs = experiment.Config().checkpoints[1:]
    for row, schedule in enumerate(experiment.Config().schedules):
        for col, threshold in enumerate(base.Config().thresholds):
            ax = axes[row, col]
            records = sorted(
                [
                    r
                    for r in summary["binned"]
                    if r["schedule"] == schedule
                    and r["threshold"] == threshold
                    and r["field"] == "node_loss"
                    and r["epoch"] > 0
                ],
                key=lambda r: r["epoch"],
            )
            for index, group in enumerate(records[0]["bins"]):
                ax.plot(
                    epochs,
                    [r["bins"][index]["mean"] for r in records],
                    marker="o",
                    ms=3,
                    label=f"degree {group['degree']:.1f}",
                )
            ax.set_xscale("log")
            ax.set_xticks(epochs, labels=epochs)
            ax.set_title(f"{schedule}\nsample = {threshold:g}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean node loss per expected pair")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Degree-bin loss: nodes weighted equally within bins; compare changes over time"
    )
    fig.tight_layout()
    fig.savefig(output / "degree_loss.png", dpi=170)
    plt.close(fig)


def write_report(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Training trajectories: issue 21",
        "",
        "Forty continuous runs: five graphs × four filtering settings × two schedules.",
        "All 20 baseline continuations reproduce the corrected epoch-five input/output",
        "hashes and effective-token counts exactly. Initialization is checkpoint zero.",
        "",
        "The baseline schedule anneals over five epochs, then stays at 0.0001.",
        "The other schedule uses a constant 0.0025.",
        "Epoch count does not redefine either schedule.",
        "",
        "## Interior-peak counts and pooled peak degrees",
        "",
        "Cells show interior maxima / 5 graphs, followed by pooled peak degree.",
        "A boundary maximum is not an inverted-U. These are descriptive quantile-bin",
        "maxima, not statistical tests. Eight requested bins give six occupied bins.",
        "",
    ]
    for response in FIELDS[:2]:
        lines += [
            f"### {response}",
            "",
            "| schedule | sample | epoch 1 | epoch 2 | epoch 5 | epoch 10 | epoch 20 |",
            "|:---|---:|:---|:---|:---|:---|:---|",
        ]
        for schedule in experiment.Config().schedules:
            for threshold in base.Config().thresholds:
                cells = []
                for epoch in experiment.Config().checkpoints[1:]:
                    local = [
                        r
                        for r in summary["per_graph"]
                        if r["schedule"] == schedule
                        and r["threshold"] == threshold
                        and r["response"] == response
                        and r["epoch"] == epoch
                    ]
                    pooled = next(
                        r
                        for r in summary["pooled"]
                        if r["schedule"] == schedule
                        and r["threshold"] == threshold
                        and r["response"] == response
                        and r["epoch"] == epoch
                    )
                    peak = pooled["degree"][pooled["peak_bin"] - 1]
                    cells.append(
                        f"{sum(r['interior_peak'] for r in local)}/5; {peak:.2f}"
                    )
                lines.append(
                    f"| {schedule} | {threshold:g} | " + " | ".join(cells) + " |"
                )
        lines.append("")
    lines += [
        "## Fixed expected-objective diagnostic",
        "",
        "Loss is normalized per expected positive pair; compare changes within",
        "each filtering condition. Values are graph means ± standard errors.",
        "",
        "| schedule | sample | loss epoch 1 | loss epoch 5 | loss epoch 20 | "
        "mean reduction 5→20 |",
        "|:---|---:|:---|:---|:---|---:|",
    ]
    for schedule in experiment.Config().schedules:
        for threshold in base.Config().thresholds:
            values = {}
            for epoch in (1, 5, 20):
                values[epoch] = [
                    r["expected_loss"]
                    for r in sorted(summary["progress"], key=lambda r: r["seed"])
                    if r["schedule"] == schedule
                    and r["threshold"] == threshold
                    and r["epoch"] == epoch
                ]
            cells = [
                f"{mean:.6f} ± {se:.6f}"
                for mean, se in [mean_se(values[e]) for e in (1, 5, 20)]
            ]
            reduction = np.mean((np.array(values[5]) - values[20]) / values[5]) * 100
            lines.append(
                f"| {schedule} | {threshold:g} | "
                + " | ".join(cells)
                + f" | {reduction:.3f}% |"
            )
    max_positive = max(
        r["positive_mass_outside_update_window"] for r in summary["progress"]
    )
    max_negative = max(
        r["negative_mass_outside_update_window"] for r in summary["progress"]
    )
    lines += [
        "",
        "## Numerical limits and files",
        "",
        "Maximum expected mass outside Gensim's open (-6,6) update interval: "
        f"positive {max_positive:.6%}, negative {max_negative:.6%}.",
        "The stable logistic loss and its gradient are idealized diagnostics:",
        "Gensim uses approximate sigmoid arithmetic and skips updates outside (-6,6).",
        "Independent retention and uniform integer draws are assumed for expectation.",
        "A stable curve, small gradient, or flat loss is not proof of convergence.",
        "These are in-sample diagnostics, not held-out predictive performance.",
        "",
        "`runs/` contains every node's norms, loss and score changes at checkpoints,",
        "plus actual effective-token accounting. `graphs/` contains expected positive",
        "exposure and hashes; expected pairs are not measured realized training pairs.",
        "Score changes span unequal checkpoint intervals: 1, 1, 3, 5 and 10 epochs.",
        "They are displacements over those intervals, not rates or speeds.",
        "The idealized gradient is neither a Gensim update nor a gauge-invariant",
        "distance to an optimum.",
        "`summary.json` preserves all graph curves, bins/counts and progress values.",
        "Norm plots show graph means ± SE, with individual epoch-20 curves in grey.",
        "The loss plots show all graph trajectories; degree loss plots show bin means.",
        "Degree-bin losses weight nodes equally after normalizing each node by its",
        "expected positive-pair count. Global loss weights by expected exposure.",
        "",
        "Neither the training budget across filtering settings nor embedding dimension",
        "is matched to the separate controlled trainer in this experiment. Generality,",
        "convergence and the cause of degree-dependent optimization remain unproven.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=experiment.Config.output)
    args = parser.parse_args()
    config = experiment.Config(output=args.output)
    output = (config.root / config.output).resolve()
    completion = json.loads((output / "completion.json").read_text())
    experiment.verify_artifacts(output, completion)
    plan = json.loads((output / "plan.json").read_text())
    experiment.verify_artifacts(config.root, plan["source_sha256"])
    runs = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*.json"))]
    if len(runs) != 40:
        raise ValueError("Expected the complete forty-trajectory sweep.")
    if not all(
        next(s for s in r["snapshots"] if s["epoch"] == 5).get("baseline_bridge_passed")
        for r in runs
        if r["schedule"] == "baseline_then_floor"
    ):
        raise ValueError("Missing exact baseline bridge.")
    graphs = {
        r["graph_file"]: json.loads((output / r["graph_file"]).read_text())
        for r in runs
    }
    summary = summarize(runs, graphs)
    base.write_json(output / "summary.json", summary)
    for schedule in config.schedules:
        plot_curves(summary, output, schedule)
    plot_progress(summary, output, "expected_loss", "loss.png")
    plot_progress(summary, output, "idealized_gradient_rms", "gradient.png")
    plot_degree_loss(summary, output)
    plot_unfiltered_comparison(summary, output)
    write_report(summary, output)
    artifacts = [
        output / "summary.json",
        output / "README.md",
        *sorted(output.glob("*.png")),
        *sorted(output.glob("*.pdf")),
    ]
    base.write_json(
        output / "report_manifest.json",
        {
            "script_sha256": base.sha256(Path(__file__)),
            "input_completion_sha256": base.sha256(output / "completion.json"),
            "artifact_sha256": {p.name: base.sha256(p) for p in artifacts},
        },
    )
    print(f"Reported all {len(runs)} trajectories: {output}")


if __name__ == "__main__":
    main()
