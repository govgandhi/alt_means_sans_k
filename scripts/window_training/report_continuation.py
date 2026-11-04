"""Pair exact context KL with trained norms across every fixed-noise window.

Reads complete continuation and original graph metadata; writes summary, tables,
figures and report manifest in a separate report/ directory. No fitting occurs.
Run: python3 scripts/window_training/report_continuation.py [--output /run/root]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_training import continue_windows as experiment
from window_training import expected_window as io
from window_training.report import COLORS, draw_curve, finish, summarize_nodes


def aggregate_drops(record: dict[str, Any]) -> dict[str, Any]:
    absolute = np.array([r["high_minus_penultimate"] for r in record["per_graph"]])
    fractional = np.array(
        [-r["high_minus_penultimate"] / r["values"][-2] for r in record["per_graph"]]
    )
    return {
        "absolute_by_graph": absolute.tolist(),
        "absolute_mean": float(absolute.mean()),
        "absolute_se": float(absolute.std(ddof=1) / np.sqrt(len(absolute))),
        "fractional_drop_by_graph": fractional.tolist(),
        "fractional_mean": float(fractional.mean()),
        "fractional_se": float(fractional.std(ddof=1) / np.sqrt(len(fractional))),
    }


def summarize(output: Path) -> dict[str, Any]:
    plan = io.read_json(output / "plan.json")
    experiment.verify_hashes(output, io.read_json(output / "completion.json"))
    experiment.verify_hashes(experiment.Config.root, plan["source_sha256"])
    reference = experiment.Config.root / plan["config"]["reference"]
    graphs = [io.read_json(reference / f"graphs/seed{seed}.json") for seed in range(5)]
    degrees = [np.array(g["degree"]) for g in graphs]
    runs = [io.read_json(p) for p in sorted(output.glob("*.json.gz"))]
    if len(runs) != 50:
        raise ValueError(
            "Both complete window and conditioning-control sweeps required."
        )
    balanced = [r for r in runs if r["balanced_restart"]]
    data: dict[str, Any] = {
        "degrees": [d.tolist() for d in degrees],
        "windows": [],
        "controls": [],
        "progress": [],
        "endpoints": [],
        "initialization_comparison": [],
    }
    for length in (1, 2, 5, 10):
        local = [r for r in balanced if r["window"] == length]
        if len(local) != 10:
            raise ValueError("Missing window starts.")
        norms = [
            np.mean(
                [r["snapshots"][-1]["balanced_norm"] for r in local if r["seed"] == s],
                axis=0,
            )
            for s in range(5)
        ]
        kl = [np.array(g["windows"][str(length)]["context_kl"]) for g in graphs]
        norm_curve = summarize_nodes(degrees, norms)
        kl_curve = summarize_nodes(degrees, kl)
        data["windows"].append(
            {
                "window": length,
                "norm": norm_curve,
                "kl": kl_curve,
                "norm_drop": aggregate_drops(norm_curve),
                "kl_drop": aggregate_drops(kl_curve),
                "stability_reached": sum(r["stability_reached"] for r in local),
                "norm_log_degree_slopes": [
                    float(np.polyfit(np.log(d), v, 1)[0])
                    for d, v in zip(degrees, norms, strict=True)
                ],
                "kl_log_degree_slopes": [
                    float(np.polyfit(np.log(d), v, 1)[0])
                    for d, v in zip(degrees, kl, strict=True)
                ],
            }
        )
        for seed in range(5):
            pair = sorted(
                [r for r in local if r["seed"] == seed],
                key=lambda r: r["initialization"],
            )
            a, b = [r["snapshots"][-1] for r in pair]
            data["initialization_comparison"].append(
                {
                    "seed": seed,
                    "window": length,
                    "loss_difference": abs(a["expected_loss"] - b["expected_loss"]),
                    "balanced_node_rms_difference": float(
                        np.sqrt(
                            np.mean(
                                (np.array(a["balanced_norm"]) - b["balanced_norm"]) ** 2
                            )
                        )
                    ),
                    "bin_max_absolute_difference": float(
                        np.max(
                            abs(np.array(a["curve"]["values"]) - b["curve"]["values"])
                        )
                    ),
                    "upper_gap_difference": abs(
                        a["curve"]["high_minus_penultimate"]
                        - b["curve"]["high_minus_penultimate"]
                    ),
                }
            )
    for run in runs:
        identity = {
            key: run[key]
            for key in ("seed", "window", "initialization", "balanced_restart")
        }
        for snapshot in run["snapshots"]:
            data["progress"].append(
                {
                    **identity,
                    **{
                        k: v
                        for k, v in snapshot.items()
                        if k not in ("raw_norm", "balanced_norm", "node_loss")
                    },
                }
            )
        last = run["snapshots"][-1]
        free_infimum = graphs[run["seed"]]["windows"][str(run["window"])]["free_score"][
            "unrestricted_loss_infimum"
        ]
        data["endpoints"].append(
            {
                **identity,
                "stability_reached": run["stability_reached"],
                "blocks": last["block"],
                "iterations": last["cumulative_iterations"],
                "loss": last["expected_loss"],
                "unrestricted_loss_infimum": free_infimum,
                "objective_suboptimality_upper_bound": last["expected_loss"]
                - free_infimum,
                "initial_loss": run["snapshots"][0]["expected_loss"],
                "gradient": last["balanced_gradient_max"],
                "final_gates": last["stability_gates"],
                "max_balanced_norm": max(last["balanced_norm"]),
                "max_relative_bin_change": last["max_relative_bin_change"],
                "absolute_upper_gap_change": last["absolute_upper_gap_change"],
                "curve": last["curve"],
                "balanced_norm": last["balanced_norm"],
            }
        )
        if run["window"] == 10 and not run["balanced_restart"]:
            match = next(
                r
                for r in balanced
                if r["window"] == 10
                and r["seed"] == run["seed"]
                and r["initialization"] == run["initialization"]
            )
            raw = run["snapshots"][2]
            bal = match["snapshots"][2]
            data["controls"].append(
                {
                    "seed": run["seed"],
                    "initialization": run["initialization"],
                    "initial_loss": run["snapshots"][0]["expected_loss"],
                    "raw_loss": raw["expected_loss"],
                    "balanced_loss": bal["expected_loss"],
                    "raw_gradient": raw["balanced_gradient_max"],
                    "balanced_gradient": bal["balanced_gradient_max"],
                    "raw_iterations": raw["cumulative_iterations"],
                    "balanced_iterations": bal["cumulative_iterations"],
                }
            )
    return data


def plots(data: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, color in zip(data["windows"], COLORS, strict=True):
        draw_curve(axes[0, 0], row["kl"], f"Window {row['window']}", color)
        draw_curve(
            axes[0, 1],
            row["norm"],
            f"Window {row['window']} ({row['stability_reached']}/10 stable)",
            color,
        )
        for seed in range(5):
            for offset in (100000, 200000):
                progress = [
                    r
                    for r in data["progress"]
                    if r["balanced_restart"]
                    and r["window"] == row["window"]
                    and r["seed"] == seed
                    and r["initialization"] == seed + offset
                ]
                axes[1, 1].plot(
                    [r["block"] for r in progress],
                    [r["curve"]["high_minus_penultimate"] for r in progress],
                    color=color,
                    alpha=0.35,
                    lw=0.8,
                )
    axes[0, 0].set(title="A  Exact context specificity", ylabel="Context KL (nats)")
    axes[0, 1].set(title="B  Trained balanced norms", ylabel="Balanced center norm")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for field, color, label in [
        ("kl_drop", "#333333", "Context KL"),
        ("norm_drop", "#bf631c", "Trained norm"),
    ]:
        for seed in range(5):
            axes[1, 0].plot(
                [r["window"] for r in data["windows"]],
                [r[field]["fractional_drop_by_graph"][seed] for r in data["windows"]],
                color=color,
                alpha=0.2,
                lw=0.7,
            )
        axes[1, 0].errorbar(
            [r["window"] for r in data["windows"]],
            [r[field]["fractional_mean"] for r in data["windows"]],
            yerr=[r[field]["fractional_se"] for r in data["windows"]],
            color=color,
            marker="o",
            capsize=3,
            label=label,
        )
    axes[1, 0].set(
        title="C  Paired upper-degree declines",
        xlabel="Window length",
        ylabel="(Preceding bin − highest bin) / preceding bin",
        xticks=[1, 2, 5, 10],
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(
        title="D  All starts across continued training",
        xlabel="Continuation block (up to 250 iterations each)",
        ylabel="Highest minus preceding bin, balanced norm",
    )
    axes[1, 1].axhline(0, color="black", ls=":", lw=0.6)
    fig.suptitle(
        "Same graphs, uniform exposure and fixed stationary negative noise\n"
        "Five graphs × two starts; means ± graph SE; check endpoint stability",
        fontsize=11,
    )
    finish(fig, destination / "kl_and_trained_norms")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row, color in zip(data["windows"][2:], COLORS[2:], strict=True):
        draw_curve(axes[0], row["kl"], f"Window {row['window']}", color)
        draw_curve(
            axes[1],
            row["norm"],
            f"Window {row['window']} ({row['stability_reached']}/10 stable)",
            color,
        )
    axes[0].set(title="Exact context KL", ylabel="Context KL (nats)")
    axes[1].set(title="Trained balanced norms", ylabel="Balanced center norm")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Windows 5 → 10: same graphs, exposure and negative noise")
    finish(fig, destination / "wide_window_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(
        axes,
        ("loss", "gradient"),
        ("Expected loss", "Canonical factor-gradient max"),
        strict=True,
    ):
        x = np.array([r["raw_" + key] for r in data["controls"]])
        y = np.array([r["balanced_" + key] for r in data["controls"]])
        ax.scatter(x, y, s=30)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="grey", ls="--")
        ax.set(
            xlabel="Raw-coordinate restart: " + label,
            ylabel="Balanced restart: " + label,
        )
        if key == "gradient":
            ax.set_xscale("log")
            ax.set_yscale("log")
    fig.suptitle("Matched window-10 controls: two fresh-memory blocks in each arm")
    finish(fig, destination / "conditioning_controls")
    fig, axes = plt.subplots(4, 5, figsize=(14, 10), sharex=True, sharey="row")
    for row, length in enumerate((1, 2, 5, 10)):
        for seed in range(5):
            ax = axes[row, seed]
            for offset, color in zip((100000, 200000), COLORS, strict=False):
                run = next(
                    r
                    for r in data["endpoints"]
                    if r["balanced_restart"]
                    and r["seed"] == seed
                    and r["window"] == length
                    and r["initialization"] == seed + offset
                )
                ax.scatter(
                    data["degrees"][seed],
                    run["balanced_norm"],
                    s=3,
                    alpha=0.25,
                    color=color,
                    rasterized=True,
                    label=f"Start {offset // 100000}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"Graph {seed}")
            if seed == 0:
                ax.set_ylabel(f"Window {length}\nBalanced norm")
            if row == 3:
                ax.set_xlabel("Degree")
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.suptitle("Every node and both starts after continuation (log axes)")
    finish(fig, destination / "all_final_nodes")


def tables(data: dict[str, Any], destination: Path) -> None:
    rows = [
        "# Generated fixed-noise comparison",
        "",
        "Average starts within graph before SE; gap = highest − preceding bin.",
        "Negative gaps indicate a decline; positive gaps indicate a rise.",
        "Stability requires every gate for two blocks; no global optimum claim.",
        "",
        "| Window | KL upper gap ± SE | Norm upper gap ± SE | Stable starts |",
        "|---|---:|---:|---:|",
    ]
    for r in data["windows"]:
        k, n = r["kl_drop"], r["norm_drop"]
        rows.append(
            f"| {r['window']} | {k['absolute_mean']:.6f} ± {k['absolute_se']:.6f} | "
            f"{n['absolute_mean']:.6f} ± {n['absolute_se']:.6f} | "
            f"{r['stability_reached']}/10 |"
        )
    rows += [
        "",
        "## All initialization disagreements",
        "",
        "| Graph | Window | Absolute loss difference | Node norm RMS difference | "
        "Maximum bin difference | Upper-gap difference |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in data["initialization_comparison"]:
        rows.append(
            f"| {r['seed']} | {r['window']} | {r['loss_difference']:.6g} | "
            f"{r['balanced_node_rms_difference']:.6g} | "
            f"{r['bin_max_absolute_difference']:.6g} | "
            f"{r['upper_gap_difference']:.6g} |"
        )
    rows += [
        "",
        "## Matched conditioning controls at block two",
        "",
        "| Graph | Start | Raw loss | Balanced loss | Raw gradient | "
        "Balanced gradient | Raw/balanced iterations |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in data["controls"]:
        rows.append(
            f"| {r['seed']} | {r['initialization']} | {r['raw_loss']:.7f} | "
            f"{r['balanced_loss']:.7f} | {r['raw_gradient']:.5g} | "
            f"{r['balanced_gradient']:.5g} | "
            f"{r['raw_iterations']}/{r['balanced_iterations']} |"
        )
    rows += [
        "",
        "## Every final endpoint",
        "",
        "| Graph | Window | Start | Balanced restart | Blocks | Iterations | "
        "Loss | Gradient | Stable | Last maximum relative bin change |",
        "|---|---|---|---|---|---|---:|---:|---|---:|",
    ]
    for r in data["endpoints"]:
        rows.append(
            f"| {r['seed']} | {r['window']} | {r['initialization']} | "
            f"{r['balanced_restart']} | {r['blocks']} | {r['iterations']} | "
            f"{r['loss']:.7f} | {r['gradient']:.5g} | {r['stability_reached']} | "
            f"{r['max_relative_bin_change']:.5g} |"
        )
    rows += [
        "",
        "## Loss accuracy versus norm stability",
        "",
        "The unrestricted infimum is a lower bound on the rank-24 infimum.",
        "Their difference bounds objective suboptimality above, not norm error.",
        "",
        "| Graph | Window | Start | Excess over unrestricted infimum | "
        "Last maximum relative bin change | Stable |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in data["endpoints"]:
        if r["balanced_restart"]:
            rows.append(
                f"| {r['seed']} | {r['window']} | {r['initialization']} | "
                f"{r['objective_suboptimality_upper_bound']:.8g} | "
                f"{r['max_relative_bin_change']:.5g} | {r['stability_reached']} |"
            )
    (destination / "tables.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=experiment.Config.output)
    args = parser.parse_args()
    output = experiment.Config.root / args.output
    destination = output / "report"
    destination.mkdir(exist_ok=True)
    data = summarize(output)
    io.write_json(destination / "summary.json", data)
    plots(data, destination)
    tables(data, destination)
    io.write_json(
        destination / "manifest.json",
        {
            "script_sha256": io.sha(Path(__file__)),
            "input_completion_sha256": io.sha(output / "completion.json"),
            "artifact_sha256": {
                name: io.sha(destination / name)
                for name in (
                    "summary.json",
                    "tables.md",
                    "kl_and_trained_norms.png",
                    "kl_and_trained_norms.pdf",
                    "conditioning_controls.png",
                    "conditioning_controls.pdf",
                    "all_final_nodes.png",
                    "all_final_nodes.pdf",
                    "wide_window_comparison.png",
                    "wide_window_comparison.pdf",
                )
            },
        },
    )
    print(f"Wrote paired KL/norm comparison: {destination}")


if __name__ == "__main__":
    main()
