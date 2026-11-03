"""Audit raw trajectory artifacts and optionally repeat one full constant-rate run.

Run: python3 scripts/training_trajectory/verify.py [--repeat]
Reads the default complete experiment; never modifies its data. --repeat uses
seed zero, filtering off, constant rate, specified independently of its outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import run as experiment
from threadpoolctl import threadpool_limits


def verify(output: Path) -> None:
    base = experiment.baseline
    plan = json.loads((output / "plan.json").read_text())
    experiment.verify_artifacts(
        output, json.loads((output / "completion.json").read_text())
    )
    experiment.verify_artifacts(experiment.Config.root, plan["source_sha256"])
    report = json.loads((output / "report_manifest.json").read_text())
    experiment.verify_artifacts(output, report["artifact_sha256"])
    assert report["script_sha256"] == base.sha256(Path(__file__).with_name("report.py"))
    assert report["input_completion_sha256"] == base.sha256(output / "completion.json")
    runs = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*.json"))]
    expected = {
        (seed, threshold, schedule)
        for seed in range(5)
        for threshold in base.Config().thresholds
        for schedule in experiment.Config().schedules
    }
    assert {(r["seed"], r["threshold"], r["schedule"]) for r in runs} == expected
    assert len(runs) == len(expected) == 40
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["per_graph"]) == 40 * 6 * 2
    for result in summary["per_graph"]:
        run = next(
            r
            for r in runs
            if all(r[k] == result[k] for k in ("seed", "threshold", "schedule"))
        )
        snapshot = next(s for s in run["snapshots"] if s["epoch"] == result["epoch"])
        graph = json.loads((output / run["graph_file"]).read_text())
        degree = np.asarray(graph["degree"])
        values = np.asarray(snapshot[result["response"]])
        assert len(values) == 600 and np.isfinite(values).all()
        edges = np.unique(np.quantile(np.log(degree), np.arange(9) / 8))
        labels = np.searchsorted(edges[1:-1], np.log(degree), side="left")
        means = [float(values[labels == label].mean()) for label in sorted(set(labels))]
        np.testing.assert_array_equal(means, result["mean_norm"])
        assert result["peak_bin"] == np.argmax(means) + 1
    old = experiment.Config.root / plan["config"]["reference"]
    with (old / "outcomes.csv").open() as stream:
        previous = {
            (int(r["seed"]), float(r["threshold"]), int(r["node"])): r
            for r in csv.DictReader(stream)
        }
    bridges = 0
    for seed in range(5):
        for threshold in base.Config().thresholds:
            paired = [
                r for r in runs if r["seed"] == seed and r["threshold"] == threshold
            ]
            assert paired[0]["epoch_counts"] == paired[1]["epoch_counts"]
    for run in runs:
        assert [s["epoch"] for s in run["snapshots"]] == list(
            experiment.Config().checkpoints
        )
        assert len(run["epoch_counts"]) == 20
        assert all(c["raw_tokens"] == 900_000 for c in run["epoch_counts"])
        if run["schedule"] == "baseline_then_floor":
            snapshot = next(s for s in run["snapshots"] if s["epoch"] == 5)
            for response in ("raw_norm", "balanced_norm"):
                reference = [
                    float(previous[run["seed"], run["threshold"], node][response])
                    for node in range(600)
                ]
                np.testing.assert_array_equal(snapshot[response], reference)
            bridges += 1
    print(
        "Audited 40 runs and 480 graph/readout/checkpoint curves; "
        f"{bridges} exact baseline bridges."
    )


def repeat(output: Path) -> None:
    config, source = experiment.Config(), experiment.baseline.Config()
    training = source.training
    seed, threshold, schedule = 0, 0.0, "constant"
    saved = json.loads((output / "runs/seed0_t0_constant.json").read_text())
    graph = experiment.configuration_graph(training.nodes, training.attachment, seed)
    _, neighbors, degree = experiment.graph_arrays(graph)
    walks, _ = experiment.sample_walks(
        neighbors,
        degree / degree.sum(),
        training.nodes * training.walks_per_node,
        training.walk_length,
        np.random.default_rng(seed + source.walk_seed_offset),
    )
    model = experiment.make_model(walks, config, threshold, seed, schedule)
    positive = experiment.expected_pairs(np.asarray(walks), np.ones(training.nodes))
    q = np.empty(training.nodes)
    q[np.asarray(model.wv.index_to_key)] = experiment.proposal_probabilities(
        model.cum_table
    )
    negative = (
        training.negatives * q[None, :] * (positive.sum(axis=1)[:, None] - positive)
    )
    callback = experiment.Recorder(config, positive, negative, None)
    experiment.require_correct_blas()
    model.train(
        walks,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=20,
        start_alpha=model.alpha,
        end_alpha=model.min_alpha,
        callbacks=[callback],
    )
    assert callback.snapshots == saved["snapshots"]
    assert model.epoch_counts == saved["epoch_counts"]
    print(
        "Fresh seed-zero unfiltered constant-rate trajectory matches "
        "every saved checkpoint field exactly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=experiment.Config.output)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    output = (experiment.Config.root / args.output).resolve()
    with threadpool_limits(limits=1):
        verify(output)
        if args.repeat:
            repeat(output)


if __name__ == "__main__":
    main()
