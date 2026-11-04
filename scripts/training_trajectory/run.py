"""Run the prespecified continuous-training diagnostic for issue 21.

Inputs: the corrected no-filter control and deterministic synthetic walks.
Outputs: per-node checkpoint metrics, exact baseline bridge checks, provenance.
Run with the pinned patched environment: python3 scripts/training_trajectory/run.py
Optional --output selects a fresh directory; completed runs verify and resume.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_subsampling_origin as baseline
import numpy as np
from check_control_gensim import blas_check
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim_training import require_correct_blas, set_negative_table
from metrics import evaluate, expected_pairs, proposal_probabilities
from negative_sampling_control import digest_array, digest_walks, token_counts
from simulate_norm_frequency import (
    balanced_left_norm,
    configuration_graph,
    graph_arrays,
    sample_walks,
)
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    output: Path = Path("results/training_trajectory")
    reference: Path = Path("results/subsampling_origin")
    checkpoints: tuple[int, ...] = (0, 1, 2, 5, 10, 20)
    schedules: tuple[str, ...] = ("baseline_then_floor", "constant")
    constant_rate: float = 0.0025
    protocol: Path = Path("analysis/training_trajectory_protocol.md")


class ScheduledWord2Vec(Word2Vec):
    """Replace horizon-dependent annealing and record native epoch token counts."""

    def __init__(self, *, decay_epochs: int, **kwargs: Any) -> None:
        self.decay_epochs = decay_epochs
        self.epoch_counts: list[dict[str, int]] = []
        super().__init__(**kwargs)

    def _get_next_alpha(self, epoch_progress: float, cur_epoch: int) -> float:
        progress = (cur_epoch + epoch_progress) / self.decay_epochs
        alpha = max(
            self.min_alpha, self.alpha - (self.alpha - self.min_alpha) * progress
        )
        self.min_alpha_yet_reached = alpha
        return alpha

    def _train_epoch(self, *args: Any, **kwargs: Any) -> tuple[int, int, int]:
        effective, raw, jobs = super()._train_epoch(*args, **kwargs)
        self.epoch_counts.append(
            {
                "epoch": len(self.epoch_counts) + 1,
                "effective_tokens": int(effective),
                "raw_tokens": int(raw),
                "jobs": jobs,
            }
        )
        return effective, raw, jobs


def make_model(
    walks: list[list[int]], config: Config, threshold: float, seed: int, schedule: str
) -> ScheduledWord2Vec:
    source = baseline.Config()
    training = source.training
    if schedule not in config.schedules:
        raise ValueError(f"Unknown schedule {schedule}")
    alpha = (
        training.learning_rate
        if schedule == "baseline_then_floor"
        else config.constant_rate
    )
    end = (
        source.min_alpha if schedule == "baseline_then_floor" else config.constant_rate
    )
    model = ScheduledWord2Vec(
        decay_epochs=training.epochs,
        vector_size=training.dimension,
        window=1,
        min_count=training.min_count,
        sg=1,
        negative=training.negatives,
        ns_exponent=1.0,
        workers=1,
        epochs=max(config.checkpoints),
        seed=seed + source.optimizer_seed_offset,
        sample=threshold,
        alpha=alpha,
        min_alpha=end,
        sorted_vocab=1,
        shrink_windows=False,
    )
    model.build_vocab(walks)
    if set(model.wv.index_to_key) != set(range(training.nodes)):
        raise ValueError("Every node must survive the vocabulary cutoff.")
    set_negative_table(model)
    return model


class Recorder(CallbackAny2Vec):
    def __init__(
        self,
        config: Config,
        positive: np.ndarray,
        negative: np.ndarray,
        reference: dict[str, Any] | None,
    ) -> None:
        self.config, self.positive, self.negative = config, positive, negative
        self.reference = reference
        self.epoch = 0
        self.snapshots: list[dict[str, Any]] = []
        self.previous_scores: np.ndarray | None = None

    def capture(self, model: ScheduledWord2Vec) -> None:
        slots = np.array([model.wv.key_to_index[n] for n in range(len(model.wv))])
        center = model.wv.vectors[slots].astype(np.float64)
        context = model.syn1neg[slots].astype(np.float64)
        metrics, self.previous_scores = evaluate(
            center, context, self.positive, self.negative, self.previous_scores
        )
        raw_norm = np.linalg.norm(center, axis=1)
        balanced = balanced_left_norm(center, context)
        snapshot = {
            "epoch": self.epoch,
            **metrics,
            "raw_norm": raw_norm.tolist(),
            "balanced_norm": balanced.tolist(),
            "input_sha256": digest_array(center),
            "output_sha256": digest_array(context),
        }
        if self.epoch == baseline.Config().training.epochs and self.reference:
            for name in ("input_sha256", "output_sha256"):
                if snapshot[name] != self.reference[name]:
                    raise AssertionError(f"Epoch-five baseline mismatch: {name}")
            snapshot["baseline_bridge_passed"] = True
        if not np.all(np.isfinite([raw_norm, balanced, metrics["node_loss"]])):
            raise ValueError("Nonfinite trajectory output.")
        self.snapshots.append(snapshot)
        print(
            f"  checkpoint {self.epoch}: expected loss {metrics['expected_loss']:.6f}",
            flush=True,
        )

    def on_train_begin(self, model: ScheduledWord2Vec) -> None:
        self.capture(model)

    def on_epoch_end(self, model: ScheduledWord2Vec) -> None:
        self.epoch += 1
        if self.epoch in self.config.checkpoints:
            self.capture(model)


def verify_artifacts(directory: Path, manifest: dict[str, str]) -> None:
    for name, expected in manifest.items():
        if baseline.sha256(directory / name) != expected:
            raise ValueError(f"Artifact changed: {directory / name}")


def plan_run(config: Config) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    output, reference = (
        (config.root / config.output).resolve(),
        config.root / config.reference,
    )
    # Existing result directories cannot be repurposed, even with a different plan.
    if output == reference.resolve() or reference.resolve() in output.parents:
        raise ValueError("Cannot overwrite the baseline reference.")
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Output is nonempty with no matching trajectory plan.")
    reference_plan = json.loads((reference / "plan.json").read_text())
    verify_artifacts(reference, json.loads((reference / "completion.json").read_text()))
    verify_artifacts(config.root, reference_plan["source_sha256"])
    source_paths = (
        [config.root / p for p in reference_plan["source_sha256"]]
        + [Path(__file__), Path(__file__).with_name("metrics.py")]
        + [config.root / config.protocol]
    )
    packages = {
        name: importlib.metadata.version(name) for name in reference_plan["packages"]
    }
    if packages != reference_plan["packages"]:
        raise RuntimeError(
            "Runtime package versions differ from the corrected baseline."
        )
    check = blas_check()
    if check != reference_plan["blas_check"]:
        raise RuntimeError("Numerical extension differs from the baseline.")
    plan = {
        "config": {
            k: v for k, v in asdict(config).items() if k not in {"root", "output"}
        },
        "reference_completion_sha256": baseline.sha256(reference / "completion.json"),
        "source_sha256": {
            str(p.relative_to(config.root)): baseline.sha256(p)
            for p in sorted(set(source_paths))
        },
        "packages": packages,
        "blas_check": check,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    plan = json.loads(json.dumps(plan, default=str))
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Trajectory plan differs; use a fresh output directory.")
    else:
        baseline.write_json(plan_path, plan)
    checks = json.loads((reference / "checks.json").read_text())
    return output, plan, checks


def run(config: Config) -> None:
    output, plan, old_checks = plan_run(config)
    completion = output / "completion.json"
    if completion.exists():
        verify_artifacts(output, json.loads(completion.read_text()))
        print("Verified all completed trajectories; no files changed.", flush=True)
        return
    require_correct_blas()
    source, training = baseline.Config(), baseline.Config().training
    (output / "runs").mkdir(exist_ok=True)
    (output / "graphs").mkdir(exist_ok=True)
    for seed in range(training.seeds):
        graph = configuration_graph(training.nodes, training.attachment, seed)
        adjacency, neighbors, degree = graph_arrays(graph)
        walks, counts = sample_walks(
            neighbors,
            degree / degree.sum(),
            training.nodes * training.walks_per_node,
            training.walk_length,
            np.random.default_rng(seed + source.walk_seed_offset),
        )
        np.testing.assert_array_equal(counts, token_counts(walks, training.nodes))
        corpus = {
            "graph_sha256": digest_array(adjacency),
            "walks_sha256": digest_walks(walks),
            "counts_sha256": digest_array(counts),
        }
        for threshold in source.thresholds:
            tag = f"seed{seed}_t{threshold:g}"
            old = next(
                c
                for c in old_checks
                if c["seed"] == seed and c["threshold"] == threshold
            )
            for name, value in corpus.items():
                if old[name] != value:
                    raise AssertionError(f"Baseline corpus mismatch: {name}")
            for schedule in config.schedules:
                name = f"{tag}_{schedule}"
                path = output / "runs" / f"{name}.json"
                marker = path.with_suffix(".sha256")
                if marker.exists():
                    if baseline.sha256(path) != marker.read_text().strip():
                        raise ValueError(f"Changed completed trajectory {name}")
                    saved = json.loads(path.read_text())
                    verify_artifacts(
                        output,
                        {saved["graph_file"]: saved["graph_file_sha256"]},
                    )
                    print(f"Verified {name}", flush=True)
                    continue
                model = make_model(walks, config, threshold, seed, schedule)
                initial = baseline.signature(model)
                if initial != old["initial"]:
                    raise AssertionError(
                        "Initialization/RNG/noise differs from baseline."
                    )
                slots = np.array(
                    [model.wv.key_to_index[n] for n in range(training.nodes)]
                )
                retention = (
                    model.wv.expandos["sample_int"][slots].astype(np.float64) + 1
                ) / 2**32
                if threshold == 0:
                    np.testing.assert_array_equal(retention, 1)
                proposal = np.empty(training.nodes)
                proposal[np.asarray(model.wv.index_to_key)] = proposal_probabilities(
                    model.cum_table
                )
                positive = expected_pairs(np.asarray(walks), retention)
                negative = (
                    training.negatives
                    * proposal[None, :]
                    * (positive.sum(axis=1)[:, None] - positive)
                )
                if threshold == 0:
                    assert positive.sum() == 2 * (training.walk_length - 1) * len(walks)
                metadata = {
                    "seed": seed,
                    "threshold": threshold,
                    **corpus,
                    "initial": initial,
                    "degree": degree.tolist(),
                    "tokens": counts.tolist(),
                    "retention": retention.tolist(),
                    "noise_probability": proposal.tolist(),
                    "expected_positive_pairs_per_epoch": float(positive.sum()),
                    "expected_positive_pairs_by_node": positive.sum(axis=1).tolist(),
                    "positive_weights_sha256": digest_array(positive),
                    "negative_weights_sha256": digest_array(negative),
                }
                graph_path = output / "graphs" / f"{tag}.json"
                if graph_path.exists():
                    if json.loads(graph_path.read_text()) != metadata:
                        raise ValueError(
                            "Evaluation distribution changed across schedules."
                        )
                else:
                    baseline.write_json(graph_path, metadata)
                recorder = Recorder(
                    config,
                    positive,
                    negative,
                    old if schedule == "baseline_then_floor" else None,
                )
                print(f"Training {name}", flush=True)
                effective, raw = model.train(
                    walks,
                    total_examples=model.corpus_count,
                    total_words=model.corpus_total_words,
                    epochs=max(config.checkpoints),
                    start_alpha=model.alpha,
                    end_alpha=model.min_alpha,
                    callbacks=[recorder],
                )
                assert raw == counts.sum() * max(config.checkpoints)
                assert (
                    sum(c["effective_tokens"] for c in model.epoch_counts) == effective
                )
                if threshold == 0:
                    assert effective == raw
                if schedule == "baseline_then_floor":
                    prefix = model.epoch_counts[: training.epochs]
                    assert (
                        sum(c["effective_tokens"] for c in prefix)
                        == old["effective_tokens"]
                    )
                result = {
                    "seed": seed,
                    "threshold": threshold,
                    "schedule": schedule,
                    "graph_file": str(graph_path.relative_to(output)),
                    "graph_file_sha256": baseline.sha256(graph_path),
                    "epoch_counts": model.epoch_counts,
                    "snapshots": recorder.snapshots,
                }
                baseline.write_json(path, result)
                marker.write_text(baseline.sha256(path) + "\n")
    verify_artifacts(config.root, plan["source_sha256"])
    names = [
        p
        for folder in (output / "graphs", output / "runs")
        for p in sorted(folder.iterdir())
    ]
    names.append(output / "plan.json")
    baseline.write_json(
        completion, {str(p.relative_to(output)): baseline.sha256(p) for p in names}
    )
    print(f"Completed 40 trajectories: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(Config(output=args.output))


if __name__ == "__main__":
    main()
