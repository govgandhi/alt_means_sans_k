"""Fork corrected Gensim at the exact PR36 epoch-five no-filter checkpoint.

Inputs: frozen protocol/requirements, deterministic graphs/walks and a verified
PR36 training_trajectory reference (rebuild it locally with its reproduce.py on
another platform). Outputs: seed-level checkpoints, deterministic factor NPZs,
metrics, state/accounting bridges and source/runtime/artifact hashes.
Run: python3 scripts/window_training/schedule_fork.py [--output /fresh/path]
Use --seeds 0 1 for resumable subsets; --reference selects a rebuilt PR36 run.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import io
import json
import platform
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "training_trajectory"))

import numpy as np  # noqa: E402
from gensim.models.callbacks import CallbackAny2Vec  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from training_trajectory import run as trajectory  # noqa: E402
from training_trajectory.metrics import (  # noqa: E402
    evaluate,
    expected_pairs,
    proposal_probabilities,
)

baseline = trajectory.baseline
digest_array = trajectory.digest_array


@dataclass(frozen=True)
class Config:
    root: Path = SCRIPTS.parent
    output: Path = Path("results/window_training_exploration/schedule")
    reference: Path = Path("results/training_trajectory")
    protocol: Path = Path("analysis/window_training_exploration_protocol.md")
    requirements: Path = Path("analysis/window_training_exploration_requirements.txt")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    checkpoints: tuple[int, ...] = (5, 6, 7, 10, 15, 20)
    fork_epoch: int = 5
    floor_rate: float = 0.0001
    reheat_rate: float = 0.0025


def state_fingerprint(model: trajectory.ScheduledWord2Vec) -> dict[str, Any]:
    """Hash training-relevant state; exclude wall-clock and lifecycle logs."""
    return {
        **baseline.signature(model),
        "sample_int_sha256": digest_array(model.wv.expandos["sample_int"]),
        "lockf_sha256": digest_array(model.wv.vectors_lockf),
        "epoch_counts": copy.deepcopy(model.epoch_counts),
        "parameters": {
            name: getattr(model, name)
            for name in (
                "alpha",
                "min_alpha",
                "min_alpha_yet_reached",
                "epochs",
                "decay_epochs",
                "train_count",
                "seed",
                "vector_size",
                "window",
                "negative",
                "ns_exponent",
                "workers",
                "sg",
                "hs",
                "sample",
                "shrink_windows",
                "batch_words",
                "corpus_count",
                "corpus_total_words",
            )
        },
    }


def assert_state(model: trajectory.ScheduledWord2Vec, expected: dict[str, Any]) -> None:
    if state_fingerprint(model) != expected:
        raise AssertionError(
            "Training state changed: factors, RNG, corpus or parameters."
        )


def fork_model(model: trajectory.ScheduledWord2Vec) -> trajectory.ScheduledWord2Vec:
    """Deep-copy all Gensim state, verifying both values and mutable isolation."""
    expected = state_fingerprint(model)
    fork = copy.deepcopy(model)
    assert_state(fork, expected)
    for original, duplicate in (
        (model.wv.vectors, fork.wv.vectors),
        (model.syn1neg, fork.syn1neg),
        (model.cum_table, fork.cum_table),
        (model.wv.expandos["sample_int"], fork.wv.expandos["sample_int"]),
        (model.wv.vectors_lockf, fork.wv.vectors_lockf),
    ):
        if np.shares_memory(original, duplicate):
            raise AssertionError("A fork shares mutable training arrays.")
    if fork.random is model.random or fork.epoch_counts is model.epoch_counts:
        raise AssertionError("A fork shares mutable RNG or training accounting.")
    return fork


def write_factors(path: Path, center: np.ndarray, context: np.ndarray) -> None:
    """Write standard NPZ arrays with fixed ZIP metadata for byte reproducibility."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in (("center", center), ("context", context)):
            stream = io.BytesIO()
            np.save(stream, values, allow_pickle=False)
            item = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o600 << 16
            archive.writestr(item, stream.getvalue(), compresslevel=9)


class Recorder(CallbackAny2Vec):
    def __init__(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        checkpoints: tuple[int, ...],
        start_epoch: int,
        learning_rate: float | None = None,
        factor_dir: Path | None = None,
        branch: str = "baseline",
    ) -> None:
        self.positive, self.negative = positive, negative
        self.checkpoints, self.epoch = checkpoints, start_epoch
        self.start_epoch, self.learning_rate = start_epoch, learning_rate
        self.factor_dir, self.branch = factor_dir, branch
        self.snapshots: list[dict[str, Any]] = []
        self.previous_scores: np.ndarray | None = None
        self.fork_scores: np.ndarray | None = None

    def capture(self, model: trajectory.ScheduledWord2Vec) -> None:
        before = state_fingerprint(model)
        slots = np.array([model.wv.key_to_index[n] for n in range(len(model.wv))])
        center = model.wv.vectors[slots].astype(np.float64)
        context = model.syn1neg[slots].astype(np.float64)
        metrics, scores = evaluate(
            center, context, self.positive, self.negative, self.previous_scores
        )
        if self.fork_scores is None:
            self.fork_scores = scores.copy()
        after_fork = model.epoch_counts[self.start_epoch : self.epoch]
        effective = sum(row["effective_tokens"] for row in after_fork)
        snapshot = {
            "epoch": self.epoch,
            **metrics,
            "raw_norm": np.linalg.norm(center, axis=1).tolist(),
            "balanced_norm": trajectory.balanced_left_norm(center, context).tolist(),
            "score_change_from_fork_rms_by_node": np.sqrt(
                np.mean((scores - self.fork_scores) ** 2, axis=1)
            ).tolist(),
            "input_sha256": digest_array(center),
            "output_sha256": digest_array(context),
            "post_fork_effective_tokens": effective,
            "post_fork_raw_tokens": sum(row["raw_tokens"] for row in after_fork),
            "post_fork_alpha_times_effective_tokens": (
                None if self.learning_rate is None else self.learning_rate * effective
            ),
            "rng_array_sha256": before["rng_array_sha256"],
            "rng_metadata": before["rng_metadata"],
        }
        if self.factor_dir is not None:
            name = f"{self.branch}_epoch{self.epoch}.npz"
            write_factors(self.factor_dir / name, center, context)
            snapshot["factor_file"] = f"factors/{name}"
            snapshot["factor_file_sha256"] = baseline.sha256(self.factor_dir / name)
        if not np.all(
            np.isfinite(
                [snapshot["raw_norm"], snapshot["balanced_norm"], metrics["node_loss"]]
            )
        ):
            raise ValueError("Nonfinite checkpoint metrics.")
        self.previous_scores = scores
        self.snapshots.append(snapshot)
        assert_state(model, before)
        print(
            f"  {self.branch} epoch {self.epoch}: loss {metrics['expected_loss']:.6f}",
            flush=True,
        )

    def on_train_begin(self, model: trajectory.ScheduledWord2Vec) -> None:
        if self.epoch in self.checkpoints:
            self.capture(model)

    def on_epoch_end(self, model: trajectory.ScheduledWord2Vec) -> None:
        self.epoch += 1
        if self.epoch in self.checkpoints:
            self.capture(model)


def train(
    model: trajectory.ScheduledWord2Vec,
    walks: list[list[int]],
    epochs: int,
    start_rate: float,
    end_rate: float,
    recorder: Recorder | None = None,
) -> tuple[int, int]:
    return model.train(
        walks,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=epochs,
        start_alpha=start_rate,
        end_alpha=end_rate,
        callbacks=[] if recorder is None else [recorder],
    )


def verify_bridge(actual: dict[str, Any], reference: dict[str, Any]) -> None:
    for name in ("input_sha256", "output_sha256"):
        if actual[name] != reference[name]:
            raise AssertionError(
                f"Saved PR36 bridge failed at epoch {actual['epoch']}: {name}"
            )


def verify_manifest(directory: Path) -> None:
    trajectory.verify_artifacts(
        directory, json.loads((directory / "completion.json").read_text())
    )


def plan_run(config: Config) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    output, reference = (
        (config.root / config.output).resolve(),
        (config.root / config.reference).resolve(),
    )
    if (
        not config.seeds
        or len(set(config.seeds)) != len(config.seeds)
        or not set(config.seeds) <= set(range(5))
    ):
        raise ValueError("Seeds must be a nonempty unique subset of 0..4.")
    if config.fork_epoch != 5 or config.checkpoints != (5, 6, 7, 10, 15, 20):
        raise ValueError(
            "The frozen design uses epoch 5 and checkpoints 5,6,7,10,15,20."
        )
    if (config.floor_rate, config.reheat_rate) != (0.0001, 0.0025):
        raise ValueError("The frozen continuation rates are 0.0001 and 0.0025.")
    reference_plan = json.loads((reference / "plan.json").read_text())
    old = (config.root / reference_plan["config"]["reference"]).resolve()
    for protected in (reference, old, config.root / "archive"):
        if (
            output == protected
            or protected in output.parents
            or output in protected.parents
        ):
            raise ValueError("Output overlaps a protected historical directory.")
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Nonempty output has no matching recorded plan.")
    for directory in (reference, old):
        verify_manifest(directory)
    if (
        baseline.sha256(old / "completion.json")
        != reference_plan["reference_completion_sha256"]
    ):
        raise ValueError("PR36 baseline reference manifest changed.")
    trajectory.verify_artifacts(config.root, reference_plan["source_sha256"])
    packages = {
        name: importlib.metadata.version(name) for name in reference_plan["packages"]
    }
    if packages != reference_plan["packages"]:
        raise RuntimeError("Runtime packages differ from the PR36 reference.")
    for line in (config.root / config.requirements).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, expected = line.split("==")
            if importlib.metadata.version(name) != expected:
                raise RuntimeError(f"Pinned runtime mismatch: {line}")
    check = trajectory.blas_check()
    if (
        check != reference_plan["blas_check"]
        or platform.python_version() != reference_plan["python"]
    ):
        raise RuntimeError("Python or corrected numerical extension differs from PR36.")
    source_paths = set(reference_plan["source_sha256"]) | {
        str(Path(__file__).relative_to(config.root)),
        str(config.protocol),
        str(config.requirements),
        "tests/test_schedule_fork.py",
    }
    plan = {
        "design": "same epoch-five state; fixed-rate floor versus reheat; no filtering",
        "config": {
            k: v
            for k, v in asdict(config).items()
            if k not in {"root", "output", "seeds"}
        },
        "seed_universe": list(range(5)),
        "reference_completion_sha256": baseline.sha256(reference / "completion.json"),
        "source_sha256": {
            p: baseline.sha256(config.root / p) for p in sorted(source_paths)
        },
        "packages": packages,
        "blas_check": check,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    plan = json.loads(json.dumps(plan, default=str))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Plan changed; use a fresh output directory.")
    if not path.exists():
        baseline.write_json(path, plan)
    return output, reference, plan, json.loads((old / "checks.json").read_text())


def run_seed(
    config: Config,
    output: Path,
    reference: Path,
    old_checks: list[dict[str, Any]],
    seed: int,
) -> None:
    destination = output / f"seed{seed}"
    if (destination / "completion.json").exists():
        verify_manifest(destination)
        print(f"Verified completed seed {seed}", flush=True)
        return
    destination.mkdir(exist_ok=True)
    factors = destination / "factors"
    factors.mkdir(exist_ok=True)
    source, training = baseline.Config(), baseline.Config().training
    saved = json.loads(
        (reference / "runs" / f"seed{seed}_t0_baseline_then_floor.json").read_text()
    )
    graph_reference = json.loads((reference / saved["graph_file"]).read_text())
    old = next(
        row for row in old_checks if row["seed"] == seed and row["threshold"] == 0
    )
    graph = trajectory.configuration_graph(training.nodes, training.attachment, seed)
    adjacency, neighbors, degree = trajectory.graph_arrays(graph)
    walks, counts = trajectory.sample_walks(
        neighbors,
        degree / degree.sum(),
        training.nodes * training.walks_per_node,
        training.walk_length,
        np.random.default_rng(seed + source.walk_seed_offset),
    )
    np.testing.assert_array_equal(
        counts, trajectory.token_counts(walks, training.nodes)
    )
    corpus = {
        "graph_sha256": digest_array(adjacency),
        "walks_sha256": trajectory.digest_walks(walks),
        "counts_sha256": digest_array(counts),
    }
    for name, value in corpus.items():
        if old[name] != value or graph_reference[name] != value:
            raise AssertionError(f"PR36 corpus mismatch: {name}")
    model = trajectory.make_model(
        walks, trajectory.Config(), 0.0, seed, "baseline_then_floor"
    )
    if (
        baseline.signature(model) != old["initial"]
        or baseline.signature(model) != graph_reference["initial"]
    ):
        raise AssertionError("PR36 initial factors/RNG/vocabulary/noise mismatch.")
    slots = np.array([model.wv.key_to_index[n] for n in range(training.nodes)])
    retention = (model.wv.expandos["sample_int"][slots].astype(float) + 1) / 2**32
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
    for name, values in (
        ("positive_weights_sha256", positive),
        ("negative_weights_sha256", negative),
    ):
        if digest_array(values) != graph_reference[name]:
            raise AssertionError(f"PR36 objective mismatch: {name}")
    prefix_recorder = Recorder(positive, negative, (5,), 0)
    print(f"Recreating seed {seed} epoch-five checkpoint", flush=True)
    effective, raw = train(
        model, walks, 5, training.learning_rate, source.min_alpha, prefix_recorder
    )
    prefix = prefix_recorder.snapshots[-1]
    verify_bridge(prefix, old)
    verify_bridge(
        prefix,
        next(snapshot for snapshot in saved["snapshots"] if snapshot["epoch"] == 5),
    )
    if (
        model.epoch_counts != saved["epoch_counts"][:5]
        or effective != old["effective_tokens"]
        or raw != old["raw_tokens"]
    ):
        raise AssertionError("PR36 prefix token accounting mismatch.")
    frozen = state_fingerprint(model)
    forks = {name: fork_model(model) for name in ("floor", "reheat")}
    metadata = {
        "seed": seed,
        "threshold": 0.0,
        **corpus,
        "degree": degree.tolist(),
        "tokens": counts.tolist(),
        "noise_probability": proposal.tolist(),
        "expected_positive_pairs_per_epoch": float(positive.sum()),
        "expected_positive_pairs_by_node": positive.sum(axis=1).tolist(),
        "positive_weights_sha256": digest_array(positive),
        "negative_weights_sha256": digest_array(negative),
        "fork_state": frozen,
        "epoch_five_pr36_bridge_passed": True,
        "reference_run_sha256": baseline.sha256(
            reference / "runs" / f"seed{seed}_t0_baseline_then_floor.json"
        ),
        "pairing": (
            "Both complete model states are deep-copied "
            "before either continuation starts."
        ),
    }
    baseline.write_json(destination / "metadata.json", metadata)
    endings = []
    for branch, rate in (("floor", config.floor_rate), ("reheat", config.reheat_rate)):
        fork = forks[branch]
        assert_state(fork, frozen)
        recorder = Recorder(
            positive,
            negative,
            config.checkpoints,
            config.fork_epoch,
            rate,
            factors,
            branch,
        )
        effective, raw = train(fork, walks, 15, rate, rate, recorder)
        if raw != counts.sum() * 15 or effective != raw:
            raise AssertionError("Unfiltered continuation token accounting mismatch.")
        if fork.epoch_counts != saved["epoch_counts"]:
            raise AssertionError("PR36 continuation epoch/token accounting mismatch.")
        verify_bridge(recorder.snapshots[0], prefix)
        if branch == "floor":
            for snapshot in recorder.snapshots:
                matching = next(
                    (s for s in saved["snapshots"] if s["epoch"] == snapshot["epoch"]),
                    None,
                )
                if matching is not None:
                    verify_bridge(snapshot, matching)
        ending = state_fingerprint(fork)
        for name in (
            "vocabulary",
            "slot_counts_sha256",
            "negative_cdf_sha256",
            "sample_int_sha256",
            "lockf_sha256",
        ):
            if ending[name] != frozen[name]:
                raise AssertionError(
                    f"Continuation changed fixed sampling state: {name}"
                )
        endings.append(ending)
        baseline.write_json(
            destination / f"{branch}.json",
            {
                "seed": seed,
                "branch": branch,
                "learning_rate": rate,
                "initial_state_matches_fork": True,
                "final_state": ending,
                "effective_tokens": effective,
                "raw_tokens": raw,
                "epoch_counts": fork.epoch_counts,
                "snapshots": recorder.snapshots,
                "pr36_uninterrupted_floor_bridge_passed": branch == "floor",
            },
        )
        assert_state(model, frozen)
    for name in ("rng_array_sha256", "rng_metadata", "epoch_counts"):
        if endings[0][name] != endings[1][name]:
            raise AssertionError(
                f"Paired continuations diverged in random stream/accounting: {name}"
            )
    baseline.write_json(
        destination / "pairing.json",
        {
            "identical_full_fork_state": True,
            "identical_final_rng_and_token_counts": True,
            "floor_matches_uninterrupted_pr36": True,
            "alpha_times_tokens_ratio_reheat_over_floor": config.reheat_rate
            / config.floor_rate,
            "caveat": (
                "Matched token counts do not imply equal optimization effort; "
                "reheat has 25 times the learning-rate exposure."
            ),
        },
    )
    names = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != "completion.json"
    )
    baseline.write_json(
        destination / "completion.json",
        {str(path.relative_to(destination)): baseline.sha256(path) for path in names},
    )


def run(config: Config) -> None:
    output, reference, plan, old_checks = plan_run(config)
    if (output / "completion.json").exists():
        verify_manifest(output)
        for seed in range(5):
            verify_manifest(output / f"seed{seed}")
        print("Verified all completed schedule forks; no files changed.", flush=True)
        return
    trajectory.require_correct_blas()
    for seed in config.seeds:
        run_seed(config, output, reference, old_checks, seed)
    trajectory.verify_artifacts(config.root, plan["source_sha256"])
    if all((output / f"seed{seed}" / "completion.json").exists() for seed in range(5)):
        for seed in range(5):
            verify_manifest(output / f"seed{seed}")
        files = [
            output / "plan.json",
            *(output / f"seed{seed}" / "completion.json" for seed in range(5)),
        ]
        baseline.write_json(
            output / "completion.json",
            {str(path.relative_to(output)): baseline.sha256(path) for path in files},
        )
        print(f"Completed all five paired schedule forks: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    parser.add_argument("--reference", type=Path, default=Config.reference)
    parser.add_argument("--seeds", type=int, nargs="+", default=Config.seeds)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(
            Config(
                output=args.output, reference=args.reference, seeds=tuple(args.seeds)
            )
        )


if __name__ == "__main__":
    main()
