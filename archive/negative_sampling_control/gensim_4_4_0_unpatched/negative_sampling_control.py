"""Isolate negative-sampler effects in the existing calibration experiment.

Inputs: the frozen prospective protocol/calibration rows and their original scripts.
Outputs: separate per-node outcomes, sampler diagnostics, curve summaries and plot.
Install analysis/negative_sampling_control_requirements.txt in a Python 3.13 venv,
then run: python3 scripts/negative_sampling_control.py
Use --output to reproduce into a different directory; completed outputs are verified
on repeat invocation. No original experiment file is modified and no model is fit
to target outcomes. See analysis/negative_sampling_control_protocol.md for scope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import prospective_norm_turnover as original
from gensim.models import Word2Vec
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ControlConfig:
    root: Path = Path(__file__).resolve().parents[1]
    output: Path = Path("results/negative_sampling_control")
    protocol: Path = Path("results/prospective_validation/protocol.json")
    calibration: Path = Path("results/prospective_validation/calibration.jsonl")
    human_protocol: Path = Path("analysis/prospective_validation_protocol.md")
    design: Path = Path("analysis/negative_sampling_control_protocol.md")
    requirements: Path = Path("analysis/negative_sampling_control_requirements.txt")
    arms: tuple[str, ...] = ("legacy", "retained", "raw")
    bins: int = 6
    min_alpha: float = 0.0001  # Gensim default used by the frozen script.
    table_domain: int = 2**31 - 1  # Gensim 4.4.0 cumulative-table domain.


def digest_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def digest_walks(walks: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for walk in walks:
        digest.update(np.asarray([len(walk), *walk], dtype="<i8").tobytes())
    return digest.hexdigest()


def token_counts(walks: list[list[int]], nodes: int) -> np.ndarray:
    counts = np.bincount(np.concatenate(walks), minlength=nodes)
    if len(counts) != nodes or np.any(counts <= 0):
        raise ValueError("Every node must occur in the corpus.")
    assert counts.sum() == sum(map(len, walks))
    return counts


def table_from_counts(counts_by_slot: np.ndarray, domain: int) -> np.ndarray:
    """Build an exponent-one table without ambiguous integer-token lookups."""
    if np.any(counts_by_slot <= 0):
        raise ValueError("Negative sampling requires positive counts for every node.")
    cumulative = np.cumsum(counts_by_slot, dtype=np.float64)
    return np.rint(cumulative / cumulative[-1] * domain).astype(np.uint32)


def table_probabilities(model: Word2Vec) -> np.ndarray:
    """Return normalized CDF masses; discrete proposal probabilities differ slightly."""
    slot_probabilities = np.diff(np.r_[0, model.cum_table.astype(np.int64)])
    result = np.empty(len(model.wv), dtype=np.float64)
    result[np.asarray(model.wv.index_to_key)] = slot_probabilities / model.cum_table[-1]
    return result


def make_template(
    walks: list[list[int]], config: original.Config, seed: int, min_alpha: float
) -> Word2Vec:
    model = Word2Vec(
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
        min_alpha=min_alpha,
        sorted_vocab=1,
        shrink_windows=False,
    )
    model.build_vocab(walks)
    if set(model.wv.index_to_key) != set(range(config.nodes)):
        raise ValueError("The vocabulary must contain every integer node ID.")
    return model


def sampler_arm(template: Word2Vec, counts: np.ndarray | None, domain: int) -> Word2Vec:
    model = deepcopy(template)
    if counts is not None:
        slots = np.asarray(model.wv.index_to_key)
        model.cum_table = table_from_counts(counts[slots], domain)
    assert model.wv.index_to_key == template.wv.index_to_key
    for source, copied in (
        (template.wv.vectors, model.wv.vectors),
        (template.syn1neg, model.syn1neg),
        (template.wv.expandos["count"], model.wv.expandos["count"]),
        (template.wv.expandos["sample_int"], model.wv.expandos["sample_int"]),
    ):
        np.testing.assert_array_equal(source, copied)
        assert not np.shares_memory(source, copied)
    state, copied_state = template.random.get_state(), model.random.get_state()
    assert state[0] == copied_state[0] and state[2:] == copied_state[2:]
    np.testing.assert_array_equal(state[1], copied_state[1])
    return model


def train(model: Word2Vec, walks: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    model.train(
        corpus_iterable=walks,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=model.epochs,
        start_alpha=model.alpha,
        end_alpha=model.min_alpha,
    )
    slots = np.asarray([model.wv.key_to_index[node] for node in range(len(model.wv))])
    center = model.wv.vectors[slots].astype(np.float64)
    context = model.syn1neg[slots].astype(np.float64)
    return np.linalg.norm(center, axis=1), original.balanced_left_norm(center, context)


def verify_features(
    saved: list[dict[str, Any]],
    degree: np.ndarray,
    pairs: np.ndarray,
    kl: np.ndarray,
    retained: np.ndarray,
) -> None:
    for field, values in (
        ("degree", degree),
        ("realized_positive_pairs", pairs),
        ("retained_tokens", retained),
    ):
        np.testing.assert_array_equal(values, [row[field] for row in saved])
    np.testing.assert_allclose(
        kl, [row["context_kl"] for row in saved], rtol=0, atol=1e-12
    )


def curve(degree: np.ndarray, values: np.ndarray, bins: int) -> dict[str, Any]:
    log_degree = np.log(degree)
    edges = np.unique(np.quantile(log_degree, np.linspace(0, 1, bins + 1)))
    labels = np.digitize(log_degree, edges[1:-1], right=True)
    chosen = [labels == label for label in np.unique(labels)]
    means = [float(values[mask].mean()) for mask in chosen]
    peak = int(np.argmax(means))
    return {
        "degree": [float(np.exp(log_degree[mask].mean())) for mask in chosen],
        "node_counts": [int(mask.sum()) for mask in chosen],
        "mean_norm": means,
        "peak_bin": peak + 1,
        "peak_minus_lowest": means[peak] - means[0],
        "peak_minus_highest": means[peak] - means[-1],
        "interior_peak": 0 < peak < len(means) - 1,
    }


def summarize(
    rows: list[dict[str, Any]], seeds: list[int], thresholds: list[float], bins: int
) -> dict[str, Any]:
    comparisons = []
    for seed in seeds:
        for threshold in thresholds:
            local = [
                r for r in rows if r["seed"] == seed and r["threshold"] == threshold
            ]
            degree = np.asarray([r["degree"] for r in local])
            for response in original.RESPONSES:
                values = {
                    arm: np.asarray([r[f"{arm}_{response}"] for r in local])
                    for arm in ("legacy", "retained", "raw")
                }
                comparisons.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "response": response,
                        "curves": {
                            a: curve(degree, v, bins) for a, v in values.items()
                        },
                        "paired_changes": {
                            f"{right}_minus_{left}": {
                                "mean": float((values[right] - values[left]).mean()),
                                "rmse": float(
                                    np.sqrt(
                                        np.mean((values[right] - values[left]) ** 2)
                                    )
                                ),
                                "spearman": original.safe_spearman(
                                    values[right], values[left]
                                ),
                            }
                            for left, right in (
                                ("legacy", "retained"),
                                ("retained", "raw"),
                            )
                        },
                    }
                )
    return {"comparisons": comparisons}


def plot(summary: dict[str, Any], thresholds: list[float], path: Path) -> None:
    colors = {"legacy": "#777777", "retained": "#0072B2", "raw": "#D55E00"}
    names = {
        "legacy": "Original implementation",
        "retained": "Correct retained counts",
        "raw": "Fixed original counts",
    }
    fig, axes = plt.subplots(2, len(thresholds), figsize=(12, 7), squeeze=False)
    for row, response in enumerate(original.RESPONSES):
        for column, threshold in enumerate(thresholds):
            ax = axes[row, column]
            records = [
                r
                for r in summary["comparisons"]
                if r["response"] == response and r["threshold"] == threshold
            ]
            for arm, color in colors.items():
                for record in records:
                    c = record["curves"][arm]
                    ax.plot(
                        c["degree"], c["mean_norm"], color=color, alpha=0.25, lw=0.8
                    )
                curves = [r["curves"][arm] for r in records]
                if len({len(c["degree"]) for c in curves}) == 1:
                    x = np.exp(np.mean(np.log([c["degree"] for c in curves]), axis=0))
                    y = np.mean([c["mean_norm"] for c in curves], axis=0)
                    ax.plot(x, y, "o-", color=color, label=names[arm], markersize=4)
            ax.set(
                xscale="log",
                xlabel="Degree (within-seed quantile bins)",
                ylabel=response.replace("_", " ").capitalize(),
                title=f"Threshold {threshold:g}",
            )
            ax.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.text(
        0.5,
        0.015,
        "Faint lines: individual graph seeds. Bold lines: means across seeds; "
        "no node-based error bars.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=180, metadata={"Software": "negative_sampling_control.py"})
    plt.close(fig)


def input_hashes(config: ControlConfig) -> dict[str, str]:
    paths = [
        config.protocol,
        config.calibration,
        config.design,
        config.requirements,
        Path("scripts/negative_sampling_control.py"),
        Path("scripts/prospective_norm_turnover.py"),
        Path("scripts/simulate_norm_frequency.py"),
        Path("scripts/simulate_exact_center_exposure.py"),
    ]
    paths.extend(
        p.relative_to(config.root)
        for p in (config.root / "results/prospective_validation").glob("*.json*")
    )
    paths.append(config.human_protocol)
    return {str(p): original.sha256_file(config.root / p) for p in sorted(set(paths))}


def run(config: ControlConfig, seeds: list[int] | None) -> None:
    output = (config.root / config.output).resolve()
    frozen = (config.root / config.protocol).parent.resolve()
    if output == frozen or output in frozen.parents or frozen in output.parents:
        raise ValueError("Diagnostic output must be separate from frozen results.")
    if (
        output.exists()
        and any(output.iterdir())
        and not (output / "plan.json").exists()
    ):
        raise FileExistsError("Diagnostic output is nonempty and has no matching plan.")
    original.verify_frozen_state(frozen, config.root / config.human_protocol)
    protocol = original.load_json(config.root / config.protocol)
    training = original.Config(**protocol["config"])
    seeds = protocol["calibration_seeds"] if seeds is None else seeds
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or not set(seeds) <= set(protocol["calibration_seeds"])
    ):
        raise ValueError("Choose distinct original calibration seeds.")
    thresholds = protocol["thresholds"]
    hashes = input_hashes(config)
    plan = {
        "status": "retrospective calibration diagnostic; no target validation",
        "source_sha256": hashes,
        "seeds": seeds,
        "thresholds": thresholds,
        "training": asdict(training),
        "min_alpha": config.min_alpha,
        "arms": config.arms,
        "bins": config.bins,
        "table_domain": config.table_domain,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in sorted(
                importlib.metadata.distributions(), key=lambda d: d.metadata["Name"]
            )
        },
    }
    # JSON normalization also makes tuples compare consistently on repeat runs.
    plan = json.loads(json.dumps(plan))
    output.mkdir(parents=True, exist_ok=True)
    plan_path, completion_path = output / "plan.json", output / "completion.json"
    if plan_path.exists():
        if original.load_json(plan_path) != plan:
            raise ValueError(
                "Existing diagnostic plan differs; choose a new output directory."
            )
        if completion_path.exists():
            for name, expected in original.load_json(completion_path).items():
                assert original.sha256_file(output / name) == expected, name
            print(
                "Verified existing complete diagnostic; no files changed.", flush=True
            )
            return
    original.write_json(plan_path, plan)
    saved_rows = original.load_jsonl(config.root / config.calibration)
    rows, checks = [], []
    for seed in seeds:
        graph = original.configuration_graph(training.nodes, training.attachment, seed)
        degree, walks, legacy_counts = original.prepare_walks(
            graph, training, "calibration", seed
        )
        raw_counts = token_counts(walks, training.nodes)
        for index, threshold in enumerate(thresholds):
            retained, pairs, kl, retained_counts = original.prepare_condition(
                walks, legacy_counts, threshold, index, "calibration", seed
            )
            saved = sorted(
                [
                    r
                    for r in saved_rows
                    if r["replicate"] == seed and r["threshold"] == threshold
                ],
                key=lambda r: r["node"],
            )
            verify_features(saved, degree, pairs, kl, retained_counts)
            np.testing.assert_array_equal(
                retained_counts, token_counts(retained, training.nodes)
            )
            template = make_template(
                retained,
                training,
                original.optimizer_seed("calibration", seed),
                config.min_alpha,
            )
            corpus_hash = digest_walks(retained)
            outcomes, probabilities, table_hashes = {}, {}, {}
            for arm, counts in (
                ("legacy", None),
                ("retained", retained_counts),
                ("raw", raw_counts),
            ):
                model = sampler_arm(template, counts, config.table_domain)
                probabilities[arm] = table_probabilities(model)
                table_hashes[arm] = digest_array(model.cum_table)
                if counts is not None:
                    np.testing.assert_allclose(
                        probabilities[arm],
                        counts / counts.sum(),
                        rtol=0,
                        atol=1 / config.table_domain,
                    )
                outcomes[arm] = train(model, retained)
                if not all(np.isfinite(values).all() for values in outcomes[arm]):
                    raise FloatingPointError(
                        f"Nonfinite norms in {seed}/{threshold}/{arm}"
                    )
                assert digest_walks(retained) == corpus_hash
                print(
                    f"seed={seed} threshold={threshold:g} arm={arm} complete",
                    flush=True,
                )
            checks.append(
                {
                    "seed": seed,
                    "threshold": threshold,
                    "frozen_features_match": True,
                    "walk_sha256": digest_walks(walks),
                    "retained_walk_sha256": corpus_hash,
                    "raw_tokens": int(raw_counts.sum()),
                    "legacy_count_total": int(legacy_counts.sum()),
                    "retained_tokens": int(retained_counts.sum()),
                    "positive_pairs_per_epoch": int(pairs.sum()),
                    "initial_center_sha256": digest_array(template.wv.vectors),
                    "initial_context_sha256": digest_array(template.syn1neg),
                    "vocabulary_sha256": digest_array(
                        np.asarray(template.wv.index_to_key, dtype="<i8")
                    ),
                    "table_sha256": table_hashes,
                    "legacy_vs_retained_table_tv": float(
                        np.abs(
                            probabilities["legacy"] - probabilities["retained"]
                        ).sum()
                        / 2
                    ),
                    "raw_vs_retained_table_tv": float(
                        np.abs(probabilities["raw"] - probabilities["retained"]).sum()
                        / 2
                    ),
                    "legacy_vs_saved_max_norm_difference": {
                        response: float(
                            np.max(
                                np.abs(
                                    outcomes["legacy"][i] - [r[response] for r in saved]
                                )
                            )
                        )
                        for i, response in enumerate(original.RESPONSES)
                    },
                }
            )
            for node in range(training.nodes):
                record = {
                    "seed": seed,
                    "threshold": threshold,
                    "node": node,
                    "degree": int(degree[node]),
                    "raw_tokens": int(raw_counts[node]),
                    "legacy_incidence_count": int(legacy_counts[node]),
                    "retained_tokens": int(retained_counts[node]),
                    "positive_pairs": int(pairs[node]),
                    "pair_marginal_context_kl": float(kl[node]),
                }
                for arm in config.arms:
                    record[f"{arm}_negative_table_mass"] = float(
                        probabilities[arm][node]
                    )
                    for i, response in enumerate(original.RESPONSES):
                        record[f"{arm}_{response}"] = float(outcomes[arm][i][node])
                rows.append(record)
    assert input_hashes(config) == hashes, "An input changed during the experiment."
    original.write_jsonl(output / "outcomes.jsonl", rows)
    original.write_json(output / "checks.json", checks)
    summary = summarize(rows, seeds, thresholds, config.bins)
    original.write_json(output / "summary.json", summary)
    plot(summary, thresholds, output / "negative_sampling_control.png")
    files = (
        "plan.json",
        "outcomes.jsonl",
        "checks.json",
        "summary.json",
        "negative_sampling_control.png",
    )
    original.write_json(
        completion_path, {name: original.sha256_file(output / name) for name in files}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ControlConfig.output)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(ControlConfig(output=args.output), args.seeds)


if __name__ == "__main__":
    main()
