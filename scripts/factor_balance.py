"""Replay existing corrected Gensim controls and audit native factor balance.

Run with PR35's pinned, patched runtime:
    python3 scripts/factor_balance.py [--output NEW_DIRECTORY]
Inputs: results/subsampling_origin, its source manifest and training code.
Outputs: results/factor_balance/{summary.json,table.tex}. Every graph, corpus,
initialization, token count and final factor hash must match the source run.
No experimental settings are changed; no source result is overwritten.
The audit covers the original five-epoch endpoints, not the custom mini-batch
pipeline or subsequent learning schedules. See analysis/factor_balance.md.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from simulate_norm_frequency import balanced_left_norm
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[1]
    source: Path = Path("results/subsampling_origin")
    output: Path = Path("results/factor_balance")


def measures(center: np.ndarray, context: np.ndarray) -> dict[str, Any]:
    if (
        center.ndim != 2
        or context.ndim != 2
        or center.shape[1] != context.shape[1]
        or min(*center.shape, *context.shape) == 0
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(context))
    ):
        raise ValueError("Factors must be nonempty finite matrices of equal width")
    left, right = center.T @ center, context.T @ context
    raw = np.linalg.norm(center, axis=1)
    if np.any(raw == 0):
        raise ValueError("Relative node errors require nonzero center rows")
    balanced = balanced_left_norm(center, context)
    relative = (raw - balanced) / raw
    return {
        "center_gram_frobenius": float(np.linalg.norm(left)),
        "context_gram_frobenius": float(np.linalg.norm(right)),
        "imbalance_frobenius": float(np.linalg.norm(left - right)),
        "relative_imbalance": float(
            np.linalg.norm(left - right) / np.linalg.norm(left)
        ),
        "relative_norm_l2_difference": float(
            np.linalg.norm(raw - balanced) / np.linalg.norm(raw)
        ),
        "maximum_node_relative_difference": float(np.max(np.abs(relative))),
        "node_relative_norm_difference": relative.tolist(),
        "raw_norm": raw.tolist(),
        "balanced_norm": balanced.tolist(),
    }


def output_directory(config: Config) -> Path:
    output = (config.root / config.output).resolve()
    protected = (config.root / config.source).resolve()
    if (
        output == protected
        or output in protected.parents
        or protected in output.parents
    ):
        raise ValueError("Output overlaps source data")
    return output


def write_table(report: dict[str, Any], output: Path) -> None:
    rows = [
        r"\begin{table}[htbp]",
        r"\caption{Native Gensim factor balance at the original five-epoch endpoints. "
        r"Ranges span five graph seeds at each threshold; the last column is the "
        r"largest individual-node discrepancy across those runs.}",
        r"\label{tab:factor-balance}",
        r"\centering",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Threshold & $\|D\|_F/\|U^\top U\|_F$ & "
        r"$\|r-r_{\rm bal}\|_2/\|r\|_2$ (\%) & Max. node error (\%) \\",
        r"\midrule",
    ]
    for threshold in report["config"]["thresholds"]:
        runs = [r["final"] for r in report["conditions"] if r["threshold"] == threshold]
        imbalance = [r["relative_imbalance"] for r in runs]
        norm_error = [100 * r["relative_norm_l2_difference"] for r in runs]
        maximum = max(100 * r["maximum_node_relative_difference"] for r in runs)
        label = "No filtering" if threshold == 0 else f"{threshold:g}"
        rows.append(
            f"{label} & {min(imbalance):.4f}--{max(imbalance):.4f} & "
            f"{min(norm_error):.3f}--{max(norm_error):.3f} & {maximum:.2f} " + r"\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "table.tex").write_text("\n".join(rows) + "\n")


def replay(config: Config) -> dict[str, Any]:
    import check_subsampling_origin as source

    training_config = source.Config(root=config.root, output=config.source)
    directory = config.root / config.source
    plan = json.loads((directory / "plan.json").read_text())
    actual_config = {
        k: v for k, v in asdict(training_config).items() if k not in {"root", "output"}
    }
    if json.loads(json.dumps(actual_config)) != plan["config"]:
        raise ValueError("Training configuration differs from source plan")
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Package differs from source plan: {name}")
    for filename, digest in plan["source_sha256"].items():
        if source.sha256(config.root / filename) != digest:
            raise ValueError(f"Source hash mismatch: {filename}")
    for filename, digest in json.loads(
        (directory / "completion.json").read_text()
    ).items():
        if source.sha256(directory / filename) != digest:
            raise ValueError(f"Result hash mismatch: {filename}")
    if source.blas_check() != plan["blas_check"]:
        raise ValueError("Patched BLAS validation differs from source plan")
    records = json.loads((directory / "checks.json").read_text())
    references = {(r["seed"], r["threshold"]): r for r in records}
    training = training_config.training
    expected = {
        (seed, threshold)
        for seed in range(training.seeds)
        for threshold in training_config.thresholds
    }
    if len(references) != len(records) or set(references) != expected:
        raise ValueError("Missing or duplicate source conditions")
    report: dict[str, Any] = {
        "scope": "native corrected Gensim, original five-epoch endpoints only",
        "config": plan["config"],
        "packages": plan["packages"],
        "source_sha256": plan["source_sha256"],
        "source_plan_sha256": source.sha256(directory / "plan.json"),
        "source_checks_sha256": source.sha256(directory / "checks.json"),
        "source_completion_sha256": source.sha256(directory / "completion.json"),
        "audit_script_sha256": source.sha256(Path(__file__)),
        "conditions": [],
    }
    with threadpool_limits(limits=1):
        for seed in range(training.seeds):
            graph = source.configuration_graph(
                training.nodes, training.attachment, seed
            )
            adjacency, neighbors, degree = source.graph_arrays(graph)
            walks, counts = source.sample_walks(
                neighbors,
                degree / degree.sum(),
                training.nodes * training.walks_per_node,
                training.walk_length,
                np.random.default_rng(seed + training_config.walk_seed_offset),
            )
            for threshold in training_config.thresholds:
                reference = references[seed, threshold]
                assert source.digest_array(adjacency) == reference["graph_sha256"]
                assert source.digest_walks(walks) == reference["walks_sha256"]
                assert source.digest_array(counts) == reference["counts_sha256"]
                model = source.make_model(walks, training_config, threshold, seed)
                assert source.signature(model) == reference["initial"]
                slots = np.array(
                    [model.wv.key_to_index[n] for n in range(training.nodes)]
                )
                initial = measures(
                    model.wv.vectors[slots].astype(float),
                    model.syn1neg[slots].astype(float),
                )
                effective, raw = source.train_checked(model, walks)
                assert (effective, raw) == (
                    reference["effective_tokens"],
                    reference["raw_tokens"],
                )
                center = model.wv.vectors[slots].astype(float)
                context = model.syn1neg[slots].astype(float)
                assert source.digest_array(center) == reference["input_sha256"]
                assert source.digest_array(context) == reference["output_sha256"]
                final = measures(center, context)
                report["conditions"].append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "factor_hashes_match": True,
                        "input_sha256": reference["input_sha256"],
                        "output_sha256": reference["output_sha256"],
                        "node_order": list(range(training.nodes)),
                        "initial": initial,
                        "final": final,
                    }
                )
                print(seed, threshold, final["relative_imbalance"], flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Config.output)
    args = parser.parse_args()
    config = Config(output=args.output)
    output = output_directory(config)
    report = replay(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    write_table(report, output)


if __name__ == "__main__":
    main()
