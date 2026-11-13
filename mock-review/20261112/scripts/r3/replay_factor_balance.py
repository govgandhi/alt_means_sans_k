"""R3 check: replay the 20 corrected Gensim controls on a reader's machine.

scripts/factor_balance.py refuses to run here because blas_check() includes the
machine-specific extension SHA-256. This wrapper keeps every other gate (graph,
walk, count and initialization digests, token counts) but records rather than
asserts the final factor digests, so we can see whether a reader who follows the
README gets bit-identical factors and the same Table S3 numbers.

Usage: python replay_factor_balance.py <repo_root> <out.json>
Writes nothing into the repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def main(root: Path, out: Path) -> None:
    sys.path.insert(0, str(root / "scripts"))
    import check_subsampling_origin as source
    from factor_balance import measures

    directory = root / "results/subsampling_origin"
    plan = json.loads((directory / "plan.json").read_text())
    actual = source.blas_check()
    recorded = plan["blas_check"]
    numeric_ok = (
        actual["wrapper"] == recorded["wrapper"]
        and actual["dot_products"] == recorded["dot_products"]
        and actual["passed"]
    )
    print("numeric BLAS checks equal:", numeric_ok)
    print(
        "extension sha recorded/actual:",
        recorded["extension_sha256"][:12],
        actual["extension_sha256"][:12],
    )
    training_config = source.Config(
        root=root, output=Path("results/subsampling_origin")
    )
    training = training_config.training
    references = {
        (r["seed"], r["threshold"]): r
        for r in json.loads((directory / "checks.json").read_text())
    }
    committed = {
        (c["seed"], c["threshold"]): c
        for c in json.loads((root / "results/factor_balance/summary.json").read_text())[
            "conditions"
        ]
    }
    rows = []
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
                ref = references[seed, threshold]
                gates = {
                    "graph": source.digest_array(adjacency) == ref["graph_sha256"],
                    "walks": source.digest_walks(walks) == ref["walks_sha256"],
                    "counts": source.digest_array(counts) == ref["counts_sha256"],
                }
                model = source.make_model(walks, training_config, threshold, seed)
                gates["initial_signature"] = source.signature(model) == ref["initial"]
                slots = np.array(
                    [model.wv.key_to_index[n] for n in range(training.nodes)]
                )
                effective, raw = source.train_checked(model, walks)
                gates["tokens"] = (effective, raw) == (
                    ref["effective_tokens"],
                    ref["raw_tokens"],
                )
                center = model.wv.vectors[slots].astype(float)
                context = model.syn1neg[slots].astype(float)
                gates["center_hash"] = (
                    source.digest_array(center) == ref["input_sha256"]
                )
                gates["context_hash"] = (
                    source.digest_array(context) == ref["output_sha256"]
                )
                final = measures(center, context)
                old = committed[seed, threshold]["final"]
                raw_old = np.array(old["raw_norm"])
                raw_new = np.array(final["raw_norm"])
                row = {
                    "seed": seed,
                    "threshold": threshold,
                    "gates": gates,
                    "relative_imbalance": [
                        old["relative_imbalance"],
                        final["relative_imbalance"],
                    ],
                    "relative_norm_l2_difference": [
                        old["relative_norm_l2_difference"],
                        final["relative_norm_l2_difference"],
                    ],
                    "max_node": [
                        old["maximum_node_relative_difference"],
                        final["maximum_node_relative_difference"],
                    ],
                    "max_abs_raw_norm_change_vs_committed": float(
                        np.max(np.abs(raw_old - raw_new))
                    ),
                }
                rows.append(row)
                print(
                    seed,
                    threshold,
                    {k: v for k, v in gates.items() if not v} or "all gates pass",
                    f"max|dnorm|={row['max_abs_raw_norm_change_vs_committed']:.3g}",
                    flush=True,
                )
    out.write_text(
        json.dumps({"numeric_blas_ok": numeric_ok, "rows": rows}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
