"""Independently rerun one target across all thresholds and compare every outcome.

Inputs: completed corrected revalidation and its frozen source/runtime plan.
Output: JSON with row counts, hashes and exact-equality result; failures exit nonzero.
Run in a fresh process: python3 scripts/check_revalidation_repeat.py
Defaults to NetScience seed301 (379 nodes), using no fitted predictor or saved model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import prospective_norm_turnover as base
import revalidate_negative_sampling as validation
from threadpoolctl import threadpool_limits


def check(config: validation.Config, family: str, replicate: int) -> dict[str, object]:
    output, training = validation.initialize(config)
    validation.verify_manifest(output, "completion.json")
    if replicate not in base.TARGET_REPLICATES[family]:
        raise ValueError("Choose an original target replicate.")
    expected = {
        base.prediction_key(row): row
        for row in base.load_jsonl(output / "outcomes.jsonl")
        if row["family"] == family and row["replicate"] == replicate
    }
    degree, walks, counts = validation.prepare(config, training, family, replicate)
    compared = 0
    for index, threshold in enumerate(base.THRESHOLDS):
        retained, pairs, kl, retained_counts = base.prepare_condition(
            walks,
            counts,
            threshold,
            index,
            family,
            replicate,
        )
        rows = validation.condition_rows(
            family,
            replicate,
            threshold,
            degree,
            counts,
            pairs,
            kl,
            retained_counts,
        )
        validation.train_pair(
            config, training, rows, walks, retained, counts, retained_counts
        )
        for row in rows:
            saved = expected[base.prediction_key(row)]
            for field, value in row.items():
                if value != saved[field]:
                    raise AssertionError(f"Independent rerun differs: {field}")
            compared += 1
    assert compared == len(expected)
    validation.verify_sources(config, output)
    return {
        "family": family,
        "replicate": replicate,
        "node_condition_rows": compared,
        "models_retrained": 6,
        "all_fields_exact": True,
        "plan_sha256": base.sha256_file(output / "plan.json"),
        "outcomes_sha256": base.sha256_file(output / "outcomes.jsonl"),
        "check_script_sha256": base.sha256_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=validation.Config.output)
    parser.add_argument(
        "--family", choices=base.TARGET_REPLICATES, default="netscience"
    )
    parser.add_argument("--replicate", type=int, default=301)
    args = parser.parse_args()
    config = validation.Config(output=args.output)
    with threadpool_limits(limits=1):
        result = check(config, args.family, args.replicate)
    destination = config.root / config.output / "repeat_check.json"
    base.write_json(destination, result)
    print(result)


if __name__ == "__main__":
    main()
