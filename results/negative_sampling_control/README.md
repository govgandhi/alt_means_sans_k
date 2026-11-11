# Negative-sampling control: complete calibration diagnostic

The binned peak location survives fixing the negative distribution in every graph/threshold condition, for both raw and balanced norms. The norm values still change. This supports persistence of the pattern within the tested calibration setup, not equality of the curves or proof of the full mechanism.

All 45 corrected-runtime models use the same real examples, initialization and training settings within each graph/threshold. Only the negative-sampling table differs between arms.

- **Legacy:** original integer-token table assignment, under corrected arithmetic.
- **Retained:** correctly assigned retained-token counts.
- **Raw:** correctly assigned original-token counts, fixed across thresholds.

The retained-versus-raw contrast answers issue #17. Legacy-versus-retained isolates the table-mapping defect. Raw and balanced norms are reported separately. An interior binned peak is descriptive, not a significance test.

## Per-graph curves

Peak bins are one-based, within each graph's occupied degree-quantile bins. Full bin degrees, node counts, mean norms and endpoint differences are in summary.json; all node records are in outcomes.jsonl.

| Seed | Threshold | Response | Occupied bins | Legacy peak | Retained peak | Raw peak | Retained peak−high-degree endpoint | Raw peak−high-degree endpoint |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.0003 | raw_norm | 5 | 1 | 1 | 1 | 0.746415 | 0.677633 |
| 0 | 0.0003 | balanced_norm | 5 | 1 | 1 | 1 | 0.749319 | 0.677989 |
| 0 | 0.001 | raw_norm | 5 | 4 | 4 | 4 | 0.474365 | 0.453143 |
| 0 | 0.001 | balanced_norm | 5 | 4 | 4 | 4 | 0.454023 | 0.433180 |
| 0 | 0.003 | raw_norm | 5 | 4 | 4 | 4 | 0.211714 | 0.207703 |
| 0 | 0.003 | balanced_norm | 5 | 4 | 4 | 4 | 0.164580 | 0.162471 |
| 1 | 0.0003 | raw_norm | 5 | 1 | 1 | 1 | 0.743809 | 0.681301 |
| 1 | 0.0003 | balanced_norm | 5 | 1 | 1 | 1 | 0.746441 | 0.681369 |
| 1 | 0.001 | raw_norm | 5 | 4 | 4 | 4 | 0.438037 | 0.404587 |
| 1 | 0.001 | balanced_norm | 5 | 4 | 4 | 4 | 0.411492 | 0.377969 |
| 1 | 0.003 | raw_norm | 5 | 4 | 4 | 4 | 0.161989 | 0.144181 |
| 1 | 0.003 | balanced_norm | 5 | 4 | 4 | 4 | 0.114199 | 0.096923 |
| 2 | 0.0003 | raw_norm | 5 | 1 | 1 | 1 | 0.752888 | 0.658874 |
| 2 | 0.0003 | balanced_norm | 5 | 1 | 1 | 1 | 0.755363 | 0.658042 |
| 2 | 0.001 | raw_norm | 5 | 4 | 4 | 4 | 0.452794 | 0.397702 |
| 2 | 0.001 | balanced_norm | 5 | 4 | 4 | 4 | 0.420735 | 0.363812 |
| 2 | 0.003 | raw_norm | 5 | 4 | 4 | 4 | 0.208668 | 0.210095 |
| 2 | 0.003 | balanced_norm | 5 | 4 | 4 | 4 | 0.147465 | 0.147834 |
| 3 | 0.0003 | raw_norm | 5 | 1 | 1 | 1 | 0.727516 | 0.685626 |
| 3 | 0.0003 | balanced_norm | 5 | 1 | 1 | 1 | 0.730917 | 0.686682 |
| 3 | 0.001 | raw_norm | 5 | 4 | 4 | 4 | 0.478325 | 0.454193 |
| 3 | 0.001 | balanced_norm | 5 | 4 | 4 | 4 | 0.451066 | 0.425576 |
| 3 | 0.003 | raw_norm | 5 | 4 | 4 | 4 | 0.227910 | 0.219873 |
| 3 | 0.003 | balanced_norm | 5 | 4 | 4 | 4 | 0.176263 | 0.167400 |
| 4 | 0.0003 | raw_norm | 5 | 1 | 1 | 1 | 0.686069 | 0.654046 |
| 4 | 0.0003 | balanced_norm | 5 | 1 | 1 | 1 | 0.688098 | 0.654026 |
| 4 | 0.001 | raw_norm | 5 | 3 | 3 | 3 | 0.296681 | 0.260096 |
| 4 | 0.001 | balanced_norm | 5 | 3 | 3 | 3 | 0.266658 | 0.228373 |
| 4 | 0.003 | raw_norm | 5 | 4 | 4 | 4 | 0.067354 | 0.059294 |
| 4 | 0.003 | balanced_norm | 5 | 5 | 4 | 4 | 0.020620 | 0.009237 |

## Runtime sensitivity

The archived unpatched-runtime legacy arm exactly reproduces every saved calibration raw norm. The corrected build passes exact scalar BLAS checks. The original arithmetic warning is absent from all corrected models.

Across all arms/nodes/conditions, the maximum absolute norm change between runtimes is 0.0647016. There are 0 changed peak-bin assignments across both responses and all arms. Full differences and per-arm warning counts are in comparison_report.json.

This is sensitivity to the corrected runtime, including recompilation; it is not an estimate attributing every numerical difference solely to the two changed declarations.

## Evidence boundary

This comparison is restricted to the original five calibration graphs and legacy positive-example filtering. It does not repair repeated-visit counting in that filtering, revalidate held-out predictions, or establish that exposure and context specificity fully explain norms. The original frozen results and protocol are unchanged.

Reproduce training with scripts/negative_sampling_control.py after the documented build; reproduce this report with scripts/report_negative_sampling_control.py. See analysis/negative_sampling_control_protocol.md for the recorded design and environment.

A fresh seed-0 rerun reproduced all 1,800 node-condition records exactly, including all three arms' raw and balanced norms.
