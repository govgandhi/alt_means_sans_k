"""Check supplement claims against results/prospective_baselines/summary.json."""
from common import *
S = json.load(open(WT / "results/prospective_baselines/summary.json"))
print("status:", S["status"]); print("datasets:", list(S["datasets"]))
for ds, comps in S["datasets"].items():
    print("\n==", ds, list(comps))
    for resp in ("raw_norm", "balanced_norm"):
        a = {(r["family"], r["threshold"], r["replicate"]): r["spearman"] for r in comps["pairs_kl"][f"{resp}_replicates"]}
        b = {(r["family"], r["threshold"], r["replicate"]): r["spearman"] for r in comps["degree_threshold"][f"{resp}_replicates"]}
        wins = sum(a[k] > b[k] for k in a); diffs = np.array([a[k] - b[k] for k in a])
        print(f"  {resp}: pairs_kl beats degree_threshold in {wins}/{len(a)} replicate groups; diff median {np.median(diffs):.3f} min {diffs.min():.3f}")
        # per family median of replicate-level differences
        for fam in FAMILIES:
            dd = [a[k] - b[k] for k in a if k[0] == fam]
            print(f"     {fam}: replicate-level diff median {np.median(dd):.3f} [{min(dd):.3f},{max(dd):.3f}]")
    for kind, s in comps.items():
        c = s["criteria"]; print(f"  {kind:18s} med_err {c['median_absolute_peak_bin_error']} frac {c['fraction_peaks_within_one_bin']:.3f} pooled {c['pooled_spearman']:.3f} all_pass {s['all_primary_criteria_passed']} within {s['raw_norm_within_curve']}")
