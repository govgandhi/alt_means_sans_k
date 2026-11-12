"""Raw vs balanced vs centered-score readouts in the controlled grid (summary.json)."""
from common import *
d = json.load(open(WT / "results/turnover_surface/summary.json"))
dis = []
for k, c in d["calibration_curves"].items():
    r, b, s = c["raw_norm"], c["balanced_norm"], c["centered_score_norm"]
    same_rb = (r["peak_bin"] == b["peak_bin"]) and (r["inverted_u"] == b["inverted_u"]) and (np.sign(r["spearman_degree"]) == np.sign(b["spearman_degree"]))
    same_rs = (r["peak_bin"] == s["peak_bin"]) and (r["inverted_u"] == s["inverted_u"])
    if not same_rb or not same_rs:
        dis.append((k, r["peak_bin"], b["peak_bin"], s["peak_bin"], r["inverted_u"], b["inverted_u"], s["inverted_u"], round(r["spearman_degree"], 3), round(b["spearman_degree"], 3), round(s["spearman_degree"], 3)))
print("conditions where raw/balanced/centered-score readouts differ (name, peak bins r/b/s, invU r/b/s, rho r/b/s):")
for x in dis: print("  ", x)
for k, v in d["held_out_targets"].items():
    print(k, "raw peak", v["raw_norm_prediction"]["actual_curve"]["peak_degree"], "centered-score peak", v["centered_score_norm_prediction"]["actual_curve"]["peak_degree"], "invU raw/centered", v["raw_norm_prediction"]["actual_curve"]["inverted_u"], v["centered_score_norm_prediction"]["actual_curve"]["inverted_u"])
