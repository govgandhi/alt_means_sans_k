"""Controlled exposure/context grid: reproduce reported numbers; absolute vs relative endpoint contrasts."""
from common import *
d = json.load(open(WT / "results/turnover_surface/summary.json"))
cc = d["calibration_curves"]
print("cond          rho_deg  lo     hi     hi-lo   (hi-lo)/lo  peakbin invU")
for k, c in cc.items():
    r = c["raw_norm"]; lo, hi = r["lowest_degree_bin"], r["highest_degree_bin"]
    print(f"{k:22s} {r['spearman_degree']:+.3f} {lo:.3f} {hi:.3f} {hi-lo:+.3f} {100*(hi-lo)/lo:+6.1f}%  {r['peak_bin']}/{r['bins']} {r['inverted_u']}")
b = cc["beta_033_lambda_000"]["raw_norm"]
print("\nbeta=1/3: peak %.4f, lowest %.4f, highest %.4f -> peak-lowest %.4f (claimed 0.093), peak-highest %.4f (claimed 0.361); rel. rise %.1f%%" % (b["peak"], b["lowest_degree_bin"], b["highest_degree_bin"], b["peak_minus_lowest"], -b["highest_minus_peak"], 100 * b["peak_minus_lowest"] / b["lowest_degree_bin"]))
h = d["held_out_targets"]["holdout_t3e-4"]["raw_norm_prediction"]
print("held-out t=3e-4 declared quadratic: R2 %.3f, Spearman(actual,pred) %.3f; peak obs %.2f vs pred %.2f" % (h["r_squared"], h["spearman_actual_predicted"], h["actual_curve"]["peak_degree"], h["predicted_curve"]["peak_degree"]))
print("held-out t=3e-4 observed: peak-lowest %.3f (%.1f%%), peak-highest %.3f, node Spearman with degree %.3f" % (h["actual_curve"]["peak_minus_lowest"], 100*h["actual_curve"]["peak_minus_lowest"]/h["actual_curve"]["lowest_degree_bin"], -h["actual_curve"]["highest_minus_peak"], h["actual_curve"]["spearman_degree"]))
rs = d["response_surface"]["raw_norm"]
print("fitted surface: fraction of calibration rows with positive d norm/d log exposure = %.3f; with positive d norm/d KL = %.4f" % (rs["fraction_positive_exposure_derivative"], rs["fraction_positive_specificity_derivative"]))
