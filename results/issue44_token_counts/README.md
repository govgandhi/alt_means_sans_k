# Issue 44: token counts versus walk-visit counts

Retrospective. The historical `results/gensim_threshold_sweep/` and
`results/norm_frequency_simulation/` outputs are unchanged. The reruns
here use the unchanged experiment scripts on `main`, which count every
token occurrence (PR 35) and train with the corrected negative table and
checked BLAS build.
Differences from the historical numbers therefore combine all PR 35
corrections. The count audit isolates what the counter alone changed.

## Count audit (no training)

Walk-visit totals undercount tokens in every corpus. Shares are normalized,
so the retention law sees only the node-to-node variation in the undercount.

| corpus | seed | tokens | walk visits | visits/tokens | node ratio min | node ratio max | slope token | slope visit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| norm_frequency/standard | 0 | 900000 | 802813 | 0.8920 | 0.7650 | 0.9816 | 0.9991 | 0.9983 |
| norm_frequency/uniform_starts | 0 | 900000 | 801216 | 0.8902 | 0.7441 | 0.9842 | 0.7767 | 0.7945 |
| norm_frequency/regular_control | 0 | 900000 | 805227 | 0.8947 | 0.8361 | 0.9307 | None | None |
| norm_frequency/standard | 1 | 900000 | 802656 | 0.8918 | 0.7041 | 0.9830 | 0.9989 | 0.9993 |
| norm_frequency/uniform_starts | 1 | 900000 | 800371 | 0.8893 | 0.6425 | 0.9822 | 0.7711 | 0.7918 |
| norm_frequency/regular_control | 1 | 900000 | 805168 | 0.8946 | 0.8409 | 0.9277 | None | None |
| norm_frequency/standard | 2 | 900000 | 802082 | 0.8912 | 0.7957 | 0.9793 | 1.0009 | 0.9983 |
| norm_frequency/uniform_starts | 2 | 900000 | 800906 | 0.8899 | 0.7845 | 0.9789 | 0.7873 | 0.8031 |
| norm_frequency/regular_control | 2 | 900000 | 805402 | 0.8949 | 0.8398 | 0.9284 | None | None |
| norm_frequency/standard | 3 | 900000 | 802041 | 0.8912 | 0.7811 | 0.9814 | 0.9960 | 0.9958 |
| norm_frequency/uniform_starts | 3 | 900000 | 800795 | 0.8898 | 0.7648 | 0.9799 | 0.7827 | 0.8009 |
| norm_frequency/regular_control | 3 | 900000 | 805120 | 0.8946 | 0.8392 | 0.9300 | None | None |
| norm_frequency/standard | 4 | 900000 | 803317 | 0.8926 | 0.7745 | 0.9917 | 1.0019 | 1.0006 |
| norm_frequency/uniform_starts | 4 | 900000 | 801126 | 0.8901 | 0.7578 | 0.9851 | 0.7794 | 0.7964 |
| norm_frequency/regular_control | 4 | 900000 | 805240 | 0.8947 | 0.8344 | 0.9281 | None | None |
| prospective/barabasi_albert | 101 | 900000 | 805170 | 0.8946 | 0.8264 | 0.9895 | 1.0011 | 0.9683 |
| prospective/barabasi_albert | 102 | 900000 | 804909 | 0.8943 | 0.8327 | 0.9914 | 1.0017 | 0.9681 |
| prospective/barabasi_albert | 103 | 900000 | 804521 | 0.8939 | 0.8406 | 0.9833 | 0.9999 | 0.9713 |
| prospective/barabasi_albert | 104 | 900000 | 804525 | 0.8939 | 0.8295 | 0.9847 | 1.0021 | 0.9721 |
| prospective/barabasi_albert | 105 | 900000 | 804786 | 0.8942 | 0.8314 | 0.9932 | 0.9994 | 0.9638 |
| prospective/powerlaw_cluster | 201 | 900000 | 800067 | 0.8890 | 0.8210 | 0.9831 | 1.0028 | 0.9646 |
| prospective/powerlaw_cluster | 202 | 900000 | 799663 | 0.8885 | 0.8200 | 0.9907 | 1.0020 | 0.9630 |
| prospective/powerlaw_cluster | 203 | 900000 | 799124 | 0.8879 | 0.8157 | 0.9868 | 1.0005 | 0.9627 |
| prospective/powerlaw_cluster | 204 | 900000 | 799763 | 0.8886 | 0.8195 | 0.9861 | 1.0013 | 0.9651 |
| prospective/powerlaw_cluster | 205 | 900000 | 799910 | 0.8888 | 0.8278 | 0.9888 | 1.0013 | 0.9636 |
| prospective/netscience | 301 | 568500 | 471919 | 0.8301 | 0.6453 | 0.9787 | 0.9996 | 0.9720 |
| prospective/netscience | 302 | 568500 | 471898 | 0.8301 | 0.6427 | 0.9863 | 0.9991 | 0.9728 |
| prospective/netscience | 303 | 568500 | 472269 | 0.8307 | 0.6348 | 0.9968 | 1.0009 | 0.9742 |
| prospective/netscience | 304 | 568500 | 471322 | 0.8291 | 0.6487 | 0.9904 | 0.9978 | 0.9715 |
| prospective/netscience | 305 | 568500 | 471908 | 0.8301 | 0.6321 | 0.9927 | 1.0042 | 0.9757 |

Retention law at each threshold: nodes downsampled (keep < 1) under each
count, the largest relative keep-probability difference, and expected
retained tokens per epoch when each law is applied to the true tokens.

| corpus | seed | t | downsampled: token / visit | max abs rel keep diff | retained: token law | retained: visit law |
|---:|---:|---:|---:|---:|---:|---:|
| norm_frequency/standard | 0 | t3e-4 | 595 / 574 | 0.1087 | 519045 | 518918 |
| norm_frequency/standard | 0 | t1e-3 | 75 / 76 | 0.0403 | 775572 | 775386 |
| norm_frequency/standard | 0 | t3e-3 | 10 / 10 | 0.0127 | 869058 | 869405 |
| norm_frequency/standard | 1 | t3e-4 | 592 / 577 | 0.0861 | 520154 | 519917 |
| norm_frequency/standard | 1 | t1e-3 | 76 / 80 | 0.0363 | 778877 | 778770 |
| norm_frequency/standard | 1 | t3e-3 | 8 / 8 | 0.0219 | 876173 | 876628 |
| norm_frequency/standard | 2 | t3e-4 | 592 / 583 | 0.0792 | 519393 | 519238 |
| norm_frequency/standard | 2 | t1e-3 | 71 / 70 | 0.0392 | 773853 | 774270 |
| norm_frequency/standard | 2 | t3e-3 | 10 / 11 | 0.0165 | 870568 | 870841 |
| norm_frequency/standard | 3 | t3e-4 | 596 / 591 | 0.0877 | 520905 | 521024 |
| norm_frequency/standard | 3 | t1e-3 | 66 / 66 | 0.0332 | 779870 | 779953 |
| norm_frequency/standard | 3 | t3e-3 | 8 / 8 | 0.0072 | 875658 | 875678 |
| norm_frequency/standard | 4 | t3e-4 | 591 / 576 | 0.0831 | 520079 | 519858 |
| norm_frequency/standard | 4 | t1e-3 | 65 / 66 | 0.0316 | 775553 | 775939 |
| norm_frequency/standard | 4 | t3e-3 | 12 / 11 | 0.0212 | 876169 | 876740 |
| prospective/barabasi_albert | 101 | t3e-4 | 585 / 591 | 0.0666 | 519176 | 516149 |
| prospective/barabasi_albert | 101 | t1e-3 | 68 / 67 | 0.0493 | 774507 | 776935 |
| prospective/barabasi_albert | 101 | t3e-3 | 13 / 13 | 0.0259 | 867830 | 868966 |
| prospective/barabasi_albert | 102 | t3e-4 | 581 / 595 | 0.0660 | 518015 | 514696 |
| prospective/barabasi_albert | 102 | t1e-3 | 63 / 63 | 0.0410 | 767003 | 769422 |
| prospective/barabasi_albert | 102 | t3e-3 | 10 / 10 | 0.0218 | 861040 | 862148 |
| prospective/barabasi_albert | 103 | t3e-4 | 590 / 597 | 0.0616 | 520768 | 518385 |
| prospective/barabasi_albert | 103 | t1e-3 | 76 / 76 | 0.0423 | 782115 | 784353 |
| prospective/barabasi_albert | 103 | t3e-3 | 9 / 8 | 0.0192 | 874256 | 874910 |
| prospective/barabasi_albert | 104 | t3e-4 | 587 / 595 | 0.0643 | 519845 | 517272 |
| prospective/barabasi_albert | 104 | t1e-3 | 67 / 67 | 0.0497 | 777683 | 780179 |
| prospective/barabasi_albert | 104 | t3e-3 | 9 / 9 | 0.0266 | 870311 | 871058 |
| prospective/barabasi_albert | 105 | t3e-4 | 588 / 596 | 0.0695 | 518892 | 515513 |
| prospective/barabasi_albert | 105 | t1e-3 | 81 / 80 | 0.0480 | 771534 | 774583 |
| prospective/barabasi_albert | 105 | t3e-3 | 9 / 9 | 0.0254 | 875103 | 876199 |
| prospective/powerlaw_cluster | 201 | t3e-4 | 578 / 593 | 0.0664 | 518358 | 514663 |
| prospective/powerlaw_cluster | 201 | t1e-3 | 66 / 68 | 0.0559 | 767811 | 771050 |
| prospective/powerlaw_cluster | 201 | t3e-3 | 9 / 7 | 0.0268 | 869854 | 871013 |
| prospective/powerlaw_cluster | 202 | t3e-4 | 586 / 594 | 0.0718 | 517408 | 513486 |
| prospective/powerlaw_cluster | 202 | t1e-3 | 68 / 68 | 0.0422 | 765333 | 768208 |
| prospective/powerlaw_cluster | 202 | t3e-3 | 10 / 10 | 0.0413 | 859638 | 861279 |
| prospective/powerlaw_cluster | 203 | t3e-4 | 589 / 600 | 0.0697 | 517677 | 513707 |
| prospective/powerlaw_cluster | 203 | t1e-3 | 71 / 72 | 0.0456 | 764274 | 767345 |
| prospective/powerlaw_cluster | 203 | t3e-3 | 10 / 10 | 0.0336 | 864344 | 866135 |
| prospective/powerlaw_cluster | 204 | t3e-4 | 585 / 594 | 0.0690 | 517312 | 513374 |
| prospective/powerlaw_cluster | 204 | t1e-3 | 60 / 60 | 0.0448 | 760509 | 763335 |
| prospective/powerlaw_cluster | 204 | t3e-3 | 14 / 14 | 0.0393 | 861167 | 863393 |
| prospective/powerlaw_cluster | 205 | t3e-4 | 590 / 596 | 0.0707 | 517586 | 513639 |
| prospective/powerlaw_cluster | 205 | t1e-3 | 64 / 63 | 0.0382 | 762053 | 765131 |
| prospective/powerlaw_cluster | 205 | t3e-3 | 11 / 11 | 0.0302 | 861215 | 863005 |
| prospective/netscience | 301 | t3e-4 | 352 / 352 | 0.1784 | 242689 | 242811 |
| prospective/netscience | 301 | t1e-3 | 140 / 135 | 0.1640 | 449931 | 450514 |
| prospective/netscience | 301 | t3e-3 | 11 / 9 | 0.1004 | 553283 | 554558 |
| prospective/netscience | 302 | t3e-4 | 352 / 352 | 0.1820 | 242631 | 242817 |
| prospective/netscience | 302 | t1e-3 | 142 / 134 | 0.1336 | 449970 | 450225 |
| prospective/netscience | 302 | t3e-3 | 12 / 10 | 0.1046 | 553224 | 554485 |
| prospective/netscience | 303 | t3e-4 | 352 / 350 | 0.1924 | 242651 | 242815 |
| prospective/netscience | 303 | t1e-3 | 144 / 135 | 0.1933 | 450202 | 450586 |
| prospective/netscience | 303 | t3e-3 | 11 / 10 | 0.1023 | 553629 | 554714 |
| prospective/netscience | 304 | t3e-4 | 352 / 352 | 0.1721 | 242711 | 242882 |
| prospective/netscience | 304 | t1e-3 | 142 / 135 | 0.1644 | 450540 | 450943 |
| prospective/netscience | 304 | t3e-3 | 11 / 10 | 0.0977 | 553534 | 554597 |
| prospective/netscience | 305 | t3e-4 | 352 / 352 | 0.1957 | 242545 | 242685 |
| prospective/netscience | 305 | t1e-3 | 143 / 135 | 0.1674 | 449396 | 449854 |
| prospective/netscience | 305 | t3e-3 | 11 / 10 | 0.1051 | 553231 | 554490 |

`norm_frequency/standard` is also the threshold-sweep, subsampling-origin and
prospective calibration corpus. The audit reproduces the frozen calibration
control's walk-visit totals for all 15 calibration conditions
(`frozen_calibration_check` in count_audit.json).

Visit/token ratio by pooled degree bin, calibration graph seed 0:

| bin | nodes | geometric mean degree | tokens | walk visits | visits/tokens |
|---:|---:|---:|---:|---:|---:|
| 0 | 258 | 2.9779 | 197235 | 176835 | 0.8966 |
| 1 | 96 | 4.0000 | 98361 | 86746 | 0.8819 |
| 2 | 77 | 5.0000 | 98244 | 87847 | 0.8942 |
| 3 | 39 | 6.0000 | 59434 | 53212 | 0.8953 |
| 4 | 58 | 7.8157 | 116864 | 104196 | 0.8916 |
| 5 | 72 | 15.5543 | 329862 | 293977 | 0.8912 |

## Gensim threshold sweep

Training uses Gensim's own (correct) counts for subsampling and negatives,
so `counts` reaches the trained norms only through the budget-matched epoch
rule. Epochs and expected retained-center totals (mean over five graphs):

| schedule | t | original epochs | visit-count epochs | corrected epochs | original total | audit visit total | corrected total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_epochs | 0.0003 | 5 | 5 | 5 | 2317913 | 2317913 | 2599576 |
| fixed_epochs | 0.0010 | 5 | 5 | 5 | 3466170 | 3466170 | 3883724 |
| fixed_epochs | 0.0030 | 5 | 5 | 5 | 3897394 | 3897394 | 4367626 |
| budget_matched | 0.0003 | 8 | 8 | 8 | 3708661 | 3708661 | 4159321 |
| budget_matched | 0.0010 | 6 | 6 | 6 | 4159404 | 4159404 | 4660469 |
| budget_matched | 0.0030 | 5 | 5 | 5 | 3897394 | 3897394 | 4367626 |

Pooled curves, historical versus corrected (raw and balanced norms).
Peak bins are zero-based indices of the occupied pooled log-degree bins.
The last column is the largest change among the three saved bin means.

| schedule | t | response | peak bin | peak degree | lowest bin | peak | highest bin | peak − lowest | highest − peak | inverted-U | max abs change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_epochs | 0.0003 | raw_norm | 0 / 0 | 2.985 / 2.985 | 4.261 / 4.268 | 4.261 / 4.268 | 3.513 / 3.510 | 0.000 / 0.000 | -0.748 / -0.759 | False / False | 0.0072 |
| fixed_epochs | 0.0003 | balanced_norm | 0 / 0 | 2.985 / 2.985 | 4.283 / 4.291 | 4.283 / 4.291 | 3.536 / 3.532 | 0.000 / 0.000 | -0.748 / -0.759 | False / False | 0.0076 |
| fixed_epochs | 0.0010 | raw_norm | 3 / 3 | 6.000 / 6.000 | 4.758 / 4.759 | 5.130 / 5.134 | 4.593 / 4.607 | 0.372 / 0.375 | -0.537 / -0.527 | True / True | 0.0139 |
| fixed_epochs | 0.0010 | balanced_norm | 3 / 3 | 6.000 / 6.000 | 4.775 / 4.776 | 5.153 / 5.156 | 4.648 / 4.661 | 0.378 / 0.380 | -0.504 / -0.495 | True / True | 0.0128 |
| fixed_epochs | 0.0030 | raw_norm | 4 / 4 | 7.731 / 7.731 | 4.798 / 4.810 | 5.367 / 5.370 | 5.128 / 5.113 | 0.569 / 0.560 | -0.238 / -0.257 | True / True | 0.0158 |
| fixed_epochs | 0.0030 | balanced_norm | 4 / 4 | 7.731 / 7.731 | 4.810 / 4.820 | 5.383 / 5.384 | 5.205 / 5.187 | 0.573 / 0.565 | -0.178 / -0.198 | True / True | 0.0185 |
| budget_matched | 0.0003 | raw_norm | 0 / 0 | 2.985 / 2.985 | 4.426 / 4.435 | 4.426 / 4.435 | 3.571 / 3.590 | 0.000 / 0.000 | -0.854 / -0.845 | False / False | 0.0184 |
| budget_matched | 0.0003 | balanced_norm | 0 / 0 | 2.985 / 2.985 | 4.457 / 4.468 | 4.457 / 4.468 | 3.606 / 3.625 | 0.000 / 0.000 | -0.851 / -0.843 | False / False | 0.0190 |
| budget_matched | 0.0010 | raw_norm | 3 / 3 | 6.000 / 6.000 | 4.863 / 4.864 | 5.213 / 5.216 | 4.609 / 4.623 | 0.350 / 0.352 | -0.603 / -0.593 | True / True | 0.0135 |
| budget_matched | 0.0010 | balanced_norm | 3 / 3 | 6.000 / 6.000 | 4.882 / 4.883 | 5.239 / 5.242 | 4.674 / 4.687 | 0.357 / 0.359 | -0.564 / -0.554 | True / True | 0.0127 |
| budget_matched | 0.0030 | raw_norm | 4 / 4 | 7.731 / 7.731 | 4.798 / 4.810 | 5.367 / 5.370 | 5.128 / 5.113 | 0.569 / 0.560 | -0.238 / -0.257 | True / True | 0.0158 |
| budget_matched | 0.0030 | balanced_norm | 4 / 4 | 7.731 / 7.731 | 4.810 / 4.820 | 5.383 / 5.384 | 5.205 / 5.187 | 0.573 / 0.565 | -0.178 / -0.198 | True / True | 0.0185 |

Every corrected pooled curve (raw norm; degree means are common to all
conditions: 2.985, 4.000, 5.000, 6.000, 7.731, 16.044; node rows per bin: 1221, 564, 360, 215, 295, 345).
The historical node rows were not committed, so only the historical summary
points above are available for the historical curves.

| schedule | t | bin 0 | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_epochs | 0.0003 | 4.2681 | 4.1852 | 4.0624 | 3.9670 | 3.7678 | 3.5095 |
| fixed_epochs | 0.0010 | 4.7594 | 4.9802 | 5.0786 | 5.1340 | 5.1058 | 4.6068 |
| fixed_epochs | 0.0030 | 4.8095 | 5.0695 | 5.2119 | 5.3185 | 5.3696 | 5.1126 |
| budget_matched | 0.0003 | 4.4347 | 4.2962 | 4.1394 | 4.0304 | 3.8138 | 3.5897 |
| budget_matched | 0.0010 | 4.8643 | 5.0816 | 5.1741 | 5.2158 | 5.1678 | 4.6229 |
| budget_matched | 0.0030 | 4.8095 | 5.0695 | 5.2119 | 5.3185 | 5.3696 | 5.1126 |

Exploratory controlled prediction versus observed raw-norm peak bins
(prediction mapped to the nearest occupied bin;
error = predicted − observed):

| t | schedule | predicted degree | predicted bin | observed bin: original / corrected | bin error: original / corrected |
|---:|---:|---:|---:|---:|---:|
| 0.0003 | fixed_epochs | 7.7336 | 4 | 0 / 0 | 4 / 4 |
| 0.0003 | budget_matched | 7.7336 | 4 | 0 / 0 | 4 / 4 |
| 0.0010 | fixed_epochs | 16.0824 | 5 | 3 / 3 | 2 / 2 |
| 0.0010 | budget_matched | 16.0824 | 5 | 3 / 3 | 2 / 2 |
| 0.0030 | fixed_epochs | 16.0824 | 5 | 4 / 4 | 1 / 1 |
| 0.0030 | budget_matched | 16.0824 | 5 | 4 / 4 | 1 / 1 |

Largest absolute node-norm difference between the corrected fixed-epoch
sweep and `results/subsampling_origin/outcomes.csv` (same graphs, corpora and
seeds; built independently):

| t | raw_norm | balanced_norm |
|---:|---:|---:|
| t3e-4 | 0.0000 | 0.0000 |
| t1e-3 | 0.0000 | 0.0000 |
| t3e-3 | 0.0000 | 0.0000 |

## Norm-frequency simulation

Training does not use `counts`; the counter changes the frequency axis and
the gauge direction. Raw norm on frequency: historical / corrected /
corrected norms on the walk-visit axis. Corrected versus visit axis
isolates the counter (same norms); historical versus visit axis isolates
the training corrections (same axis). The gauge norms need the trained
factors, which are not saved, so their changes below are not separated.

| condition | metric | historical | corrected | corrected, visit axis |
|---:|---:|---:|---:|---:|
| standard | spearman | -0.0324 | 0.0920 | -0.0317 |
| standard | peak_bin | 5 | 5 | 5 |
| standard | bins | 8 | 8 | 8 |
| standard | binned_peak_z | 0.3930 | 0.5490 | 0.3973 |
| standard | last_bin_minus_peak | -0.4711 | -0.4699 | -0.4565 |
| standard | inverted_u_by_bins | True | True | True |
| standard | spearman_norm_context_kl | 0.5346 | 0.5347 |  |
| uniform_starts | spearman | 0.2047 | 0.2861 | 0.2038 |
| uniform_starts | peak_bin | 6 | 6 | 6 |
| uniform_starts | bins | 8 | 8 | 8 |
| uniform_starts | binned_peak_z | 0.6911 | 0.7215 | 0.6919 |
| uniform_starts | last_bin_minus_peak | -0.3861 | -0.3978 | -0.3866 |
| uniform_starts | inverted_u_by_bins | True | True | True |
| uniform_starts | spearman_norm_context_kl | 0.5307 | 0.5284 |  |
| no_subsampling | spearman | 0.6742 | 0.7192 | 0.6541 |
| no_subsampling | peak_bin | 6 | 6 | 6 |
| no_subsampling | bins | 8 | 8 | 8 |
| no_subsampling | binned_peak_z | 1.2073 | 1.0957 | 1.1389 |
| no_subsampling | last_bin_minus_peak | -0.1545 | -0.1892 | -0.1858 |
| no_subsampling | inverted_u_by_bins | True | True | True |
| no_subsampling | spearman_norm_context_kl | -0.2638 | -0.2367 |  |
| regular_control | spearman | 0.2416 | 0.0885 | 0.0904 |
| regular_control | peak_bin | 6 | 3 | 4 |
| regular_control | bins | 8 | 8 | 8 |
| regular_control | binned_peak_z | 0.9325 | 0.3974 | 0.3938 |
| regular_control | last_bin_minus_peak | -0.0090 | -0.0470 | -0.0478 |
| regular_control | inverted_u_by_bins | False | False | False |
| regular_control | spearman_norm_context_kl | None | None |  |

Gauge and graph checks (historical / corrected):

| quantity | historical | corrected |
|---:|---:|---:|
| gauge raw_norm_frequency_spearman | -0.0324 | 0.0920 |
| gauge shrunk_direction_norm_frequency_spearman | -0.0950 | 0.0346 |
| gauge stretched_direction_norm_frequency_spearman | 0.3364 | 0.2808 |
| gauge balanced_norm_frequency_spearman | -0.0151 | 0.1080 |
| population pooled_kl_slope_on_log_degree | -0.9941 | -0.9941 |
| population spearman_kl_degree | -0.7074 | -0.7074 |

Corrected binned raw-norm curves on each frequency axis (eight quantile
bins of log frequency; standard errors are node-based, as in the
historical plot routine):

| condition | axis | bin mean frequency | bin mean raw norm |
|---:|---:|---:|---:|
| standard | token_axis | 723, 765, 793, 950, 1094, 1351, 1824, 3929 | 4.719, 4.748, 4.782, 4.930, 5.017, 5.114, 5.110, 4.644 |
| standard | walk_visit_axis | 634, 684, 717, 846, 979, 1206, 1632, 3504 | 5.050, 4.829, 4.521, 4.983, 4.905, 5.073, 5.071, 4.617 |
| uniform_starts | token_axis | 828, 926, 1004, 1097, 1214, 1407, 1751, 3370 | 4.383, 4.803, 4.990, 5.090, 5.125, 5.242, 5.249, 4.851 |
| uniform_starts | walk_visit_axis | 763, 821, 866, 956, 1064, 1239, 1563, 3042 | 4.509, 4.836, 5.068, 4.964, 5.129, 5.200, 5.210, 4.824 |
| no_subsampling | token_axis | 723, 765, 793, 950, 1094, 1351, 1824, 3929 | 4.854, 4.890, 4.899, 5.098, 5.207, 5.391, 5.526, 5.337 |
| no_subsampling | walk_visit_axis | 634, 684, 717, 846, 979, 1206, 1632, 3504 | 4.960, 4.906, 4.822, 5.104, 5.183, 5.375, 5.517, 5.331 |
| regular_control | token_axis | 1149, 1237, 1299, 1362, 1433, 1527, 1685, 2247 | 5.392, 5.409, 5.428, 5.438, 5.435, 5.435, 5.433, 5.391 |
| regular_control | walk_visit_axis | 1045, 1121, 1174, 1226, 1288, 1368, 1499, 1964 | 5.392, 5.406, 5.430, 5.435, 5.438, 5.435, 5.433, 5.390 |

Largest absolute node-norm difference from `results/subsampling_origin/`:

| condition | raw_norm | balanced_norm |
|---:|---:|---:|
| standard | 0.0000 | 0.0000 |
| no_subsampling | 0.0000 | 0.0000 |

## Reproduce

Set up the pinned runtime from
`analysis/negative_sampling_revalidation_protocol.md` (Python 3.13.12,
`analysis/negative_sampling_control_requirements.txt`,
`python3 scripts/build_control_gensim.py --install`), then run
`python3 scripts/issue44_token_counts.py --output /fresh/path`. `plan.json`
records the environment and source hashes; `completion.json` hashes the
rerun outputs (PDFs excluded). The `run` stage verifies an existing output
against both before doing anything.
