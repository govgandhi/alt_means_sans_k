# Addendum to the prospective validation protocol (issue 44)

`prospective_validation_protocol.md` is unchanged: its SHA-256 is recorded in
`results/prospective_validation/freeze_manifest.json` (`human_protocol_sha256`).
This note records how the frozen run implemented its "standard retention law".

The frozen pipeline applied the Gensim retention law outside Gensim to
**walk-visit shares**, not token shares. Its counter (`counts[walk] += 1`) added
one per walk that visited a node, however often the node repeated in that walk.
PR 35 replaced it with `np.add.at`. Evidence that the frozen features used the
walk-visit counter: the calibration control reconstructs all 15 frozen
calibration conditions exactly (positive pairs, retained tokens and KL) with
that counter (`results/negative_sampling_control/checks.json`,
`frozen_features_match`), and the issue 44 audit reproduces its walk-visit totals.

Size of the departure, from `results/issue44_token_counts/count_audit.json`
(no training; same graphs and walk seeds as the frozen run):

| corpora | tokens undercounted | visit-count slope on degree (token) | largest per-node keep difference (t = 3e-4 / 1e-3 / 3e-3) | expected retained tokens |
|---|---:|---:|---:|---:|
| calibration, 5 graphs | 10.7–10.9% | 0.996–1.001 (0.996–1.002) | 10.9% / 4.0% / 2.2% | within 0.07% |
| direct BA, 5 graphs | 10.5–10.6% | 0.964–0.972 (0.999–1.002) | 6.9% / 5.0% / 2.7% | within 0.65% |
| Holme–Kim, 5 graphs | 11.1–11.2% | 0.963–0.965 (1.001–1.003) | 7.2% / 5.6% / 4.1% | within 0.77% |
| NetScience, 5 walk seeds | 16.9–17.1% | 0.972–0.976 (0.998–1.004) | 19.6% / 19.3% / 10.5% | within 0.23% |

The undercount is nearly uniform in degree on the calibration graphs. It is not
uniform on the target graphs. At t = 3e-4 the visit law thins 574–591 of 600
calibration nodes, where the token law thins 591–596. On the direct BA and
Holme–Kim targets it thins more nodes than the token law, 591–600 against 578–590.

The predictor's inputs, positive-pair counts and context KL, were measured after
this thinning. The frozen test is therefore internally consistent. The departure
is in what the threshold t means: it is not Gensim's literal threshold on token
frequency. The frozen protocol, predictions and outcomes remain the historical
record. They cannot be regenerated with the corrected counter. The retrospective
revalidation with token counts, which also corrects the negative table and
arithmetic, is in `results/negative_sampling_revalidation/`.
