# Prior work on the norm–frequency curve, and what we can claim

Literature check of 2026-11-23 for the norm tests (fresh corpus, matched-τ step sizes, window matching, specificity, animation) and the PRE manuscript.
The test scripts named below live in `scripts/norm_tests/` and arrive in follow-up PRs; numbers quoted here come from those runs.
Sources were checked against arXiv abstract pages or API records, ACL Anthology, Crossref, or the PDFs; section, equation, and figure numbers come from the PDFs.
Searches used OpenAlex (keyword and `cites:` filters on Schakel & Wilson, Wilson & Schakel, Oyama et al., Murray et al., Zipfian whitening, Arora et al., Mimno & Thompson), the arXiv API, and web search; Semantic Scholar rate-limited most queries.

## Bottom line

- The rising branch (rare items are undertrained) is well established, and the reversal of the norm–degree ordering during training has been shown twice: Zhang, Lu & Shai (2018) on BlogCatalog graphs and Oyama et al. (2023, App. G) on text.
- Norm tracks context specificity (KL) at fixed frequency: Wilson & Schakel (2015) showed this causally in text, and Oyama et al. (2023) derived an approximation at the SGNS optimum.
- No prior work found on: step size at fixed summed learning rate, the gradient-flow limit, corpus reuse as the source of a hump, the window acting beyond update count, or Gensim's score clipping.
- No theorem anywhere covers the inverted U of trained SGNS norms; existing theory is either at the optimum (Oyama, Zipfian whitening, Arora) or for gradient flow without frequency weighting (Karkada et al.).
- New negative result from this round: at the PNAS settings on BlogCatalog, the trained model learns each node's specificity in its predicted contexts, but its vector norm does not carry it, and Oyama's approximation fails; window 1 is the cause (window 5 restores it), and in text the relation is moderate.

## The PNAS hypothesis

Murray et al. (PNAS 2023), main text near Fig. 5 and SI S16 (arXiv 2012.02785v3, pp. 61–63):

- Rising branch: rarely updated organizations stay near their initialization; walks started uniformly give flatter norms (Fig. S32b), though a falling trend at large sizes remains.
- Falling branch: borrowed from Schakel & Wilson; an organization in many contexts gets a short vector (R² = 0.17 for "expected connectedness").
  The paper states that its experiments do not support specificity as the sole mechanism and that hierarchy distortion is also unsupported.
- Configuration-model, Barabási–Albert, and nested networks with 5-step walks reproduce the concave shape; a regular tree does not.
- Settings: Gensim SGNS, d = 300, window 1, negative exponent 1, 5 negatives, 5 epochs, default learning rate, fixed corpus.
  These are our setting B.

Verdict from the tests:

- Rising branch = undertraining: supported, and the peak's position moves with training.
- Falling branch = specificity: partly.
  At fixed degree, specificity predicts length (BlogCatalog within-bin Spearman 0.23 at B, 0.78 at A, 0.89 at SW), and the correlation grows with epochs.
  But on configuration-model walks like Fig. S33, the decline at the top comes from finite SGD steps: at the same summed learning rate, gradient flow gives norms rising with degree.
  At B on BlogCatalog the norm does not encode the specificity the model has learned; window 1 causes this (next section).
  On the flat-optimum expected-BA corpus, the hump comes from reusing one sampled corpus.

## Does Oyama's norm–KL relation hold in our runs?

`scripts/norm_tests/oyama_train.py` saves factors at 1, 2, 5, 10, 20, 50 epochs; `oyama_check.py` compares, per node, the exact KL of the model's own predicted context distribution, Oyama's quadratic form (their Eq. 11–12, centered, metric G = context covariance under the unigram), the balanced norm, and empirical specificity KL(C_i‖π).

Values at 5 → 50 epochs; "within" is the Spearman correlation inside quarter-decade frequency bins.
BlogCatalog seeds 0 and 1 agree to within 0.03 on every readout, so one seed is shown.

| Run | model KL vs empirical KL (Pearson) | quadratic vs model KL (Pearson) | balanced norm vs empirical KL, global (Spearman) | same, within bins | balanced-norm peak at 50 epochs |
|---|---|---|---|---|---|
| BlogCatalog, B (PNAS) | 0.73 → 0.85 | −0.37 → −0.52 | −0.72 → −0.63 | 0.38 → 0.42 | degree 74 |
| BlogCatalog, B with window 5 | 0.82 → 0.74 | 0.85 → 0.79 | 0.91 → 0.87 | 0.70 → 0.57 | lowest bin |
| BlogCatalog, A (defaults) | 0.38 → 0.81 | 0.56 → 0.88 | 0.74 → 0.94 | 0.75 → 0.73 | lowest bin |
| hep-th text, B | 0.36 → 0.77 | 0.62 → 0.35 | −0.13 → 0.39 | 0.63 → 0.45 | count 236 |
| hep-th text, A | −0.22 → 0.26 | 0.76 → 0.67 | −0.66 → 0.49 | 0.40 → 0.50 | count 23 |

- On BlogCatalog at window 5 (A, or B with only the window changed), Oyama's relation holds: the quadratic form matches the model's KL, the norm tracks specificity, and it falls monotonically with degree by 50 epochs.
- At B on BlogCatalog, the model's predicted context distribution matches empirical specificity (mean KL 3.97 vs 4.05 nats), yet the norm keeps rising with degree up to about 74, and Oyama's quadratic form anticorrelates with the model's KL in both seeds.
  Changing only the window to 5 restores the relation, so window-1 graph contexts break it; a plausible cause is that they lie far from the second-order regime Oyama's expansion assumes (untested).
- In text the relation is moderate at both settings, and at B the quadratic approximation weakens as training proceeds.
- Caveats: empirical KL ignores subsampling (A) and, for text, is the finite-corpus KL, which is inflated for rare words (Oyama et al. §6); the window-5 B run used Gensim's fixed window while its KL used shrinking-window weights.

## What theory exists

- **At the optimum.**
  Oyama et al. (2023) approximate 2·KL(p(·|w)‖p(·)) by the centered, whitened squared norm, assuming perfectly trained SGNS and a second-order expansion; it is an approximation, not a theorem.
  Yokoi et al. (2024, Zipfian whitening, Thm. 1) give a theorem for a refined, reverse-KL version.
  Arora et al. (2016, Thm. 2.2) prove that squared norm rises with log frequency under their generative model, the opposite of the falling branch.
- **Training speed.**
  Kunstner et al. (2024, Thm. 3) show gradient flow fits class k at rate ∝ π_k, which gives the rising branch.
  Gong et al. (2018, §3) report frequent words move about twice as far as rare ones during word2vec training.
- **Gradient-flow dynamics.**
  Karkada et al. (2025) solve gradient flow for a quartic approximation of word2vec with constant weighting, so it has no frequency dependence; their App. A.2 leaves finite step sizes open.
- **Finite steps.**
  Smith et al. (2021, Eq. 22) show SGD follows a modified loss with a per-example gradient-variance term proportional to the step size, which is absent under gradient flow and nonzero at the optimum.
  It is the best generic candidate for the step-size effect.
  Barrett & Dherin (2021), Gidel et al. (2019), Wang et al. (2022), and Kunin et al. (2021) give related discrete-step results for full-batch or factorized models; none treats per-item norms by frequency.
- **Our manuscript.**
  Population geometry (degree cancellation at unigram negatives, credited to Davison, Morgan & Ward 2024), the realized-graph score target, a two-force response model that is a framework rather than a derivation, and a boundary result that near-optimal loss can leave norm shapes unsettled.
  None of it derives the trained curve.

A test of the Smith et al. mechanism (`stepsize_specificity.py`, PR #36 graphs, 5 seeds): relative to gradient flow at τ = 0.0628, the extra shrinkage at constant rate 0.025 tracks degree (Spearman −0.82).
At fixed degree, more specific nodes shrink slightly less (within-bin Spearman 0.27), but KL adds almost nothing once degree is in the regression (standardized coefficients −0.109 for log degree, 0.005 for KL).
So the context-diversity version of that mechanism is at most a small part; the step-size effect is tied to degree or exposure.

## Claim by claim

| Claim | Closest precedent | What is new |
|---|---|---|
| Rare items are short because undertrained | Schakel & Wilson 2015 (heuristic); Gong et al. 2018 §3; Murray et al. SI S16; Oyama et al. App. G; Kunstner et al. 2024 Thm. 3 | Nothing; cite |
| The ordering changes with training | Zhang, Lu & Shai 2018, Fig. 2 (BlogCatalog; LINE, DeepWalk, node2vec; constant rate 0.025; high-degree norms plateau, low-degree overtake by about 10 iterations); Oyama et al. App. G (Wikipedia; hump at 10 epochs, monotone decline at 100) | Peak location tracked continuously and shown to depend on epochs, window, and learning rate, in text and graphs |
| Step size matters at fixed summed learning rate; gradient flow rises | None in embeddings; generic: Smith et al. 2021, Barrett & Dherin 2021; Stollenwerk & Stollenwerk 2025 show the optimizer (Adam vs. coupled Adam) flips the sign of norm–frequency correlation in LMs | Strongest claim: matched-τ sweep, gradient-flow limit, and shrinkage tied to degree more than KL |
| Reusing one corpus makes a hump where the optimum is flat | Ingredients only: Oyama et al. §6 (finite-sample KL bias); Wilson & Schakel 2015 (equal contexts, unequal norms); Valentini et al. 2023 (frequency artifacts in cosine on shuffled corpora); Antoniak & Mimno 2018 (bootstrap variability) | The fresh-vs-reused experiment; practical relevance because PyG `Node2Vec` resamples walks every batch while Gensim pipelines reuse a fixed corpus. On BlogCatalog a fresh corpus leaves the curve unchanged, so this applies to the flat-optimum control |
| Norm tracks specificity at fixed frequency | Wilson & Schakel 2015 §5 (controlled noise mixing, text); Schakel & Wilson 2015; Oyama et al. 2023 | Graph version with window-specific KL against the stationary law (closed form at window 1); growth with training; and the boundary: at PNAS settings on graphs (window 1) the model learns KL but the norm does not encode it |
| Window acts beyond update count | None found | w1 at 15 epochs vs w5 at 5 epochs on the same BA graph |
| Gensim's ±6 update skipping is not responsible | None found | Methods note |

Other relevant work:

- Su, Ren & Veitch (2026, arXiv 2606.30625) derive that optimization dynamics imprint specificity in contrastive, scale-invariant embedding norms; longer means less specific there, the opposite sign. It is the closest "training sets the norm" theory.
- Pagliardini et al. (2018, Fig. 1) and Khodak et al. (2018) show further instances of the inverted U in text.
- Sulc (2026, arXiv 2604.00911) reports DeepWalk norm–degree Spearman −0.80 at 100 epochs; Barros et al. (2021, arXiv 2111.06161) assume the opposite sign.
  Both fit a budget-dependent curve.
- Kim, Kojaku & Ahn (Sci. Adv. 2026) use cosine because norms reflect random-walk frequency, citing Murray et al.

## Searches that found nothing

- Norm–frequency or norm–degree curves as a function of learning rate or SGD step size (SGNS, node embeddings, LMs).
- Peak position versus epochs, window, or learning rate beyond Oyama's two epoch settings.
- Fixed versus resampled walk corpora and the effect on norms.
- KL(context‖stationary) versus norm in graph embeddings.
- MAX_EXP clipping effects on norms.
- Norm–frequency tracked across LM pretraining checkpoints.

## Manuscript actions

- Cite Oyama et al. App. G and Zhang, Lu & Shai (2018) where the PRE manuscript discusses training dependence; both precede "the curve depends on training budget."
- Cite Wilson & Schakel (2015) §5 as the controlled precedent for specificity at fixed frequency (it is in the bib and cited once).
- Add Kunstner et al. (2024) for the rising branch and Smith et al. (2021) as the candidate step-size mechanism; add Stollenwerk & Stollenwerk (2025) and Su, Ren & Veitch (2026) as related optimizer-dependence work.
- `references.bib` has duplicate entries (`oyama2022norm`/`oyama2023`, `schakel2015measuring`/`schakel2015`, `murray2023unsupervised`/`murray2023`); the key `oyama2022norm` points to the EMNLP 2023 paper.

## Verified sources

- Antoniak, M., & Mimno, D. (2018). Evaluating the stability of embedding-based word similarities. TACL 6. https://aclanthology.org/Q18-1008/
- Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2016). A latent variable model approach to PMI-based word embeddings. TACL. https://arxiv.org/abs/1502.03520
- Barrett, D. G. T., & Dherin, B. (2021). Implicit gradient regularization. ICLR. https://arxiv.org/abs/2009.11162
- Gidel, G., Bach, F., & Lacoste-Julien, S. (2019). Implicit regularization of discrete gradient dynamics in linear neural networks. NeurIPS. https://arxiv.org/abs/1904.13262
- Gong, C., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y. (2018). FRAGE: Frequency-agnostic word representation. NeurIPS. https://arxiv.org/abs/1809.06858
- Hellrich, J., & Hahn, U. (2016). Bad company—neighborhoods in neural embedding spaces considered harmful. COLING. https://aclanthology.org/C16-1262/
- Karkada, D., Simon, J. B., Bahri, Y., & DeWeese, M. R. (2025). Closed-form training dynamics reveal learned features and linear structure in word2vec-like models. NeurIPS. https://arxiv.org/abs/2502.09863
- Kunin, D., Sagastuy-Brena, J., Ganguli, S., Yamins, D. L. K., & Tanaka, H. (2021). Neural mechanics. ICLR. https://arxiv.org/abs/2012.04728
- Kunstner, F., Yadav, R., Milligan, A., Schmidt, M., & Bietti, A. (2024). Heavy-tailed class imbalance and why Adam outperforms gradient descent on language models. NeurIPS. https://arxiv.org/abs/2402.19449
- Murray, D., Yoon, J., Kojaku, S., Costas, R., Jung, W.-S., Milojević, S., & Ahn, Y.-Y. (2023). Unsupervised embedding of trajectories captures the latent structure of scientific migration. PNAS 120(52). https://doi.org/10.1073/pnas.2305414120
- Oyama, M., Yokoi, S., & Shimodaira, H. (2023). Norm of word embedding encodes information gain. EMNLP. https://aclanthology.org/2023.emnlp-main.131/
- Pagliardini, M., Gupta, P., & Jaggi, M. (2018). Unsupervised learning of sentence embeddings using compositional n-gram features. NAACL. https://aclanthology.org/N18-1049/
- Qiu, J., Wang, C., Liao, B., Peng, R., & Tang, J. (2020). A matrix Chernoff bound for Markov chains and its application to co-occurrence matrices. NeurIPS. https://arxiv.org/abs/2008.02464
- Schakel, A. M. J., & Wilson, B. J. (2015). Measuring word significance using distributed representations of words. https://arxiv.org/abs/1508.02297
- Smith, S. L., Dherin, B., Barrett, D. G. T., & De, S. (2021). On the origin of implicit regularization in stochastic gradient descent. ICLR. https://arxiv.org/abs/2101.12176
- Stollenwerk, F., & Stollenwerk, T. (2025). Better embeddings with coupled Adam. ACL. https://aclanthology.org/2025.acl-long.1321/
- Su, Z., Ren, J., & Veitch, V. (2026). Optimization dynamics imprint semantic specificity in contrastive embedding norms. https://arxiv.org/abs/2606.30625
- Sulc, A. (2026). Event embedding of protein networks: Compositional learning of biological function. https://arxiv.org/abs/2604.00911
- Valentini, F., Sosa, J., Slezak, D., & Altszyler, E. (2023). Investigating the frequency distortion of word embeddings and its impact on bias metrics. Findings of EMNLP. https://aclanthology.org/2023.findings-emnlp.9/
- Wang, Y., Chen, M., Zhao, T., & Tao, M. (2022). Large learning rate tames homogeneity: Convergence and balancing effect. ICLR. https://arxiv.org/abs/2110.03677
- Wilson, B. J., & Schakel, A. M. J. (2015). Controlled experiments for word embeddings. https://arxiv.org/abs/1510.02675
- Yokoi, S., Bao, H., Kurita, H., & Shimodaira, H. (2024). Zipfian whitening. NeurIPS. https://arxiv.org/abs/2411.00680
- Zhang, Y., Lu, J., & Shai, O. (2018). Improve network embeddings with regularization. CIKM, 1643–1646. https://doi.org/10.1145/3269206.3269320 (full text read from the author-hosted copy at https://jlu.myweb.cs.uwindsor.ca/n2v.pdf)
