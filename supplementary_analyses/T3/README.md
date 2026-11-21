# T3: feature-space scores on learned embeddings (post-hoc exploratory)

Summary written 2026-11-21 16:21:32 by `scripts/p4_rev_t3_summarize.py` (SHA-256 735d3ca7900ab854bfe2dadfdfe4d24074d56a558fdf06b15e84b00cd969b960).
Input: `per_run_feature_scores.csv` (75 runs; expected 75). Scoring script `scripts/p4_rev_t3_feature_scores.py`.
Python 3.12.4, numpy 2.5.3, pandas 3.0.6; bootstrap rng 42, 10000 resamples over seeds.

Consistency with score_export (energy within 1e-3, equal labels): 75 of 75 runs pass.

Fixed settings: fit on known-class training windows (class-stratified subsample of at most 200,000, rng 42); Mahalanobis class-conditional with Ledoit-Wolf covariance, minimum over known classes; kNN on L2-normalised embeddings, k = 50; ReAct clip at the 90th percentile of training activations; ViM subspace dimension d/2. None was changed after seeing test results. All scores: higher = more unknown; held-out class positive.

Status: exploratory, no verdict. Files: `summary_feature_scores.csv`, `paired_vs_energy.csv`.
