# U10 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U10
Script SHA-256: 33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: 75. rng seed 42, bootstrap 10000 over the five seeds.
Started 2026-11-21 19:12:20, ended 2026-11-21 19:16:34.

Scores from float16 logits19 converted to float32 (softmax over all 19 logits for GEN, M = 19, gamma 0.1). Energy reference = exported float32 energy. Check: -logsumexp of the stored float16 logits versus exported energy: max |difference| = 5.722e-06 (criterion 1e-3; PASS).

Paired AUROC difference to energy, mean over the five classes per seed, then seed interval:

| model       |   gen_gamma0.1_auroc_minus_energy |   gen_gamma0.1_auroc_minus_energy_lo |   gen_gamma0.1_auroc_minus_energy_hi |   logit_norm_auroc_minus_energy |   logit_norm_auroc_minus_energy_lo |   logit_norm_auroc_minus_energy_hi |   maxlogit_minus_mean_auroc_minus_energy |   maxlogit_minus_mean_auroc_minus_energy_lo |   maxlogit_minus_mean_auroc_minus_energy_hi |   n_seeds |
|:------------|----------------------------------:|-------------------------------------:|-------------------------------------:|--------------------------------:|-----------------------------------:|-----------------------------------:|-----------------------------------------:|--------------------------------------------:|--------------------------------------------:|----------:|
| bilstm_dnn  |                            0.1116 |                               0.0802 |                               0.1323 |                          0.1251 |                             0.0811 |                             0.1614 |                                   0.1045 |                                      0.0771 |                                      0.1326 |         5 |
| trace_moe   |                           -0.015  |                              -0.0314 |                               0.0015 |                         -0.2562 |                            -0.3167 |                            -0.1956 |                                  -0.0355 |                                     -0.0499 |                                     -0.0215 |         5 |
| transformer |                           -0.0513 |                              -0.0625 |                              -0.0412 |                         -0.4742 |                            -0.5348 |                            -0.4011 |                                  -0.0641 |                                     -0.0784 |                                     -0.0507 |         5 |
