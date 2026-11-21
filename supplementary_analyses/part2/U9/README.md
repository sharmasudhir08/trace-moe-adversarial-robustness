# U9 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U9
Script SHA-256: 33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: 75. rng seed 42, bootstrap 10000 over the five seeds.
Started 2026-11-21 19:11:02, ended 2026-11-21 19:12:20.

Known windows only (held-out class excluded); a prediction is wrong if the argmax over the 18 known-class logits differs from y19 (held-out logit masked, as in T2). Score = -energy or -MSP (higher = more likely wrong); AURC accepts the lowest scores first. Float16 stored logits are used only for the argmax.

Pooled over classes (mean over the five classes per seed, then seed interval):

| model       |   auroc_energy |   auroc_energy_lo |   auroc_energy_hi |   auroc_msp |   auroc_msp_lo |   auroc_msp_hi |   auroc_diff_energy_minus_msp |   auroc_diff_energy_minus_msp_lo |   auroc_diff_energy_minus_msp_hi |   aurc_diff_energy_minus_msp |   aurc_diff_energy_minus_msp_lo |   aurc_diff_energy_minus_msp_hi |   n_seeds |
|:------------|---------------:|------------------:|------------------:|------------:|---------------:|---------------:|------------------------------:|---------------------------------:|---------------------------------:|-----------------------------:|--------------------------------:|--------------------------------:|----------:|
| bilstm_dnn  |         0.7142 |            0.6778 |            0.7415 |      0.8715 |         0.8587 |         0.8839 |                       -0.1573 |                          -0.2045 |                          -0.1212 |                      -0.0011 |                         -0.0051 |                          0.0027 |         5 |
| trace_moe   |         0.7258 |            0.6978 |            0.7582 |      0.7054 |         0.6872 |         0.7291 |                        0.0205 |                           0.0066 |                           0.0339 |                      -0.0169 |                         -0.018  |                         -0.0153 |         5 |
| transformer |         0.6634 |            0.6476 |            0.6788 |      0.6831 |         0.6693 |         0.6968 |                       -0.0196 |                          -0.0324 |                          -0.0032 |                       0.0053 |                          0.0038 |                          0.0073 |         5 |
