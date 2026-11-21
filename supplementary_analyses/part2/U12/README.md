# U12 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U12
Script SHA-256: 33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: 75. rng seed 42, bootstrap 10000 over the five seeds.
Started 2026-11-21 19:10:53, ended 2026-11-21 19:11:02.

Threshold = q-th percentile of known validation energy (q in 1, 2, 5, 10); energy flag only. Check: q = 5 threshold equals exported thr, max difference 4.77e-07 (criterion 1e-6; PASS).

Mean over the five classes and five seeds:

| model       |   quantile |   caught_energy |   caught_energy_lo |   caught_energy_hi |   far_known |   far_known_lo |   far_known_hi |   n_seeds |
|:------------|-----------:|----------------:|-------------------:|-------------------:|------------:|---------------:|---------------:|----------:|
| bilstm_dnn  |          1 |          0.2089 |             0.1494 |             0.2439 |      0.0085 |         0.0081 |         0.0089 |         5 |
| bilstm_dnn  |          2 |          0.2942 |             0.2442 |             0.3237 |      0.0163 |         0.0154 |         0.0171 |         5 |
| bilstm_dnn  |          5 |          0.3764 |             0.3056 |             0.4297 |      0.051  |         0.0448 |         0.0554 |         5 |
| bilstm_dnn  |         10 |          0.4069 |             0.3284 |             0.4774 |      0.102  |         0.0919 |         0.1135 |         5 |
| trace_moe   |          1 |          0.2762 |             0.2119 |             0.3405 |      0.0091 |         0.009  |         0.0092 |         5 |
| trace_moe   |          2 |          0.3382 |             0.269  |             0.4112 |      0.0163 |         0.016  |         0.0167 |         5 |
| trace_moe   |          5 |          0.5009 |             0.4642 |             0.5364 |      0.0418 |         0.0399 |         0.0447 |         5 |
| trace_moe   |         10 |          0.5574 |             0.5266 |             0.5852 |      0.0941 |         0.0863 |         0.1019 |         5 |
| transformer |          1 |          0.3267 |             0.2989 |             0.3522 |      0.0092 |         0.0086 |         0.0101 |         5 |
| transformer |          2 |          0.3887 |             0.3539 |             0.4133 |      0.0168 |         0.0159 |         0.0182 |         5 |
| transformer |          5 |          0.5605 |             0.5496 |             0.5694 |      0.044  |         0.0427 |         0.0455 |         5 |
| transformer |         10 |          0.5945 |             0.5807 |             0.6076 |      0.0882 |         0.0801 |         0.1002 |         5 |
