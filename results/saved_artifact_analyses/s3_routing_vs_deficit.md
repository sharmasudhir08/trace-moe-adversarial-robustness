# S3 (exploratory): router instability versus the eps 0.25 deficit, five seeds

|   seed |   instability_eps025 |   f1_trace_moe |   f1_best_other |   deficit |   rank |
|-------:|---------------------:|---------------:|----------------:|----------:|-------:|
|     42 |               0.341  |         0.7034 |          0.7315 |    0.0281 |      3 |
|     43 |               0.2266 |         0.7164 |          0.7369 |    0.0205 |      3 |
|     44 |               0.171  |         0.749  |          0.7607 |    0.0118 |      2 |
|     45 |               0.2669 |         0.74   |          0.7556 |    0.0157 |      2 |
|     46 |               0.1991 |         0.7582 |          0.7632 |    0.0049 |      2 |

Spearman rho = 0.800 (exact two-sided permutation p = 0.133); Pearson r = 0.818. n = 5, descriptive only.

Instability is from clean-trained models, the deficit from fine-tuned models.