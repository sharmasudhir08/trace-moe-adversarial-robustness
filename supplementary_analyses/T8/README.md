# T8 FLOPs (not a new measurement of speed; analytic count)

Script `scripts/p4_rev_t8_flops.py` SHA-256 181e0b66abc420c5d9ef83519ae9cadd5e2fabe32793fbed20d9acf6ddea2cf0; Python 3.12.4; 2026-11-21 11:54:53.

| model       |   params_flops_file |   params_paper4_benchmark | params_equal   |   flops_per_window |   mflops_per_window |   torch_flopcounter_mflops_lower_bound |   relative_gap_to_counter |   energy_score_ops |   msp_score_ops |   score_ops_share_of_model_flops |
|:------------|--------------------:|--------------------------:|:---------------|-------------------:|--------------------:|---------------------------------------:|--------------------------:|-------------------:|----------------:|---------------------------------:|
| bilstm_dnn  |               70811 |                     70811 | True           |            3067648 |              3.0676 |                                 0.0724 |                  0.976383 |                 76 |             111 |                          6.1e-05 |
| transformer |              308251 |                    308251 | True           |            8841984 |              8.842  |                                 8.5798 |                  0.029648 |                 76 |             111 |                          2.1e-05 |
| trace_moe   |              685041 |                    685041 | True           |           11542016 |             11.542  |                                11.2297 |                  0.027059 |                 76 |             111 |                          1.6e-05 |

TRACE-MoE count 11.542 MFLOPs against the manuscript's 11.54: relative difference 0.0173% (stop criterion 5%).
The torch FlopCounterMode figure counts matmul kernels only, so it is a lower bound; for the BiLSTM-DNN it misses the fused LSTM kernels (large gap, explained by the method note in flops.json). The two methods agree within 5% for the Transformer-like models (see relative_gap_to_counter); the BiLSTM-DNN difference is explained, not an error. Score overhead is an analytic elementary-operation count (exp, add, divide, max and log each one operation), negligible against the model FLOPs.
