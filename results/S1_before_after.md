# Supplementary Table S1: verdicts before and after replacing the one-epoch runs

Before = Charter v1.3 run set (31 of 75 runs trained for one epoch). After = Charter v1.4 run set (all 75 runs trained under the specified 50-epoch protocol, patience 10). Decision rules, margins, thresholds and the statistical recipe are identical in both columns. Every hypothesis and every pair is listed, changed or not; nothing is selected. All values are read from the verdict files; no statistic is recomputed for this table.

## Overall verdict lines

| Item | Before (v1.3 run set) | After (v1.4 run set) | Changed |
|---|---|---|---|
| H1.overall_verdict | SUPPORTED (strong, overall) | MIXED | yes |
| H2.per_model_breakdown.bilstm_dnn.verdict | NOT SUPPORTED | NOT SUPPORTED | no |
| H2.per_model_breakdown.trace_moe.verdict | NOT SUPPORTED | NOT SUPPORTED | no |
| H2.per_model_breakdown.transformer.verdict | MIXED | SUPPORTED (strong) | yes |
| H3.per_baseline_breakdown.bilstm_dnn.verdict | SUPPORTED (strong) | SUPPORTED (strong) | no |
| H3.per_baseline_breakdown.transformer.verdict | SUPPORTED (preliminary) | NOT SUPPORTED | yes |
| H4.overall_verdict | INCONCLUSIVE (overall; ≥3 pairs inconclusive at 5-seed scope) | SUPPORTED (overall; all 15 pairs within 10 pt) | yes |

## H1  (5 rows, 1 verdict changes); metric: mean

| Pair | Before: metric [95% CI] | After: metric [95% CI] | Before verdict | After verdict | Changed |
|---|---|---|---|---|---|
| ARP_Spoofing | 0.9762 [0.9460, 0.9950] | 0.9239 [0.7808, 0.9992] | SUPPORTED (strong) | INCONCLUSIVE | yes |
| MQTT-DDoS-Publish_Flood | 1.0000 [1.0000, 1.0000] | 1.0000 [0.9999, 1.0000] | SUPPORTED (strong) | SUPPORTED (strong) | no |
| MQTT-Malformed_Data | 0.9770 [0.9394, 1.0000] | 0.9941 [0.9858, 1.0000] | SUPPORTED (strong) | SUPPORTED (strong) | no |
| Recon-VulScan | 0.9969 [0.9927, 0.9998] | 0.9998 [0.9994, 1.0000] | SUPPORTED (strong) | SUPPORTED (strong) | no |
| TCP_IP-DoS-UDP | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | SUPPORTED (strong) | SUPPORTED (strong) | no |

## H2  (15 rows, 8 verdict changes); metric: mean_diff

| Pair | Before: metric [95% CI] | After: metric [95% CI] | Before verdict | After verdict | Changed |
|---|---|---|---|---|---|
| bilstm_dnn x ARP_Spoofing | -0.1517 [-0.2288, -0.0977] | -0.1174 [-0.1697, -0.0701] | NOT SUPPORTED | NOT SUPPORTED | no |
| bilstm_dnn x MQTT-DDoS-Publish_Flood | 0.0174 [-0.1915, 0.2439] | -0.1082 [-0.2703, 0.0540] | SUPPORTED (preliminary) | INCONCLUSIVE | yes |
| bilstm_dnn x MQTT-Malformed_Data | -0.1160 [-0.1578, -0.0743] | -0.0649 [-0.1228, -0.0188] | NOT SUPPORTED | NOT SUPPORTED | no |
| bilstm_dnn x Recon-VulScan | -0.1850 [-0.3039, -0.0699] | -0.1236 [-0.1774, -0.0676] | NOT SUPPORTED | NOT SUPPORTED | no |
| bilstm_dnn x TCP_IP-DoS-UDP | -0.0978 [-0.2297, -0.0069] | -0.0775 [-0.2272, 0.0714] | NOT SUPPORTED | INCONCLUSIVE | yes |
| trace_moe x ARP_Spoofing | 0.0044 [-0.0086, 0.0183] | -0.0027 [-0.0276, 0.0174] | NOT SUPPORTED | NOT SUPPORTED | no |
| trace_moe x MQTT-DDoS-Publish_Flood | 0.1658 [-0.0147, 0.3700] | -0.0537 [-0.1933, 0.0858] | SUPPORTED (preliminary) | INCONCLUSIVE | yes |
| trace_moe x MQTT-Malformed_Data | -0.0006 [-0.0059, 0.0038] | -0.0006 [-0.0067, 0.0055] | NOT SUPPORTED | NOT SUPPORTED | no |
| trace_moe x Recon-VulScan | -0.0030 [-0.0110, 0.0035] | -0.0023 [-0.0070, 0.0032] | NOT SUPPORTED | NOT SUPPORTED | no |
| trace_moe x TCP_IP-DoS-UDP | -0.1155 [-0.2388, 0.0044] | -0.1286 [-0.2563, -0.0238] | INCONCLUSIVE | NOT SUPPORTED | yes |
| transformer x ARP_Spoofing | 0.0005 [-0.0178, 0.0171] | 0.0207 [0.0109, 0.0304] | NOT SUPPORTED | SUPPORTED | yes |
| transformer x MQTT-DDoS-Publish_Flood | -0.2327 [-0.3729, -0.0948] | -0.1811 [-0.2938, -0.0684] | NOT SUPPORTED | NOT SUPPORTED | no |
| transformer x MQTT-Malformed_Data | -0.0346 [-0.1186, 0.0124] | 0.0261 [0.0080, 0.0510] | INCONCLUSIVE | SUPPORTED | yes |
| transformer x Recon-VulScan | -0.0668 [-0.1951, 0.0615] | 0.0398 [0.0175, 0.0681] | INCONCLUSIVE | SUPPORTED | yes |
| transformer x TCP_IP-DoS-UDP | 0.0657 [-0.0858, 0.2261] | 0.0888 [0.0010, 0.2091] | SUPPORTED (preliminary) | SUPPORTED | yes |

## H3  (10 rows, 4 verdict changes); metric: mean_diff

| Pair | Before: metric [95% CI] | After: metric [95% CI] | Before verdict | After verdict | Changed |
|---|---|---|---|---|---|
| bilstm_dnn x ARP_Spoofing | 0.2607 [0.1045, 0.4362] | 0.1084 [0.0484, 0.1700] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x MQTT-DDoS-Publish_Flood | 0.2213 [0.1285, 0.3629] | -0.0085 [-0.2076, 0.1361] | SUPPORTED | INCONCLUSIVE | yes |
| bilstm_dnn x MQTT-Malformed_Data | 0.1553 [0.0824, 0.2190] | 0.0753 [0.0255, 0.1424] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x Recon-VulScan | 0.3182 [0.1829, 0.4535] | 0.2413 [0.1660, 0.3166] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x TCP_IP-DoS-UDP | -0.2150 [-0.4467, 0.0554] | -0.3511 [-0.5995, -0.0301] | INCONCLUSIVE | NOT SUPPORTED | yes |
| transformer x ARP_Spoofing | 0.0041 [-0.0047, 0.0181] | -0.0163 [-0.0467, 0.0009] | NOT SUPPORTED | NOT SUPPORTED | no |
| transformer x MQTT-DDoS-Publish_Flood | 0.4173 [0.2704, 0.5750] | 0.2097 [0.1120, 0.3130] | SUPPORTED | SUPPORTED | no |
| transformer x MQTT-Malformed_Data | 0.0470 [-0.0090, 0.1450] | -0.0006 [-0.0102, 0.0071] | SUPPORTED (preliminary) | NOT SUPPORTED | yes |
| transformer x Recon-VulScan | 0.1285 [0.0014, 0.2584] | -0.0051 [-0.0191, 0.0082] | SUPPORTED | NOT SUPPORTED | yes |
| transformer x TCP_IP-DoS-UDP | -0.0888 [-0.4373, 0.2596] | -0.3535 [-0.6009, 0.0117] | INCONCLUSIVE | INCONCLUSIVE | no |

## H4  (15 rows, 4 verdict changes); metric: mean_degr

| Pair | Before: metric [95% CI] | After: metric [95% CI] | Before verdict | After verdict | Changed |
|---|---|---|---|---|---|
| bilstm_dnn x ARP_Spoofing | 0.0455 [0.0050, 0.0867] | 0.0006 [-0.0047, 0.0053] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x MQTT-DDoS-Publish_Flood | 0.0635 [-0.0186, 0.1505] | -0.0175 [-0.0197, -0.0148] | INCONCLUSIVE | SUPPORTED | yes |
| bilstm_dnn x MQTT-Malformed_Data | 0.0523 [-0.0000, 0.1227] | -0.0014 [-0.0041, 0.0011] | INCONCLUSIVE | SUPPORTED | yes |
| bilstm_dnn x Recon-VulScan | 0.0680 [-0.0006, 0.1497] | -0.0051 [-0.0114, 0.0013] | INCONCLUSIVE | SUPPORTED | yes |
| bilstm_dnn x TCP_IP-DoS-UDP | 0.0827 [0.0011, 0.1653] | 0.0019 [0.0008, 0.0030] | INCONCLUSIVE | SUPPORTED | yes |
| trace_moe x ARP_Spoofing | 0.0094 [-0.0141, 0.0305] | -0.0184 [-0.0247, -0.0125] | SUPPORTED | SUPPORTED | no |
| trace_moe x MQTT-DDoS-Publish_Flood | 0.0031 [-0.0188, 0.0250] | -0.0140 [-0.0222, -0.0070] | SUPPORTED | SUPPORTED | no |
| trace_moe x MQTT-Malformed_Data | 0.0159 [-0.0040, 0.0359] | 0.0052 [-0.0049, 0.0144] | SUPPORTED | SUPPORTED | no |
| trace_moe x Recon-VulScan | 0.0191 [-0.0054, 0.0476] | 0.0005 [-0.0066, 0.0069] | SUPPORTED | SUPPORTED | no |
| trace_moe x TCP_IP-DoS-UDP | 0.0247 [0.0022, 0.0476] | 0.0065 [-0.0001, 0.0131] | SUPPORTED | SUPPORTED | no |
| transformer x ARP_Spoofing | -0.0112 [-0.0374, 0.0140] | -0.0238 [-0.0428, -0.0086] | SUPPORTED | SUPPORTED | no |
| transformer x MQTT-DDoS-Publish_Flood | -0.0063 [-0.0291, 0.0153] | -0.0164 [-0.0329, -0.0033] | SUPPORTED | SUPPORTED | no |
| transformer x MQTT-Malformed_Data | 0.0060 [-0.0198, 0.0293] | -0.0077 [-0.0242, 0.0054] | SUPPORTED | SUPPORTED | no |
| transformer x Recon-VulScan | 0.0086 [-0.0115, 0.0304] | -0.0094 [-0.0164, -0.0024] | SUPPORTED | SUPPORTED | no |
| transformer x TCP_IP-DoS-UDP | 0.0095 [-0.0166, 0.0311] | -0.0054 [-0.0223, 0.0077] | SUPPORTED | SUPPORTED | no |

## ESA1  (15 rows, no verdict column); metric: mean_far

| Pair | Before: metric [95% CI] | After: metric [95% CI] | Before verdict | After verdict | Changed |
|---|---|---|---|---|---|
| bilstm_dnn x ARP_Spoofing | 0.0520 [-0.0058, 0.0121] | 0.0435 [-0.0079, -0.0051] |  |  | n/a |
| bilstm_dnn x MQTT-DDoS-Publish_Flood | 0.0718 [-0.0163, 0.0897] | 0.0428 [-0.0159, 0.0017] |  |  | n/a |
| bilstm_dnn x MQTT-Malformed_Data | 0.0573 [-0.0160, 0.0434] | 0.0631 [-0.0066, 0.0462] |  |  | n/a |
| bilstm_dnn x Recon-VulScan | 0.0478 [-0.0239, 0.0334] | 0.0671 [-0.0135, 0.0477] |  |  | n/a |
| bilstm_dnn x TCP_IP-DoS-UDP | 0.0439 [-0.0158, 0.0081] | 0.0385 [-0.0159, -0.0079] |  |  | n/a |
| trace_moe x ARP_Spoofing | 0.0432 [-0.0089, -0.0047] | 0.0404 [-0.0152, -0.0053] |  |  | n/a |
| trace_moe x MQTT-DDoS-Publish_Flood | 0.0415 [-0.0105, -0.0064] | 0.0434 [-0.0110, -0.0012] |  |  | n/a |
| trace_moe x MQTT-Malformed_Data | 0.0395 [-0.0139, -0.0083] | 0.0378 [-0.0152, -0.0092] |  |  | n/a |
| trace_moe x Recon-VulScan | 0.0407 [-0.0123, -0.0066] | 0.0409 [-0.0118, -0.0065] |  |  | n/a |
| trace_moe x TCP_IP-DoS-UDP | 0.0408 [-0.0121, -0.0064] | 0.0466 [-0.0107, 0.0083] |  |  | n/a |
| transformer x ARP_Spoofing | 0.0465 [-0.0124, 0.0060] | 0.0494 [-0.0067, 0.0068] |  |  | n/a |
| transformer x MQTT-DDoS-Publish_Flood | 0.0431 [-0.0077, -0.0061] | 0.0426 [-0.0080, -0.0069] |  |  | n/a |
| transformer x MQTT-Malformed_Data | 0.0407 [-0.0136, -0.0070] | 0.0424 [-0.0082, -0.0071] |  |  | n/a |
| transformer x Recon-VulScan | 0.0454 [-0.0107, 0.0046] | 0.0423 [-0.0079, -0.0074] |  |  | n/a |
| transformer x TCP_IP-DoS-UDP | 0.0398 [-0.0153, -0.0068] | 0.0431 [-0.0076, -0.0062] |  |  | n/a |

## H5  (15 rows, 0 verdict changes); metric: mean_auroc

| Pair | Before: metric [95% CI] | After: metric [95% CI] | Before verdict | After verdict | Changed |
|---|---|---|---|---|---|
| bilstm_dnn x ARP_Spoofing | 0.9215 [0.9196, 0.9230] | 0.9215 [0.9196, 0.9230] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x MQTT-DDoS-Publish_Flood | 0.9361 [0.9360, 0.9363] | 0.9361 [0.9360, 0.9363] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x MQTT-Malformed_Data | 0.8879 [0.8857, 0.8896] | 0.8879 [0.8857, 0.8896] | SUPPORTED | SUPPORTED | no |
| bilstm_dnn x Recon-VulScan | 0.7656 [0.7642, 0.7674] | 0.7656 [0.7642, 0.7674] | INCONCLUSIVE | INCONCLUSIVE | no |
| bilstm_dnn x TCP_IP-DoS-UDP | 0.9513 [0.9314, 0.9712] | 0.9513 [0.9314, 0.9712] | SUPPORTED | SUPPORTED | no |
| trace_moe x ARP_Spoofing | 0.9215 [0.9196, 0.9230] | 0.9215 [0.9196, 0.9230] | SUPPORTED | SUPPORTED | no |
| trace_moe x MQTT-DDoS-Publish_Flood | 0.9361 [0.9360, 0.9363] | 0.9361 [0.9360, 0.9363] | SUPPORTED | SUPPORTED | no |
| trace_moe x MQTT-Malformed_Data | 0.8879 [0.8857, 0.8896] | 0.8879 [0.8857, 0.8896] | SUPPORTED | SUPPORTED | no |
| trace_moe x Recon-VulScan | 0.7656 [0.7642, 0.7674] | 0.7656 [0.7642, 0.7674] | INCONCLUSIVE | INCONCLUSIVE | no |
| trace_moe x TCP_IP-DoS-UDP | 0.9513 [0.9314, 0.9712] | 0.9513 [0.9314, 0.9712] | SUPPORTED | SUPPORTED | no |
| transformer x ARP_Spoofing | 0.9215 [0.9196, 0.9230] | 0.9215 [0.9196, 0.9230] | SUPPORTED | SUPPORTED | no |
| transformer x MQTT-DDoS-Publish_Flood | 0.9361 [0.9360, 0.9363] | 0.9361 [0.9360, 0.9363] | SUPPORTED | SUPPORTED | no |
| transformer x MQTT-Malformed_Data | 0.8879 [0.8857, 0.8896] | 0.8879 [0.8857, 0.8896] | SUPPORTED | SUPPORTED | no |
| transformer x Recon-VulScan | 0.7656 [0.7642, 0.7674] | 0.7656 [0.7642, 0.7674] | INCONCLUSIVE | INCONCLUSIVE | no |
| transformer x TCP_IP-DoS-UDP | 0.9513 [0.9314, 0.9712] | 0.9513 [0.9314, 0.9712] | SUPPORTED | SUPPORTED | no |

## H1 per-class detection rate by seed (TRACE-MoE caught_any)

| Class | Run set | seed42 | seed43 | seed44 | seed45 | seed46 | mean |
|---|---|---|---|---|---|---|---|
| ARP_Spoofing | before | 0.9902 | 0.9942 | 0.9167 | 0.9983 | 0.9815 | 0.9762 |
| ARP_Spoofing | after | 0.6408 | 0.9988 | 1.0000 | 0.9983 | 0.9815 | 0.9239 |
| Recon-VulScan | before | 0.9990 | 0.9961 | 0.9892 | 1.0000 | 1.0000 | 0.9969 |
| Recon-VulScan | after | 0.9990 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 |
| MQTT-Malformed_Data | before | 1.0000 | 0.9059 | 1.0000 | 0.9792 | 1.0000 | 0.9770 |
| MQTT-Malformed_Data | after | 1.0000 | 0.9913 | 1.0000 | 0.9792 | 1.0000 | 0.9941 |
| TCP_IP-DoS-UDP | before | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TCP_IP-DoS-UDP | after | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MQTT-DDoS-Publish_Flood | before | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MQTT-DDoS-Publish_Flood | after | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Cause of the seed-42 ARP_Spoofing H1 change (TRACE-MoE)

The canonical seed-42 run was replaced by the existing complete rerun (`__v3`: same configuration, code fingerprint and data fingerprint as the other seed-42 runs). Old run: 1 epoch; new run: 22 epochs (early stopping, patience 10). Measured values read from the two metrics.json files:

| Quantity | Old (1 epoch) | New (50-epoch protocol) |
|---|---|---|
| held_out_caught_any (H1 rate) | 0.9902 | 0.6408 |
| held_out_detected_as_attack | 0.2076 | 0.3036 |
| held_out_flagged_unknown (energy) | 0.9740 | 0.4083 |
| known_classes_macro_f1 | 0.8055 | 0.8409 |
| known_false_alarm_rate | 0.0458 | 0.0302 |
| energy AUROC | 0.9818 | 0.9093 |

Observation: the better-trained model has higher known-class macro-F1 and a lower false-alarm rate, but flags far fewer held-out ARP windows as unknown by energy (the fixed energy threshold is calibrated on known data), so the safety-view catch rate falls. Interpretation (not tested here): a longer-trained model fits the known classes more tightly, which can move ARP windows closer to known-class energy values. The other four seeds for this class (43 to 46) are above 0.98. The run is retained as it is; no rerun or exclusion was made.
