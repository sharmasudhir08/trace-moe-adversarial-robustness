# H3 — TRACE-MoE energy AUROC vs baselines

**Charter rule**: paired bootstrap of (trace_moe.energy_auroc − baseline.energy_auroc).
- SUPPORTED (per-baseline overall): ≥ 3 of 5 class pairs SUPPORTED
- NOT SUPPORTED (per-baseline overall): ≥ 3 of 5 class pairs NOT SUPPORTED
- INCONCLUSIVE (per-baseline overall): ≥ 3 of 5 class pairs INCONCLUSIVE
- MIXED (per-baseline overall): otherwise

**Per-baseline overall**:
- vs `transformer`: **NOT SUPPORTED** (SUPPORTED 1, INC 1, NOT 3)
- vs `bilstm_dnn`: **SUPPORTED (strong)** (SUPPORTED 3, INC 1, NOT 1)

## Per (baseline × class) pairs

| Baseline | Class | diff_s42 | diff_s43 | diff_s44 | diff_s45 | diff_s46 | mean_diff | CI 95% | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| transformer | ARP_Spoofing | -0.0759 | +0.0014 | +0.0013 | -0.0070 | -0.0010 | -0.0163 | [-0.0467, +0.0009] | NOT SUPPORTED |
| transformer | Recon-VulScan | -0.0125 | -0.0306 | -0.0059 | +0.0145 | +0.0089 | -0.0051 | [-0.0191, +0.0082] | NOT SUPPORTED |
| transformer | MQTT-Malformed_Data | +0.0049 | -0.0008 | +0.0124 | -0.0182 | -0.0011 | -0.0006 | [-0.0102, +0.0071] | NOT SUPPORTED |
| transformer | TCP_IP-DoS-UDP | -0.6462 | -0.4002 | -0.6286 | -0.4546 | +0.3624 | -0.3535 | [-0.6009, +0.0117] | INCONCLUSIVE |
| transformer | MQTT-DDoS-Publish_Flood | +0.4025 | +0.1320 | +0.0394 | +0.2257 | +0.2492 | +0.2097 | [+0.1120, +0.3130] | SUPPORTED |
| bilstm_dnn | ARP_Spoofing | +0.0076 | +0.2106 | +0.0661 | +0.1044 | +0.1534 | +0.1084 | [+0.0484, +0.1700] | SUPPORTED |
| bilstm_dnn | Recon-VulScan | +0.1585 | +0.2478 | +0.3354 | +0.3323 | +0.1325 | +0.2413 | [+0.1660, +0.3166] | SUPPORTED |
| bilstm_dnn | MQTT-Malformed_Data | +0.0882 | +0.0335 | +0.0104 | +0.0398 | +0.2046 | +0.0753 | [+0.0255, +0.1424] | SUPPORTED |
| bilstm_dnn | TCP_IP-DoS-UDP | -0.2774 | -0.4894 | -0.6687 | -0.5854 | +0.2651 | -0.3511 | [-0.5995, -0.0301] | NOT SUPPORTED |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | +0.1314 | -0.0648 | -0.3751 | +0.1521 | +0.1137 | -0.0085 | [-0.2076, +0.1361] | INCONCLUSIVE |