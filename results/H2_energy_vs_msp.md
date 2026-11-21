# H2 — Energy abstention vs MSP AUROC

**Charter rule**: paired bootstrap 95% CI of (energy_auroc − msp_auroc).
- SUPPORTED: CI excludes 0 AND mean > 0
- NOT SUPPORTED: CI excludes 0 AND mean < 0
- INCONCLUSIVE: CI includes 0 (CI-wider-than-margin at 5-seed scope)

**Per-model overall**:
- `trace_moe`: **NOT SUPPORTED** (SUPPORTED 0, INC 1, NOT 4)
- `transformer`: **SUPPORTED (strong)** (SUPPORTED 4, INC 0, NOT 1)
- `bilstm_dnn`: **NOT SUPPORTED** (SUPPORTED 0, INC 2, NOT 3)

**Primary H2 answer (for TRACE-MoE)**: NOT SUPPORTED

## Per (model × class) pairs

| Model | Class | diff_s42 | diff_s43 | diff_s44 | diff_s45 | diff_s46 | mean_diff | CI 95% | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | -0.0478 | +0.0014 | +0.0030 | +0.0275 | +0.0025 | -0.0027 | [-0.0276, +0.0174] | NOT SUPPORTED |
| trace_moe | Recon-VulScan | -0.0029 | -0.0103 | -0.0045 | +0.0071 | -0.0008 | -0.0023 | [-0.0070, +0.0032] | NOT SUPPORTED |
| trace_moe | MQTT-Malformed_Data | +0.0014 | -0.0066 | +0.0067 | -0.0108 | +0.0062 | -0.0006 | [-0.0067, +0.0055] | NOT SUPPORTED |
| trace_moe | TCP_IP-DoS-UDP | -0.3558 | -0.0716 | -0.1207 | +0.0475 | -0.1425 | -0.1286 | [-0.2563, -0.0238] | NOT SUPPORTED |
| trace_moe | MQTT-DDoS-Publish_Flood | +0.1269 | -0.2888 | -0.1322 | -0.1243 | +0.1496 | -0.0537 | [-0.1933, +0.0858] | INCONCLUSIVE |
| transformer | ARP_Spoofing | +0.0075 | +0.0358 | +0.0241 | +0.0282 | +0.0077 | +0.0207 | [+0.0109, +0.0304] | SUPPORTED |
| transformer | Recon-VulScan | +0.0885 | +0.0247 | +0.0100 | +0.0545 | +0.0215 | +0.0398 | [+0.0175, +0.0681] | SUPPORTED |
| transformer | MQTT-Malformed_Data | +0.0032 | +0.0279 | +0.0734 | +0.0191 | +0.0072 | +0.0261 | [+0.0080, +0.0510] | SUPPORTED |
| transformer | TCP_IP-DoS-UDP | +0.0042 | -0.0208 | +0.1196 | +0.3073 | +0.0336 | +0.0888 | [+0.0010, +0.2091] | SUPPORTED |
| transformer | MQTT-DDoS-Publish_Flood | -0.0681 | +0.0040 | -0.2520 | -0.2138 | -0.3756 | -0.1811 | [-0.2938, -0.0684] | NOT SUPPORTED |
| bilstm_dnn | ARP_Spoofing | -0.0715 | -0.2122 | -0.0447 | -0.1183 | -0.1405 | -0.1174 | [-0.1697, -0.0701] | NOT SUPPORTED |
| bilstm_dnn | Recon-VulScan | -0.0503 | -0.1765 | -0.1797 | -0.1745 | -0.0371 | -0.1236 | [-0.1774, -0.0676] | NOT SUPPORTED |
| bilstm_dnn | MQTT-Malformed_Data | -0.0727 | -0.0351 | +0.0056 | -0.0526 | -0.1697 | -0.0649 | [-0.1228, -0.0188] | NOT SUPPORTED |
| bilstm_dnn | TCP_IP-DoS-UDP | -0.3448 | -0.1041 | +0.1755 | +0.0024 | -0.1164 | -0.0775 | [-0.2272, +0.0714] | INCONCLUSIVE |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | +0.0446 | -0.2937 | +0.1386 | -0.3339 | -0.0965 | -0.1082 | [-0.2703, +0.0540] | INCONCLUSIVE |