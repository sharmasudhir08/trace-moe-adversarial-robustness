# Paper 4 H5 verdict — twin-alone unknown-attack AUROC

**Pre-registered**: Paper 4 Charter v1.2 (DRAFT), H5 twin.
**Twin design (A2)**: 5 LOAO-specific protocol-cluster profiles; held-out class never in train.
**Score**: nearest-known-cluster Mahalanobis `D_min(x) = min_c D(x | c)`.
**Verdict rule (per pair)**: SUPPORTED if mean AUROC >= 0.80 AND CI lower >= 0.75.

## Per-pair verdicts

| Model | Held-out class | n seeds | Mean AUROC | CI_lo | CI_hi | Mean FPR@95 | Verdict |
|---|---|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | 5 | 0.9215 | 0.9196 | 0.9230 | 0.3688 | SUPPORTED |
| trace_moe | Recon-VulScan | 5 | 0.7656 | 0.7642 | 0.7674 | 0.7214 | INCONCLUSIVE |
| trace_moe | MQTT-Malformed_Data | 5 | 0.8879 | 0.8857 | 0.8896 | 0.3187 | SUPPORTED |
| trace_moe | TCP_IP-DoS-UDP | 5 | 0.9513 | 0.9314 | 0.9712 | 0.1074 | SUPPORTED |
| trace_moe | MQTT-DDoS-Publish_Flood | 5 | 0.9361 | 0.9360 | 0.9363 | 0.1326 | SUPPORTED |
| transformer | ARP_Spoofing | 5 | 0.9215 | 0.9196 | 0.9230 | 0.3688 | SUPPORTED |
| transformer | Recon-VulScan | 5 | 0.7656 | 0.7642 | 0.7674 | 0.7214 | INCONCLUSIVE |
| transformer | MQTT-Malformed_Data | 5 | 0.8879 | 0.8857 | 0.8896 | 0.3187 | SUPPORTED |
| transformer | TCP_IP-DoS-UDP | 5 | 0.9513 | 0.9314 | 0.9712 | 0.1074 | SUPPORTED |
| transformer | MQTT-DDoS-Publish_Flood | 5 | 0.9361 | 0.9360 | 0.9363 | 0.1326 | SUPPORTED |
| bilstm_dnn | ARP_Spoofing | 5 | 0.9215 | 0.9196 | 0.9230 | 0.3688 | SUPPORTED |
| bilstm_dnn | Recon-VulScan | 5 | 0.7656 | 0.7642 | 0.7674 | 0.7214 | INCONCLUSIVE |
| bilstm_dnn | MQTT-Malformed_Data | 5 | 0.8879 | 0.8857 | 0.8896 | 0.3187 | SUPPORTED |
| bilstm_dnn | TCP_IP-DoS-UDP | 5 | 0.9513 | 0.9314 | 0.9712 | 0.1074 | SUPPORTED |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | 5 | 0.9361 | 0.9360 | 0.9363 | 0.1326 | SUPPORTED |

## Overall per-model

- **trace_moe**: SUPPORTED (strong, overall)
- **transformer**: SUPPORTED (strong, overall)
- **bilstm_dnn**: SUPPORTED (strong, overall)