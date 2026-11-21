# T11 Charter ESA2 (Spearman, energy vs 1 - MSP on held-out windows)

Script `scripts/p4_rev_saved_scores.py` SHA-256 5189b8f69c91518855135ad90430046203c8e50cd04ff83bc4a7b7c20153d0b1; Python 3.12.4, numpy 2.5.3, pandas 3.0.6; rng 42; runs read: 75 (expected 75); 2026-11-21 11:54:41.

Sign convention chosen before computing: both scores oriented higher = more unknown (-energy vs 1 - MSP). Rule: Charter v1.4 three-tier, 0.9 threshold, bootstrap 10,000 over seeds.

| model       | held_out_class          |   n_seeds |   mean_rho |   ci_lo |   ci_hi | esa2_tier        |
|:------------|:------------------------|----------:|-----------:|--------:|--------:|:-----------------|
| bilstm_dnn  | ARP_Spoofing            |         5 |     0.6369 |  0.4914 |  0.7398 | RANKING IMPROVES |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood |         5 |     0.7183 |  0.6761 |  0.7728 | RANKING IMPROVES |
| bilstm_dnn  | MQTT-Malformed_Data     |         5 |     0.6659 |  0.5849 |  0.7548 | RANKING IMPROVES |
| bilstm_dnn  | Recon-VulScan           |         5 |     0.5788 |  0.3781 |  0.7386 | RANKING IMPROVES |
| bilstm_dnn  | TCP_IP-DoS-UDP          |         5 |     0.9093 |  0.8465 |  0.9667 | INCONCLUSIVE     |
| trace_moe   | ARP_Spoofing            |         5 |     0.8255 |  0.7882 |  0.8639 | RANKING IMPROVES |
| trace_moe   | MQTT-DDoS-Publish_Flood |         5 |     0.394  |  0.2415 |  0.5745 | RANKING IMPROVES |
| trace_moe   | MQTT-Malformed_Data     |         5 |     0.7776 |  0.7285 |  0.836  | RANKING IMPROVES |
| trace_moe   | Recon-VulScan           |         5 |     0.7782 |  0.7399 |  0.8161 | RANKING IMPROVES |
| trace_moe   | TCP_IP-DoS-UDP          |         5 |     0.4761 |  0.1638 |  0.7885 | RANKING IMPROVES |
| transformer | ARP_Spoofing            |         5 |     0.7436 |  0.5449 |  0.8684 | RANKING IMPROVES |
| transformer | MQTT-DDoS-Publish_Flood |         5 |     0.7822 |  0.607  |  0.8871 | RANKING IMPROVES |
| transformer | MQTT-Malformed_Data     |         5 |     0.7424 |  0.6903 |  0.8025 | RANKING IMPROVES |
| transformer | Recon-VulScan           |         5 |     0.83   |  0.7862 |  0.8738 | RANKING IMPROVES |
| transformer | TCP_IP-DoS-UDP          |         5 |     0.8205 |  0.6153 |  0.9657 | INCONCLUSIVE     |

Reported directionally; no overall ESA2 claim at this scope.
