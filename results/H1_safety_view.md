# H1 — Safety-view detection rate (TRACE-MoE caught_any ≥ 0.95)

**Charter rule**: four-tier verdict.
- SUPPORTED (strong): mean ≥ 0.95 AND CI lower bound ≥ 0.90
- SUPPORTED (preliminary): mean ≥ 0.95, CI lower bound < 0.90
- INCONCLUSIVE: mean < 0.95, CI straddles 0.95
- NOT SUPPORTED: CI upper bound < 0.95

**Overall verdict**: `MIXED`

## Per class (TRACE-MoE caught_any)

| Class | seed42 | seed43 | seed44 | seed45 | seed46 | mean | CI 95% | CI width | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| ARP_Spoofing | 0.6408 | 0.9988 | 1.0000 | 0.9983 | 0.9815 | 0.9239 | [0.7808, 0.9992] | 0.2184 | INCONCLUSIVE |
| Recon-VulScan | 0.9990 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | [0.9994, 1.0000] | 0.0006 | SUPPORTED (strong) |
| MQTT-Malformed_Data | 1.0000 | 0.9913 | 1.0000 | 0.9792 | 1.0000 | 0.9941 | [0.9858, 1.0000] | 0.0142 | SUPPORTED (strong) |
| TCP_IP-DoS-UDP | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] | 0.0000 | SUPPORTED (strong) |
| MQTT-DDoS-Publish_Flood | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [0.9999, 1.0000] | 0.0001 | SUPPORTED (strong) |