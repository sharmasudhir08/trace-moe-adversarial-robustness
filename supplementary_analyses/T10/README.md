# T10 flood-class heterogeneity (post-hoc exploratory)

Script `scripts/p4_rev_t10_flood_clusters.py` SHA-256 31e7072943d58df97f7f4ebc1e82677d1c16dc90dfcb418ba225e751532306c0; Python 3.12.4, numpy 2.5.3; rng 42; 2026-11-21 12:28:43.

Pass check: max |all-window AUROC - A7_energy_auroc| over 30 runs = 8.33e-17 (criterion 1e-9).

Mean silhouette by class and k:

| held_out_class          |   k |   silhouette |
|:------------------------|----:|-------------:|
| MQTT-DDoS-Publish_Flood |   2 |       0.3657 |
| MQTT-DDoS-Publish_Flood |   3 |       0.3305 |
| MQTT-DDoS-Publish_Flood |   4 |       0.3619 |
| MQTT-DDoS-Publish_Flood |   5 |       0.3209 |
| TCP_IP-DoS-UDP          |   2 |       0.8532 |
| TCP_IP-DoS-UDP          |   3 |       0.4448 |
| TCP_IP-DoS-UDP          |   4 |       0.456  |
| TCP_IP-DoS-UDP          |   5 |       0.4644 |

Files: `clusters.csv` (cluster by source file), `auroc_by_cluster.csv`, `silhouette.csv`, `figures/`. Whatever is found is reported; a null result is acceptable.
