# T6 cluster bootstrap (post-hoc exploratory)

Script `scripts/p4_rev_t6_cluster_bootstrap.py` SHA-256 1c85d860f00d1a47f84909ff088bac3b525c780bc91b824ec924ca43be799ed2; Python 3.12.4, numpy 2.5.3; rng 42; 2000 resamples; 2026-11-21 12:30:10.

Checks: max |binned AUROC - exact AUROC| over all runs = 1.00e-04; max |point H1 mean - H1_safety_view.csv| = 1.11e-16; max |point H5 mean - twin_h5_verdict.csv| = 8.35e-06 (H5 table is rounded to 6 decimals).
Seed-only intervals use 2,000 resamples here and may differ slightly from the paper's 10,000-resample intervals (rng stream differs).

| held_out_class          | quantity                        |   point_mean_over_seeds |   seed_only_lo |   seed_only_hi |   n_heldout_files |   n_known_files |   window_lo |   window_hi |   cluster_lo |   cluster_hi |
|:------------------------|:--------------------------------|------------------------:|---------------:|---------------:|------------------:|----------------:|------------:|------------:|-------------:|-------------:|
| ARP_Spoofing            | H1_caught_any_trace_moe         |                  0.9239 |         0.7808 |         0.9992 |                 1 |              20 |      0.78   |      0.9994 |          nan |          nan |
| ARP_Spoofing            | H5_pcrp_auroc                   |                  0.9215 |         0.9196 |         0.923  |                 1 |              20 |      0.9177 |      0.9251 |          nan |          nan |
| ARP_Spoofing            | H2_energy_minus_msp_trace_moe   |                 -0.0027 |        -0.0276 |         0.0174 |                 1 |              20 |     -0.0274 |      0.0167 |          nan |          nan |
| ARP_Spoofing            | H2_energy_minus_msp_transformer |                  0.0207 |         0.0109 |         0.0304 |                 1 |              20 |      0.0102 |      0.0323 |          nan |          nan |
| ARP_Spoofing            | H2_energy_minus_msp_bilstm_dnn  |                 -0.1175 |        -0.1692 |        -0.0693 |                 1 |              20 |     -0.1708 |     -0.0682 |          nan |          nan |
| Recon-VulScan           | H1_caught_any_trace_moe         |                  0.9998 |         0.9994 |         1      |                 1 |              20 |      0.9988 |      1      |          nan |          nan |
| Recon-VulScan           | H5_pcrp_auroc                   |                  0.7656 |         0.7642 |         0.7674 |                 1 |              20 |      0.7561 |      0.7752 |          nan |          nan |
| Recon-VulScan           | H2_energy_minus_msp_trace_moe   |                 -0.0023 |        -0.007  |         0.0031 |                 1 |              20 |     -0.0078 |      0.0032 |          nan |          nan |
| Recon-VulScan           | H2_energy_minus_msp_transformer |                  0.0398 |         0.0175 |         0.0683 |                 1 |              20 |      0.0172 |      0.0685 |          nan |          nan |
| Recon-VulScan           | H2_energy_minus_msp_bilstm_dnn  |                 -0.1237 |        -0.1775 |        -0.0677 |                 1 |              20 |     -0.1814 |     -0.0648 |          nan |          nan |
| MQTT-Malformed_Data     | H1_caught_any_trace_moe         |                  0.9941 |         0.9858 |         1      |                 1 |              20 |      0.9855 |      1      |          nan |          nan |
| MQTT-Malformed_Data     | H5_pcrp_auroc                   |                  0.8879 |         0.8859 |         0.8896 |                 1 |              20 |      0.8846 |      0.8913 |          nan |          nan |
| MQTT-Malformed_Data     | H2_energy_minus_msp_trace_moe   |                 -0.0006 |        -0.0067 |         0.0054 |                 1 |              20 |     -0.0067 |      0.0058 |          nan |          nan |
| MQTT-Malformed_Data     | H2_energy_minus_msp_transformer |                  0.0261 |         0.0082 |         0.0502 |                 1 |              20 |      0.0083 |      0.051  |          nan |          nan |
| MQTT-Malformed_Data     | H2_energy_minus_msp_bilstm_dnn  |                 -0.0649 |        -0.1194 |        -0.0217 |                 1 |              20 |     -0.1213 |     -0.0201 |          nan |          nan |
| TCP_IP-DoS-UDP          | H1_caught_any_trace_moe         |                  1      |         1      |         1      |                 1 |              20 |      1      |      1      |          nan |          nan |
| TCP_IP-DoS-UDP          | H5_pcrp_auroc                   |                  0.9513 |         0.9314 |         0.9712 |                 1 |              20 |      0.9313 |      0.9711 |          nan |          nan |
| TCP_IP-DoS-UDP          | H2_energy_minus_msp_trace_moe   |                 -0.1286 |        -0.252  |        -0.0198 |                 1 |              20 |     -0.2523 |     -0.02   |          nan |          nan |
| TCP_IP-DoS-UDP          | H2_energy_minus_msp_transformer |                  0.0888 |         0.0051 |         0.2091 |                 1 |              20 |      0.005  |      0.2078 |          nan |          nan |
| TCP_IP-DoS-UDP          | H2_energy_minus_msp_bilstm_dnn  |                 -0.0775 |        -0.2272 |         0.0612 |                 1 |              20 |     -0.2273 |      0.0615 |          nan |          nan |
| MQTT-DDoS-Publish_Flood | H1_caught_any_trace_moe         |                  1      |         0.9999 |         1      |                 1 |              20 |      0.9999 |      1      |          nan |          nan |
| MQTT-DDoS-Publish_Flood | H5_pcrp_auroc                   |                  0.9361 |         0.936  |         0.9363 |                 1 |              20 |      0.9356 |      0.9366 |          nan |          nan |
| MQTT-DDoS-Publish_Flood | H2_energy_minus_msp_trace_moe   |                 -0.0537 |        -0.1933 |         0.0858 |                 1 |              20 |     -0.1933 |      0.0872 |          nan |          nan |
| MQTT-DDoS-Publish_Flood | H2_energy_minus_msp_transformer |                 -0.1811 |        -0.2938 |        -0.0616 |                 1 |              20 |     -0.2934 |     -0.0637 |          nan |          nan |
| MQTT-DDoS-Publish_Flood | H2_energy_minus_msp_bilstm_dnn  |                 -0.1082 |        -0.2703 |         0.0522 |                 1 |              20 |     -0.2708 |      0.0496 |          nan |          nan |

File counts per class: see `file_counts.csv`. Classes with fewer than 2 held-out files have no cluster interval.
