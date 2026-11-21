# U7 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch2.py --date 20261121 --tasks U7
Script SHA-256: f664f9ee387f9f0e6dff3eeedc9ae53e734cbbe00269a42d1147eb7f21f43aaf
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz, part2/meta_cache (rebuilt test-window metadata). Runs read: 75. rng seed 42, bootstrap 10000 over seeds.
Started 2026-11-21 19:37:39, ended 2026-11-21 19:38:13.

Windows common to the five seeds of a model and class (test windows differ slightly between seeds because of duplicate removal; matched by global row id of the last row; held-out labels checked equal). Ensemble = mean of energy z-scores (each seed: its own known validation energy mean and SD); AUROC held-out against known on the common windows, compared with the mean of the five single-seed AUROCs on the same windows. A held-out window is missed by a seed if its energy is above that seed's threshold. Single combined run per model and class, so no seed interval; this is a descriptive ensemble note, not a new detector.

Ensemble:

| model       | held_out_class          |   n_common_windows |   n_heldout_common |   auroc_ensemble |   mean_single_seed_auroc_same_windows |   best_single |   worst_single |   ensemble_minus_mean_single |
|:------------|:------------------------|-------------------:|-------------------:|-----------------:|--------------------------------------:|--------------:|---------------:|-----------------------------:|
| trace_moe   | ARP_Spoofing            |            1582155 |               1729 |           0.9901 |                                0.9658 |        0.9884 |         0.9092 |                       0.0243 |
| trace_moe   | MQTT-DDoS-Publish_Flood |            1581124 |               8401 |           0.2912 |                                0.3257 |        0.4908 |         0.1571 |                      -0.0345 |
| trace_moe   | MQTT-Malformed_Data     |            1580921 |               1732 |           0.9901 |                                0.981  |        0.987  |         0.9688 |                       0.0091 |
| trace_moe   | Recon-VulScan           |            1580497 |               1016 |           0.9809 |                                0.9692 |        0.9804 |         0.9461 |                       0.0117 |
| trace_moe   | TCP_IP-DoS-UDP          |            1581597 |             137538 |           0.3005 |                                0.3142 |        0.7327 |         0.0575 |                      -0.0138 |
| transformer | ARP_Spoofing            |            1582155 |               1729 |           0.991  |                                0.9821 |        0.9893 |         0.9742 |                       0.0089 |
| transformer | MQTT-DDoS-Publish_Flood |            1581124 |               8401 |           0.0764 |                                0.1159 |        0.1803 |         0.0251 |                      -0.0395 |
| transformer | MQTT-Malformed_Data     |            1580921 |               1732 |           0.984  |                                0.9816 |        0.987  |         0.9704 |                       0.0024 |
| transformer | Recon-VulScan           |            1580497 |               1016 |           0.9787 |                                0.9743 |        0.9811 |         0.9659 |                       0.0044 |
| transformer | TCP_IP-DoS-UDP          |            1581597 |             137538 |           0.6619 |                                0.6667 |        0.843  |         0.3696 |                      -0.0048 |
| bilstm_dnn  | ARP_Spoofing            |            1582155 |               1729 |           0.8799 |                                0.8574 |        0.9202 |         0.7674 |                       0.0225 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood |            1581124 |               8401 |           0.1971 |                                0.334  |        0.5939 |         0.1799 |                      -0.1369 |
| bilstm_dnn  | MQTT-Malformed_Data     |            1580921 |               1732 |           0.9606 |                                0.9059 |        0.9724 |         0.7818 |                       0.0547 |
| bilstm_dnn  | Recon-VulScan           |            1580497 |               1016 |           0.8128 |                                0.7279 |        0.847  |         0.6359 |                       0.0849 |
| bilstm_dnn  | TCP_IP-DoS-UDP          |            1581597 |             137538 |           0.7817 |                                0.665  |        0.9211 |         0.4677 |                       0.1166 |

Miss overlap:

| model       | held_out_class          |   mean_miss_rate |   mean_pairwise_jaccard |   share_missed_by_all_5 |   share_missed_by_any |   expected_share_all_5_if_independent |
|:------------|:------------------------|-----------------:|------------------------:|------------------------:|----------------------:|--------------------------------------:|
| trace_moe   | ARP_Spoofing            |           0.1828 |                  0.1428 |                  0.0075 |                0.6703 |                                0      |
| trace_moe   | MQTT-DDoS-Publish_Flood |           0.9775 |                  0.9646 |                  0.9225 |                0.9999 |                                0.8916 |
| trace_moe   | MQTT-Malformed_Data     |           0.1245 |                  0.1822 |                  0.0064 |                0.3585 |                                0      |
| trace_moe   | Recon-VulScan           |           0.211  |                  0.5145 |                  0.1024 |                0.4252 |                                0.0003 |
| trace_moe   | TCP_IP-DoS-UDP          |           0.9999 |                  0.9999 |                  0.9997 |                0.9999 |                                0.9995 |
| transformer | ARP_Spoofing            |           0.0346 |                  0.3802 |                  0.011  |                0.0769 |                                0      |
| transformer | MQTT-DDoS-Publish_Flood |           0.9776 |                  0.9702 |                  0.9393 |                0.9995 |                                0.8922 |
| transformer | MQTT-Malformed_Data     |           0.0487 |                  0.0197 |                  0      |                0.2159 |                                0      |
| transformer | Recon-VulScan           |           0.1368 |                  0.5279 |                  0.0561 |                0.2392 |                                0      |
| transformer | TCP_IP-DoS-UDP          |           0.9999 |                  0.9999 |                  0.9998 |                1      |                                0.9996 |
| bilstm_dnn  | ARP_Spoofing            |           0.3087 |                  0.5814 |                  0.1799 |                0.5211 |                                0.0024 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood |           0.8759 |                  0.8931 |                  0.7849 |                0.9819 |                                0.5139 |
| bilstm_dnn  | MQTT-Malformed_Data     |           0.4117 |                  0.2768 |                  0.0566 |                0.9931 |                                0.0044 |
| bilstm_dnn  | Recon-VulScan           |           0.6419 |                  0.5727 |                  0.2746 |                0.9626 |                                0.1031 |
| bilstm_dnn  | TCP_IP-DoS-UDP          |           0.8797 |                  0.791  |                  0.5322 |                0.9974 |                                0.5036 |
