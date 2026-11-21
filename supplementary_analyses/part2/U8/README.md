# U8 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch2.py --date 20261121 --tasks U8
Script SHA-256: f664f9ee387f9f0e6dff3eeedc9ae53e734cbbe00269a42d1147eb7f21f43aaf
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz, part2/meta_cache (rebuilt test-window metadata). Runs read: 75. rng seed 42, bootstrap 10000 over seeds.
Started 2026-11-21 19:37:20, ended 2026-11-21 19:37:27.

Each held-out class has one test file; held-out windows are taken in file order (row_in_file). first_flag_position = number of held-out windows before the first flag (0 = the first window is flagged); -1 = never flagged. Descriptive only; no window-time scale is available (stride 1, no timestamps used). ttd_cdf.pdf: share of the 25 runs per model whose first flag falls within x windows (never-flagged runs stay below 1).

| model       | held_out_class          | flag        |   runs |   ever_flagged |   share_flagged |   first_pos_median |   first_pos_max |
|:------------|:------------------------|:------------|-------:|---------------:|----------------:|-------------------:|----------------:|
| bilstm_dnn  | ARP_Spoofing            | binary_head |      5 |              1 |          0.4287 |                  0 |               2 |
| bilstm_dnn  | ARP_Spoofing            | either      |      5 |              1 |          0.9323 |                  0 |               0 |
| bilstm_dnn  | ARP_Spoofing            | energy      |      5 |              1 |          0.6913 |                  1 |               1 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | binary_head |      5 |              1 |          0.9998 |                  0 |               0 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | either      |      5 |              1 |          1      |                  0 |               0 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | energy      |      5 |              1 |          0.1241 |                292 |             406 |
| bilstm_dnn  | MQTT-Malformed_Data     | binary_head |      5 |              1 |          0.3624 |                  0 |               0 |
| bilstm_dnn  | MQTT-Malformed_Data     | either      |      5 |              1 |          0.812  |                  0 |               0 |
| bilstm_dnn  | MQTT-Malformed_Data     | energy      |      5 |              1 |          0.5883 |                  1 |              23 |
| bilstm_dnn  | Recon-VulScan           | binary_head |      5 |              1 |          0.5496 |                  5 |               6 |
| bilstm_dnn  | Recon-VulScan           | either      |      5 |              1 |          0.8419 |                  0 |               4 |
| bilstm_dnn  | Recon-VulScan           | energy      |      5 |              1 |          0.3581 |                  0 |               4 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | binary_head |      5 |              1 |          1      |                  0 |               0 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | either      |      5 |              1 |          1      |                  0 |               0 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | energy      |      5 |              1 |          0.1203 |                 19 |              29 |
| trace_moe   | ARP_Spoofing            | binary_head |      5 |              1 |          0.333  |                  1 |               1 |
| trace_moe   | ARP_Spoofing            | either      |      5 |              1 |          0.9239 |                  0 |               1 |
| trace_moe   | ARP_Spoofing            | energy      |      5 |              1 |          0.8172 |                  0 |               4 |
| trace_moe   | MQTT-DDoS-Publish_Flood | binary_head |      5 |              1 |          0.9998 |                  0 |               0 |
| trace_moe   | MQTT-DDoS-Publish_Flood | either      |      5 |              1 |          1      |                  0 |               0 |
| trace_moe   | MQTT-DDoS-Publish_Flood | energy      |      5 |              1 |          0.0225 |                536 |             658 |
| trace_moe   | MQTT-Malformed_Data     | binary_head |      5 |              1 |          0.3587 |                  0 |               3 |
| trace_moe   | MQTT-Malformed_Data     | either      |      5 |              1 |          0.9941 |                  0 |               0 |
| trace_moe   | MQTT-Malformed_Data     | energy      |      5 |              1 |          0.8755 |                  0 |               0 |
| trace_moe   | Recon-VulScan           | binary_head |      5 |              1 |          0.5608 |                  4 |               5 |
| trace_moe   | Recon-VulScan           | either      |      5 |              1 |          0.9998 |                  0 |               0 |
| trace_moe   | Recon-VulScan           | energy      |      5 |              1 |          0.789  |                  0 |               1 |
| trace_moe   | TCP_IP-DoS-UDP          | binary_head |      5 |              1 |          1      |                  0 |               0 |
| trace_moe   | TCP_IP-DoS-UDP          | either      |      5 |              1 |          1      |                  0 |               0 |
| trace_moe   | TCP_IP-DoS-UDP          | energy      |      5 |              1 |          0.0001 |              54688 |          101735 |
| transformer | ARP_Spoofing            | binary_head |      5 |              1 |          0.2704 |                 13 |              38 |
| transformer | ARP_Spoofing            | either      |      5 |              1 |          0.998  |                  0 |               0 |
| transformer | ARP_Spoofing            | energy      |      5 |              1 |          0.9654 |                  0 |               0 |
| transformer | MQTT-DDoS-Publish_Flood | binary_head |      5 |              1 |          0.9998 |                  0 |               0 |
| transformer | MQTT-DDoS-Publish_Flood | either      |      5 |              1 |          1      |                  0 |               0 |
| transformer | MQTT-DDoS-Publish_Flood | energy      |      5 |              1 |          0.0224 |                533 |             559 |
| transformer | MQTT-Malformed_Data     | binary_head |      5 |              1 |          0.3367 |                  0 |               0 |
| transformer | MQTT-Malformed_Data     | either      |      5 |              1 |          0.9991 |                  0 |               0 |
| transformer | MQTT-Malformed_Data     | energy      |      5 |              1 |          0.9513 |                  0 |               1 |
| transformer | Recon-VulScan           | binary_head |      5 |              1 |          0.5864 |                  1 |               4 |
| transformer | Recon-VulScan           | either      |      5 |              1 |          0.9969 |                  0 |               0 |
| transformer | Recon-VulScan           | energy      |      5 |              1 |          0.8632 |                  0 |               0 |
| transformer | TCP_IP-DoS-UDP          | binary_head |      5 |              1 |          1      |                  0 |               0 |
| transformer | TCP_IP-DoS-UDP          | either      |      5 |              1 |          1      |                  0 |               0 |
| transformer | TCP_IP-DoS-UDP          | energy      |      5 |              1 |          0.0001 |             101733 |          101737 |
