# U1 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U1
Script SHA-256: 33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: 75. rng seed 42, bootstrap 10000 over the five seeds.
Started 2026-11-21 19:16:34, ended 2026-11-21 19:16:36.

Held-out windows = test windows whose true 19-class label is the held-out class; shares of predicted 19-class labels and 6-class categories (mean over seeds with seed interval). Checks: shares sum to 1 (max deviation 2.2e-16); binary-head attack share equals A2_binary_rate (max difference 1.1e-16).

Top three absorbing classes:

| model       | held_out_class          | predicted_class        | is_sibling   | is_benign   |   share |   share_lo |   share_hi |   n_seeds |
|:------------|:------------------------|:-----------------------|:-------------|:------------|--------:|-----------:|-----------:|----------:|
| trace_moe   | TCP_IP-DoS-UDP          | TCP_IP-DDoS-UDP        | False        | False       |  0.9999 |     0.9999 |     0.9999 |         5 |
| transformer | TCP_IP-DoS-UDP          | TCP_IP-DDoS-UDP        | False        | False       |  0.9999 |     0.9999 |     0.9999 |         5 |
| trace_moe   | MQTT-DDoS-Publish_Flood | MQTT-DoS-Publish_Flood | True         | False       |  0.9852 |     0.9658 |     0.9977 |         5 |
| transformer | MQTT-DDoS-Publish_Flood | MQTT-DoS-Publish_Flood | True         | False       |  0.984  |     0.9718 |     0.9959 |         5 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | TCP_IP-DDoS-UDP        | False        | False       |  0.9819 |     0.9671 |     0.9923 |         5 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | MQTT-DoS-Publish_Flood | True         | False       |  0.8766 |     0.8543 |     0.8942 |         5 |
| transformer | ARP_Spoofing            | Benign                 | False        | True        |  0.5917 |     0.5318 |     0.6821 |         5 |
| trace_moe   | ARP_Spoofing            | Benign                 | False        | True        |  0.5898 |     0.5397 |     0.6285 |         5 |
| transformer | MQTT-Malformed_Data     | Benign                 | False        | True        |  0.4868 |     0.405  |     0.5745 |         5 |
| trace_moe   | MQTT-Malformed_Data     | Benign                 | False        | True        |  0.4833 |     0.4651 |     0.5014 |         5 |
| bilstm_dnn  | ARP_Spoofing            | Recon-Port_Scan        | False        | False       |  0.4198 |     0.4138 |     0.427  |         5 |
| bilstm_dnn  | MQTT-Malformed_Data     | Benign                 | False        | True        |  0.408  |     0.3486 |     0.4492 |         5 |
| trace_moe   | Recon-VulScan           | Benign                 | False        | True        |  0.3807 |     0.365  |     0.4041 |         5 |
| trace_moe   | ARP_Spoofing            | Recon-Port_Scan        | False        | False       |  0.338  |     0.3135 |     0.3659 |         5 |
| bilstm_dnn  | MQTT-Malformed_Data     | ARP_Spoofing           | False        | False       |  0.318  |     0.2742 |     0.3756 |         5 |
| transformer | Recon-VulScan           | Benign                 | False        | True        |  0.3146 |     0.2722 |     0.3563 |         5 |
| transformer | ARP_Spoofing            | Recon-Port_Scan        | False        | False       |  0.3035 |     0.2302 |     0.3491 |         5 |
| transformer | Recon-VulScan           | Recon-Port_Scan        | True         | False       |  0.2959 |     0.2705 |     0.312  |         5 |
| bilstm_dnn  | Recon-VulScan           | Recon-OS_Scan          | True         | False       |  0.2781 |     0.2598 |     0.2965 |         5 |
| trace_moe   | Recon-VulScan           | Recon-Port_Scan        | True         | False       |  0.2748 |     0.2579 |     0.2911 |         5 |
| trace_moe   | Recon-VulScan           | Recon-OS_Scan          | True         | False       |  0.2719 |     0.2665 |     0.278  |         5 |
| bilstm_dnn  | Recon-VulScan           | Benign                 | False        | True        |  0.2689 |     0.2341 |     0.3026 |         5 |
| bilstm_dnn  | ARP_Spoofing            | Recon-VulScan          | False        | False       |  0.2636 |     0.2266 |     0.2922 |         5 |
| transformer | Recon-VulScan           | Recon-OS_Scan          | True         | False       |  0.2482 |     0.2374 |     0.2575 |         5 |
| bilstm_dnn  | Recon-VulScan           | Recon-Port_Scan        | True         | False       |  0.2465 |     0.2358 |     0.2581 |         5 |
| bilstm_dnn  | ARP_Spoofing            | Benign                 | False        | True        |  0.2349 |     0.1907 |     0.2791 |         5 |
| transformer | MQTT-Malformed_Data     | ARP_Spoofing           | False        | False       |  0.2208 |     0.154  |     0.3106 |         5 |
| transformer | MQTT-Malformed_Data     | Recon-OS_Scan          | False        | False       |  0.2109 |     0.1969 |     0.2229 |         5 |
| trace_moe   | MQTT-Malformed_Data     | ARP_Spoofing           | False        | False       |  0.2062 |     0.1917 |     0.229  |         5 |
| trace_moe   | MQTT-Malformed_Data     | Recon-OS_Scan          | False        | False       |  0.1738 |     0.152  |     0.1992 |         5 |
| bilstm_dnn  | MQTT-Malformed_Data     | Recon-OS_Scan          | False        | False       |  0.0962 |     0.0729 |     0.1311 |         5 |
| transformer | ARP_Spoofing            | Recon-VulScan          | False        | False       |  0.0802 |     0.0652 |     0.1016 |         5 |
| trace_moe   | ARP_Spoofing            | Recon-VulScan          | False        | False       |  0.0586 |     0.0503 |     0.0688 |         5 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | TCP_IP-DoS-SYN         | False        | False       |  0.0503 |     0.0284 |     0.0752 |         5 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | TCP_IP-DoS-TCP         | False        | False       |  0.0301 |     0.0094 |     0.0582 |         5 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | TCP_IP-DoS-ICMP        | True         | False       |  0.0139 |     0.0044 |     0.0296 |         5 |
| transformer | MQTT-DDoS-Publish_Flood | Recon-Port_Scan        | False        | False       |  0.0105 |     0.0012 |     0.0225 |         5 |
| trace_moe   | MQTT-DDoS-Publish_Flood | TCP_IP-DDoS-SYN        | False        | False       |  0.0094 |     0      |     0.0282 |         5 |
| trace_moe   | MQTT-DDoS-Publish_Flood | MQTT-DoS-Connect_Flood | True         | False       |  0.0024 |     0.0007 |     0.0041 |         5 |
| transformer | MQTT-DDoS-Publish_Flood | MQTT-DoS-Connect_Flood | True         | False       |  0.0022 |     0.0009 |     0.0035 |         5 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | TCP_IP-DoS-TCP         | True         | False       |  0.0021 |     0.001  |     0.0037 |         5 |
| transformer | TCP_IP-DoS-UDP          | TCP_IP-DoS-SYN         | True         | False       |  0      |     0      |     0      |         5 |
| trace_moe   | TCP_IP-DoS-UDP          | TCP_IP-DoS-ICMP        | True         | False       |  0      |     0      |     0      |         5 |
| trace_moe   | TCP_IP-DoS-UDP          | TCP_IP-DoS-SYN         | True         | False       |  0      |     0      |     0      |         5 |
| transformer | TCP_IP-DoS-UDP          | ARP_Spoofing           | False        | False       |  0      |     0      |     0      |         5 |

Share absorbed by the held-out class's own 6-category (siblings plus other classes of that category):

| model       | held_out_class          |   share |
|:------------|:------------------------|--------:|
| bilstm_dnn  | ARP_Spoofing            |   0     |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood |   0.905 |
| bilstm_dnn  | MQTT-Malformed_Data     |   0.046 |
| bilstm_dnn  | Recon-VulScan           |   0.531 |
| bilstm_dnn  | TCP_IP-DoS-UDP          |   0.017 |
| trace_moe   | ARP_Spoofing            |   0     |
| trace_moe   | MQTT-DDoS-Publish_Flood |   0.988 |
| trace_moe   | MQTT-Malformed_Data     |   0     |
| trace_moe   | Recon-VulScan           |   0.552 |
| trace_moe   | TCP_IP-DoS-UDP          |   0     |
| transformer | ARP_Spoofing            |   0     |
| transformer | MQTT-DDoS-Publish_Flood |   0.988 |
| transformer | MQTT-Malformed_Data     |   0.015 |
| transformer | Recon-VulScan           |   0.55  |
| transformer | TCP_IP-DoS-UDP          |   0     |
