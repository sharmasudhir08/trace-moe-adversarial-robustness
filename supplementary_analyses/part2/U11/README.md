# U11 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U11
Script SHA-256: 33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: 75. rng seed 42, bootstrap 10000 over the five seeds.
Started 2026-11-21 19:16:36, ended 2026-11-21 19:16:44.

Degradation = full-model per-class F1 (mean over seeds, results_v2/tables/per_class_f1_official.csv) minus per-class F1 of the LOAO run on the known test windows (held-out windows excluded), positive = worse. Check: mean over the 18 known classes of the LOAO F1 equals the H4 p3c value, max difference 0.00e+00 (criterion 1e-6; PASS). The full-model baseline is the seed-mean per-class F1 (per-seed per-class values are not stored in that table), so per-seed degradations here are not identical to the H4 per-seed degradations.

Three largest mean degradations per model and held-out class:

| model       | held_out_class          | known_class             | is_sibling   |   degradation |   degradation_lo |   degradation_hi |   f1_loao |   f1_loao_lo |   f1_loao_hi |   n_seeds |
|:------------|:------------------------|:------------------------|:-------------|--------------:|-----------------:|-----------------:|----------:|-------------:|-------------:|----------:|
| transformer | MQTT-DDoS-Publish_Flood | ARP_Spoofing            | False        |        0.1099 |           0.0707 |           0.1491 |    0.3526 |       0.3134 |       0.3908 |         5 |
| trace_moe   | TCP_IP-DoS-UDP          | ARP_Spoofing            | False        |        0.0601 |          -0.0112 |           0.1266 |    0.4872 |       0.4207 |       0.5585 |         5 |
| trace_moe   | MQTT-DDoS-Publish_Flood | ARP_Spoofing            | False        |        0.0439 |          -0.0321 |           0.1355 |    0.5034 |       0.4141 |       0.5817 |         5 |
| trace_moe   | MQTT-Malformed_Data     | Recon-OS_Scan           | False        |        0.0426 |           0.021  |           0.0663 |    0.4163 |       0.3947 |       0.4389 |         5 |
| transformer | MQTT-Malformed_Data     | ARP_Spoofing            | False        |        0.0356 |          -0.063  |           0.1217 |    0.4269 |       0.3408 |       0.5243 |         5 |
| trace_moe   | Recon-VulScan           | ARP_Spoofing            | False        |        0.0316 |           0.0073 |           0.0558 |    0.5157 |       0.4915 |       0.54   |         5 |
| transformer | TCP_IP-DoS-UDP          | Recon-OS_Scan           | False        |        0.0306 |          -0.0104 |           0.0808 |    0.3713 |       0.327  |       0.4123 |         5 |
| transformer | ARP_Spoofing            | Recon-OS_Scan           | False        |        0.0278 |          -0.0346 |           0.0868 |    0.3741 |       0.3151 |       0.4365 |         5 |
| transformer | TCP_IP-DoS-UDP          | ARP_Spoofing            | False        |        0.0268 |          -0.0336 |           0.0872 |    0.4357 |       0.3753 |       0.4961 |         5 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | MQTT-Malformed_Data     | False        |        0.0222 |          -0.0156 |           0.0772 |    0.5378 |       0.4828 |       0.5756 |         5 |
| bilstm_dnn  | ARP_Spoofing            | Recon-Port_Scan         | False        |        0.0207 |          -0.0005 |           0.0372 |    0.6083 |       0.5917 |       0.6287 |         5 |
| trace_moe   | MQTT-Malformed_Data     | Recon-Port_Scan         | False        |        0.0195 |           0.0056 |           0.0327 |    0.8914 |       0.878  |       0.9053 |         5 |
| trace_moe   | TCP_IP-DoS-UDP          | Recon-Ping_Sweep        | False        |        0.0194 |          -0.0134 |           0.0723 |    0.581  |       0.5284 |       0.6138 |         5 |
| transformer | MQTT-DDoS-Publish_Flood | Benign                  | False        |        0.0193 |           0.0086 |           0.0299 |    0.9271 |       0.9165 |       0.9378 |         5 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | ARP_Spoofing            | False        |        0.0172 |           0.0105 |           0.024  |    0.2252 |       0.2184 |       0.2319 |         5 |
| bilstm_dnn  | Recon-VulScan           | Recon-Ping_Sweep        | True         |        0.0171 |           0.0009 |           0.0426 |    0.4017 |       0.3762 |       0.4177 |         5 |
| bilstm_dnn  | ARP_Spoofing            | Recon-Ping_Sweep        | False        |        0.0159 |          -0.0151 |           0.0542 |    0.4029 |       0.3646 |       0.4339 |         5 |
| bilstm_dnn  | Recon-VulScan           | MQTT-Malformed_Data     | False        |        0.0156 |          -0.0232 |           0.0791 |    0.5444 |       0.4809 |       0.5832 |         5 |
| transformer | ARP_Spoofing            | MQTT-Malformed_Data     | False        |        0.0146 |           0.002  |           0.0271 |    0.7797 |       0.7672 |       0.7923 |         5 |
| trace_moe   | MQTT-Malformed_Data     | ARP_Spoofing            | False        |        0.0133 |          -0.0833 |           0.0817 |    0.534  |       0.4656 |       0.6306 |         5 |
| transformer | TCP_IP-DoS-UDP          | Recon-Port_Scan         | False        |        0.013  |          -0.0133 |           0.0473 |    0.8481 |       0.8138 |       0.8744 |         5 |
| trace_moe   | MQTT-DDoS-Publish_Flood | Recon-Ping_Sweep        | False        |        0.013  |          -0.0127 |           0.0541 |    0.5874 |       0.5463 |       0.6131 |         5 |
| trace_moe   | TCP_IP-DoS-UDP          | Recon-OS_Scan           | False        |        0.0113 |          -0.0061 |           0.0291 |    0.4476 |       0.4298 |       0.465  |         5 |
| transformer | ARP_Spoofing            | Recon-Port_Scan         | False        |        0.0105 |          -0.0199 |           0.0408 |    0.8506 |       0.8203 |       0.881  |         5 |
| trace_moe   | Recon-VulScan           | Recon-Ping_Sweep        | True         |        0.0101 |          -0.0179 |           0.0422 |    0.5903 |       0.5582 |       0.6183 |         5 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | MQTT-DDoS-Publish_Flood | False        |        0.0088 |          -0.0079 |           0.0194 |    0.2085 |       0.1979 |       0.2252 |         5 |
| trace_moe   | MQTT-DDoS-Publish_Flood | Benign                  | False        |        0.0087 |          -0.0041 |           0.0272 |    0.9543 |       0.9358 |       0.9674 |         5 |
| bilstm_dnn  | ARP_Spoofing            | MQTT-DDoS-Publish_Flood | False        |        0.0083 |           0      |           0.0159 |    0.209  |       0.2014 |       0.2173 |         5 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | Recon-Ping_Sweep        | False        |        0.0064 |          -0.0078 |           0.0207 |    0.4124 |       0.3981 |       0.4266 |         5 |
| transformer | MQTT-Malformed_Data     | Benign                  | False        |        0.0062 |          -0.0145 |           0.027  |    0.9402 |       0.9194 |       0.9609 |         5 |
| bilstm_dnn  | MQTT-Malformed_Data     | Recon-Ping_Sweep        | False        |        0.0054 |          -0.0246 |           0.0287 |    0.4134 |       0.3895 |       0.4434 |         5 |
| bilstm_dnn  | MQTT-Malformed_Data     | MQTT-DDoS-Publish_Flood | True         |        0.0042 |          -0.0048 |           0.0116 |    0.2131 |       0.2056 |       0.2221 |         5 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | Recon-OS_Scan           | False        |        0.0032 |           0.0013 |           0.0052 |    0.0932 |       0.0912 |       0.0951 |         5 |
| bilstm_dnn  | MQTT-Malformed_Data     | MQTT-DoS-Publish_Flood  | True         |        0.0031 |          -0.0033 |           0.0066 |    0.6988 |       0.6953 |       0.7051 |         5 |
| bilstm_dnn  | Recon-VulScan           | MQTT-DoS-Publish_Flood  | False        |        0.0025 |           0.0002 |           0.0048 |    0.6994 |       0.6971 |       0.7017 |         5 |
| trace_moe   | Recon-VulScan           | MQTT-DDoS-Publish_Flood | False        |        0.0023 |          -0.0001 |           0.0055 |    0.1879 |       0.1849 |       0.1903 |         5 |
| trace_moe   | ARP_Spoofing            | Recon-Port_Scan         | False        |        0.0021 |          -0.0023 |           0.0088 |    0.9088 |       0.9024 |       0.9132 |         5 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | MQTT-DoS-Connect_Flood  | True         |        0.0018 |          -0.0011 |           0.0074 |    0.989  |       0.9834 |       0.9919 |         5 |
| trace_moe   | ARP_Spoofing            | MQTT-DDoS-Publish_Flood | False        |        0.0014 |          -0.0003 |           0.0029 |    0.1888 |       0.1873 |       0.1905 |         5 |
| trace_moe   | ARP_Spoofing            | MQTT-DoS-Connect_Flood  | False        |        0.0012 |          -0.0011 |           0.0038 |    0.9965 |       0.9939 |       0.9986 |         5 |
| transformer | Recon-VulScan           | ARP_Spoofing            | False        |        0.0011 |          -0.0416 |           0.0292 |    0.4614 |       0.4333 |       0.5041 |         5 |
| transformer | Recon-VulScan           | TCP_IP-DoS-UDP          | False        |        0.001  |          -0.0003 |           0.0025 |    0.9983 |       0.9968 |       0.9996 |         5 |
| transformer | MQTT-Malformed_Data     | MQTT-DDoS-Publish_Flood | True         |        0.0008 |          -0.0004 |           0.0027 |    0.189  |       0.1871 |       0.1902 |         5 |
| transformer | Recon-VulScan           | TCP_IP-DDoS-TCP         | False        |        0.0005 |           0      |           0.0011 |    0.9984 |       0.9978 |       0.9989 |         5 |
| transformer | MQTT-DDoS-Publish_Flood | MQTT-DDoS-Connect_Flood | True         |        0.0002 |          -0.0002 |           0.0008 |    0.9994 |       0.9988 |       0.9998 |         5 |

Mean degradation siblings vs others:

| model       | held_out_class          | is_sibling   |   degradation |
|:------------|:------------------------|:-------------|--------------:|
| bilstm_dnn  | ARP_Spoofing            | False        |        0.0006 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | False        |       -0.0016 |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | True         |       -0.0731 |
| bilstm_dnn  | MQTT-Malformed_Data     | False        |       -0.0023 |
| bilstm_dnn  | MQTT-Malformed_Data     | True         |        0.0015 |
| bilstm_dnn  | Recon-VulScan           | False        |       -0.0017 |
| bilstm_dnn  | Recon-VulScan           | True         |       -0.0219 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | False        |        0.0023 |
| bilstm_dnn  | TCP_IP-DoS-UDP          | True         |        0.0003 |
| trace_moe   | ARP_Spoofing            | False        |       -0.0184 |
| trace_moe   | MQTT-DDoS-Publish_Flood | False        |        0.0035 |
| trace_moe   | MQTT-DDoS-Publish_Flood | True         |       -0.0752 |
| trace_moe   | MQTT-Malformed_Data     | False        |        0.0065 |
| trace_moe   | MQTT-Malformed_Data     | True         |        0.0005 |
| trace_moe   | Recon-VulScan           | False        |        0.0014 |
| trace_moe   | Recon-VulScan           | True         |       -0.0041 |
| trace_moe   | TCP_IP-DoS-UDP          | False        |        0.0078 |
| trace_moe   | TCP_IP-DoS-UDP          | True         |       -0      |
| transformer | ARP_Spoofing            | False        |       -0.0238 |
| transformer | MQTT-DDoS-Publish_Flood | False        |        0.0028 |
| transformer | MQTT-DDoS-Publish_Flood | True         |       -0.0837 |
| transformer | MQTT-Malformed_Data     | False        |       -0.0076 |
| transformer | MQTT-Malformed_Data     | True         |       -0.0083 |
| transformer | Recon-VulScan           | False        |       -0.0019 |
| transformer | Recon-VulScan           | True         |       -0.0468 |
| transformer | TCP_IP-DoS-UDP          | False        |       -0.0064 |
| transformer | TCP_IP-DoS-UDP          | True         |       -0.0004 |
