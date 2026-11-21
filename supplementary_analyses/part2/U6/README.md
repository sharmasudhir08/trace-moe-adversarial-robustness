# U6 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch1.py --date 20261121 --tasks U6
Script SHA-256: 33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz and metrics.json, h8_verdict CSVs. Runs read: 75. rng seed 42, bootstrap 10000 over the five seeds.
Started 2026-11-21 19:10:53, ended 2026-11-21 19:11:02.

Flag = energy < exported thr (5th percentile of known validation energy). Check: overall known FAR equals ESA1 far_s<seed> within 9.71e-17 (criterion 1e-6; PASS).

Top three known classes by contribution per model and held-out class:

| model       | held_out_class          | known_class      |   flag_rate |   flag_rate_lo |   flag_rate_hi |   contribution_to_far |   contribution_to_far_lo |   contribution_to_far_hi |   n_seeds | exceeds_5pct   |
|:------------|:------------------------|:-----------------|------------:|---------------:|---------------:|----------------------:|-------------------------:|-------------------------:|----------:|:---------------|
| bilstm_dnn  | Recon-VulScan           | TCP_IP-DDoS-TCP  |      0.3345 |         0.0048 |         0.7155 |                0.0362 |                   0.0005 |                   0.0773 |         5 | True           |
| transformer | TCP_IP-DoS-UDP          | Benign           |      0.9517 |         0.9317 |         0.9709 |                0.0247 |                   0.0242 |                   0.0252 |         5 | True           |
| transformer | ARP_Spoofing            | Benign           |      0.9998 |         0.9997 |         0.9999 |                0.0237 |                   0.0237 |                   0.0238 |         5 | True           |
| transformer | Recon-VulScan           | Benign           |      0.9966 |         0.9948 |         0.9985 |                0.0237 |                   0.0236 |                   0.0237 |         5 | True           |
| trace_moe   | Recon-VulScan           | Benign           |      0.9934 |         0.9873 |         0.9993 |                0.0236 |                   0.0234 |                   0.0237 |         5 | True           |
| trace_moe   | MQTT-DDoS-Publish_Flood | Benign           |      0.9867 |         0.9642 |         0.9996 |                0.0235 |                   0.023  |                   0.0238 |         5 | True           |
| transformer | MQTT-DDoS-Publish_Flood | Benign           |      0.9774 |         0.9623 |         0.9894 |                0.0233 |                   0.0229 |                   0.0236 |         5 | True           |
| transformer | MQTT-Malformed_Data     | Benign           |      0.9785 |         0.9545 |         0.9959 |                0.0232 |                   0.0227 |                   0.0237 |         5 | True           |
| bilstm_dnn  | MQTT-Malformed_Data     | TCP_IP-DDoS-ICMP |      0.1072 |         0.0009 |         0.3179 |                0.0232 |                   0.0002 |                   0.0688 |         5 | True           |
| trace_moe   | TCP_IP-DoS-UDP          | Benign           |      0.7544 |         0.3815 |         0.9753 |                0.0196 |                   0.0098 |                   0.0253 |         5 | True           |
| trace_moe   | MQTT-Malformed_Data     | Benign           |      0.8232 |         0.6114 |         0.9977 |                0.0196 |                   0.0145 |                   0.0237 |         5 | True           |
| trace_moe   | ARP_Spoofing            | Benign           |      0.8164 |         0.4595 |         0.9998 |                0.0194 |                   0.0109 |                   0.0238 |         5 | True           |
| bilstm_dnn  | ARP_Spoofing            | Benign           |      0.6889 |         0.4538 |         0.9135 |                0.0164 |                   0.0108 |                   0.0217 |         5 | True           |
| transformer | TCP_IP-DoS-UDP          | Recon-Port_Scan  |      0.9311 |         0.8993 |         0.9587 |                0.0145 |                   0.014  |                   0.015  |         5 | True           |
| bilstm_dnn  | TCP_IP-DoS-UDP          | Benign           |      0.5561 |         0.427  |         0.6629 |                0.0144 |                   0.0106 |                   0.0172 |         5 | True           |
| transformer | Recon-VulScan           | Recon-Port_Scan  |      0.9961 |         0.9949 |         0.9973 |                0.0142 |                   0.0142 |                   0.0142 |         5 | True           |
| transformer | MQTT-DDoS-Publish_Flood | Recon-Port_Scan  |      0.988  |         0.9837 |         0.9923 |                0.0142 |                   0.0141 |                   0.0142 |         5 | True           |
| transformer | ARP_Spoofing            | Recon-Port_Scan  |      0.9922 |         0.9871 |         0.9967 |                0.0142 |                   0.0141 |                   0.0142 |         5 | True           |
| transformer | MQTT-Malformed_Data     | Recon-Port_Scan  |      0.9898 |         0.9822 |         0.9952 |                0.0141 |                   0.014  |                   0.0142 |         5 | True           |
| bilstm_dnn  | Recon-VulScan           | TCP_IP-DoS-UDP   |      0.1502 |         0.0013 |         0.445  |                0.0131 |                   0.0001 |                   0.0387 |         5 | True           |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | TCP_IP-DDoS-TCP  |      0.1024 |         0.0104 |         0.2509 |                0.0111 |                   0.0011 |                   0.0272 |         5 | True           |
| trace_moe   | Recon-VulScan           | Recon-Port_Scan  |      0.7627 |         0.5215 |         0.9518 |                0.0109 |                   0.0074 |                   0.0136 |         5 | True           |
| bilstm_dnn  | MQTT-Malformed_Data     | TCP_IP-DDoS-TCP  |      0.0957 |         0.0043 |         0.2331 |                0.0104 |                   0.0005 |                   0.0253 |         5 | True           |
| bilstm_dnn  | MQTT-Malformed_Data     | Benign           |      0.414  |         0.2008 |         0.5875 |                0.0098 |                   0.0048 |                   0.014  |         5 | True           |
| bilstm_dnn  | ARP_Spoofing            | TCP_IP-DDoS-SYN  |      0.0884 |         0.0138 |         0.1858 |                0.0096 |                   0.0015 |                   0.0201 |         5 | True           |
| trace_moe   | TCP_IP-DoS-UDP          | TCP_IP-DoS-TCP   |      0.1653 |         0.0001 |         0.4954 |                0.0093 |                   0      |                   0.0277 |         5 | True           |
| trace_moe   | MQTT-DDoS-Publish_Flood | Recon-Port_Scan  |      0.6378 |         0.5245 |         0.8042 |                0.0091 |                   0.0075 |                   0.0116 |         5 | True           |
| trace_moe   | ARP_Spoofing            | Recon-Port_Scan  |      0.5893 |         0.2976 |         0.8762 |                0.0084 |                   0.0043 |                   0.0125 |         5 | True           |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | Benign           |      0.3295 |         0.1642 |         0.5309 |                0.0079 |                   0.0039 |                   0.0127 |         5 | True           |
| bilstm_dnn  | Recon-VulScan           | Benign           |      0.3231 |         0.1909 |         0.4727 |                0.0077 |                   0.0045 |                   0.0113 |         5 | True           |
| trace_moe   | MQTT-Malformed_Data     | Recon-Port_Scan  |      0.5156 |         0.2678 |         0.7722 |                0.0074 |                   0.0038 |                   0.011  |         5 | True           |
| bilstm_dnn  | ARP_Spoofing            | TCP_IP-DDoS-TCP  |      0.0677 |         0.0071 |         0.1283 |                0.0073 |                   0.0008 |                   0.0139 |         5 | True           |
| bilstm_dnn  | MQTT-DDoS-Publish_Flood | TCP_IP-DDoS-SYN  |      0.0637 |         0.0154 |         0.1525 |                0.007  |                   0.0017 |                   0.0167 |         5 | True           |
| transformer | ARP_Spoofing            | TCP_IP-DoS-SYN   |      0.1061 |         0.0027 |         0.2295 |                0.0066 |                   0.0002 |                   0.0143 |         5 | True           |
| bilstm_dnn  | TCP_IP-DoS-UDP          | Recon-Port_Scan  |      0.4095 |         0.2048 |         0.6005 |                0.0064 |                   0.003  |                   0.0094 |         5 | True           |
| bilstm_dnn  | TCP_IP-DoS-UDP          | TCP_IP-DDoS-SYN  |      0.0521 |         0.0185 |         0.1076 |                0.0062 |                   0.0022 |                   0.0129 |         5 | True           |
| trace_moe   | TCP_IP-DoS-UDP          | Recon-Port_Scan  |      0.3511 |         0.1557 |         0.5464 |                0.0055 |                   0.0024 |                   0.0085 |         5 | True           |
| trace_moe   | ARP_Spoofing            | TCP_IP-DoS-ICMP  |      0.0821 |         0.0154 |         0.175  |                0.0046 |                   0.0009 |                   0.0098 |         5 | True           |
| trace_moe   | MQTT-Malformed_Data     | TCP_IP-DDoS-ICMP |      0.0207 |         0.0002 |         0.05   |                0.0045 |                   0.0001 |                   0.0108 |         5 | False          |
| trace_moe   | MQTT-DDoS-Publish_Flood | TCP_IP-DDoS-TCP  |      0.0301 |         0.0005 |         0.0893 |                0.0033 |                   0.0001 |                   0.0097 |         5 | False          |
| transformer | MQTT-Malformed_Data     | Recon-OS_Scan    |      0.9162 |         0.8153 |         0.978  |                0.002  |                   0.0018 |                   0.0022 |         5 | True           |
| transformer | MQTT-DDoS-Publish_Flood | Recon-OS_Scan    |      0.8918 |         0.8216 |         0.962  |                0.002  |                   0.0018 |                   0.0021 |         5 | True           |
| trace_moe   | Recon-VulScan           | TCP_IP-DDoS-TCP  |      0.018  |         0.005  |         0.0342 |                0.002  |                   0.0005 |                   0.0037 |         5 | False          |
| transformer | Recon-VulScan           | Recon-OS_Scan    |      0.7888 |         0.7396 |         0.843  |                0.0017 |                   0.0016 |                   0.0019 |         5 | True           |
| transformer | TCP_IP-DoS-UDP          | Recon-OS_Scan    |      0.6353 |         0.5746 |         0.6844 |                0.0015 |                   0.0014 |                   0.0017 |         5 | True           |
