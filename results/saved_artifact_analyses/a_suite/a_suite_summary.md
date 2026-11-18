# A1: conditional ASR (bounds; exact values need per-window clean and attacked predictions, which were not saved)

ASR = 1 - attack_detection_rate on the attack windows. f = ASR at eps 0 (clean miss floor). Conditional ASR = P(evaded | clean-detected) lies in [(ASR-f)/(1-f), min(1, ASR/(1-f))]; the lower end is exact if no clean-missed window becomes detected under attack.

| curve         | model       |   eps |   asr_uncond |   clean_miss_floor |   cond_asr_lower |   cond_asr_upper |   floor_share_of_asr |
|:--------------|:------------|------:|-------------:|-------------------:|-----------------:|-----------------:|---------------------:|
| clean-trained | trace_moe   |  0.05 |        0.055 |              0.051 |            0.004 |            0.058 |                0.927 |
| clean-trained | trace_moe   |  0.1  |        0.06  |              0.051 |            0.01  |            0.064 |                0.85  |
| clean-trained | trace_moe   |  0.25 |        0.081 |              0.051 |            0.031 |            0.085 |                0.638 |
| clean-trained | trace_moe   |  0.5  |        0.208 |              0.051 |            0.165 |            0.219 |                0.269 |
| clean-trained | transformer |  0.05 |        0.062 |              0.058 |            0.005 |            0.066 |                0.93  |
| clean-trained | transformer |  0.1  |        0.065 |              0.058 |            0.008 |            0.069 |                0.884 |
| clean-trained | transformer |  0.25 |        0.14  |              0.058 |            0.087 |            0.148 |                0.442 |
| clean-trained | transformer |  0.5  |        0.352 |              0.058 |            0.312 |            0.374 |                0.169 |
| clean-trained | bilstm_dnn  |  0.05 |        0.082 |              0.081 |            0.001 |            0.09  |                0.985 |
| clean-trained | bilstm_dnn  |  0.1  |        0.084 |              0.081 |            0.004 |            0.092 |                0.961 |
| clean-trained | bilstm_dnn  |  0.25 |        0.104 |              0.081 |            0.025 |            0.114 |                0.779 |
| clean-trained | bilstm_dnn  |  0.5  |        0.127 |              0.081 |            0.05  |            0.138 |                0.648 |
| clean-trained | tfsage_lite |  0.05 |        0.05  |              0.046 |            0.005 |            0.053 |                0.912 |
| clean-trained | tfsage_lite |  0.1  |        0.054 |              0.046 |            0.009 |            0.057 |                0.848 |
| clean-trained | tfsage_lite |  0.25 |        0.075 |              0.046 |            0.03  |            0.078 |                0.626 |
| clean-trained | tfsage_lite |  0.5  |        0.231 |              0.046 |            0.194 |            0.242 |                0.201 |
| fine-tuned    | trace_moe   |  0.05 |        0.045 |              0.042 |            0.003 |            0.047 |                0.931 |
| fine-tuned    | trace_moe   |  0.1  |        0.049 |              0.042 |            0.007 |            0.051 |                0.862 |
| fine-tuned    | trace_moe   |  0.25 |        0.06  |              0.042 |            0.018 |            0.062 |                0.714 |
| fine-tuned    | trace_moe   |  0.5  |        0.104 |              0.042 |            0.064 |            0.109 |                0.424 |
| fine-tuned    | transformer |  0.05 |        0.053 |              0.049 |            0.004 |            0.056 |                0.931 |
| fine-tuned    | transformer |  0.1  |        0.056 |              0.049 |            0.007 |            0.059 |                0.875 |
| fine-tuned    | transformer |  0.25 |        0.067 |              0.049 |            0.019 |            0.071 |                0.735 |
| fine-tuned    | transformer |  0.5  |        0.23  |              0.049 |            0.19  |            0.242 |                0.224 |
| fine-tuned    | bilstm_dnn  |  0.05 |        0.079 |              0.078 |            0.002 |            0.086 |                0.978 |
| fine-tuned    | bilstm_dnn  |  0.1  |        0.081 |              0.078 |            0.004 |            0.088 |                0.958 |
| fine-tuned    | bilstm_dnn  |  0.25 |        0.084 |              0.078 |            0.007 |            0.092 |                0.921 |
| fine-tuned    | bilstm_dnn  |  0.5  |        0.107 |              0.078 |            0.032 |            0.116 |                0.735 |
| fine-tuned    | tfsage_lite |  0.05 |        0.043 |              0.04  |            0.004 |            0.045 |                0.912 |
| fine-tuned    | tfsage_lite |  0.1  |        0.047 |              0.04  |            0.008 |            0.049 |                0.837 |
| fine-tuned    | tfsage_lite |  0.25 |        0.057 |              0.04  |            0.018 |            0.059 |                0.7   |
| fine-tuned    | tfsage_lite |  0.5  |        0.183 |              0.04  |            0.149 |            0.19  |                0.221 |


# A2: per-class degradation DeltaF1 = F1(eps 0) - F1(eps 0.50), mean over five seeds


## clean-trained curve

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean_delta |   rank_spread |   rank_trace_moe |   rank_others_mean |
|:------------------------|------------:|--------------:|-------------:|--------------:|-------------:|--------------:|-----------------:|-------------------:|
| TCP_IP-DoS-ICMP         |       0.874 |         0.972 |        0.774 |         0.796 |        0.854 |             3 |                2 |              4     |
| TCP_IP-DoS-UDP          |       0.722 |         0.89  |        0.87  |         0.908 |        0.847 |             3 |                4 |              2.333 |
| TCP_IP-DDoS-ICMP        |       0.793 |         0.982 |        0.688 |         0.677 |        0.785 |             5 |                3 |              5     |
| TCP_IP-DoS-TCP          |       0.594 |         0.785 |        0.819 |         0.914 |        0.778 |             6 |                7 |              3.667 |
| TCP_IP-DDoS-TCP         |       0.617 |         0.834 |        0.774 |         0.808 |        0.758 |             3 |                5 |              4.667 |
| TCP_IP-DoS-SYN          |       0.904 |         0.749 |        0.631 |         0.687 |        0.743 |             7 |                1 |              6.667 |
| TCP_IP-DDoS-UDP         |       0.605 |         0.997 |        0.834 |         0.391 |        0.707 |             7 |                6 |              3.667 |
| MQTT-DoS-Connect_Flood  |       0.303 |         0.606 |        0.831 |         0.433 |        0.543 |             9 |               12 |              6.333 |
| MQTT-DDoS-Connect_Flood |       0.519 |         0.7   |        0.55  |         0.278 |        0.512 |             4 |                8 |             10     |
| TCP_IP-DDoS-SYN         |       0.499 |         0.601 |        0.605 |         0.321 |        0.506 |             1 |                9 |              9.667 |
| MQTT-Malformed_Data     |       0.308 |         0.358 |        0.252 |         0.297 |        0.303 |             1 |               11 |             11.333 |
| ARP_Spoofing            |       0.339 |         0.334 |        0.265 |         0.183 |        0.28  |             5 |               10 |             12.667 |
| Benign                  |       0.27  |         0.316 |        0.156 |         0.335 |        0.269 |             6 |               13 |             12.333 |
| Recon-Port_Scan         |       0.212 |         0.197 |        0.193 |         0.187 |        0.197 |             1 |               15 |             14.333 |
| Recon-OS_Scan           |       0.262 |         0.255 |       -0.014 |         0.199 |        0.176 |             6 |               14 |             15.333 |
| Recon-VulScan           |       0.137 |         0.116 |        0.126 |         0.104 |        0.121 |             1 |               16 |             16.333 |
| Recon-Ping_Sweep        |       0.031 |         0.04  |        0.203 |         0.046 |        0.08  |             4 |               17 |             15.667 |
| MQTT-DoS-Publish_Flood  |       0.004 |         0.002 |        0.133 |         0.007 |        0.037 |             3 |               19 |             17.333 |
| MQTT-DDoS-Publish_Flood |       0.005 |         0.001 |        0.055 |         0.004 |        0.017 |             1 |               18 |             18.667 |

Most degraded (pooled mean): TCP_IP-DoS-ICMP, TCP_IP-DoS-UDP, TCP_IP-DDoS-ICMP. Least degraded: Recon-Ping_Sweep, MQTT-DoS-Publish_Flood, MQTT-DDoS-Publish_Flood.
TRACE-MoE rank differs from the mean rank of the other three by >= 5 places: TCP_IP-DoS-SYN, MQTT-DoS-Connect_Flood.

## adversarially fine-tuned curve

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean_delta |   rank_spread |   rank_trace_moe |   rank_others_mean |
|:------------------------|------------:|--------------:|-------------:|--------------:|-------------:|--------------:|-----------------:|-------------------:|
| TCP_IP-DoS-UDP          |       0.673 |         0.775 |        0.685 |         0.812 |        0.736 |             3 |                3 |              2     |
| TCP_IP-DoS-ICMP         |       0.767 |         0.809 |        0.634 |         0.614 |        0.706 |             4 |                2 |              4     |
| TCP_IP-DDoS-UDP         |       0.646 |         0.807 |        0.643 |         0.66  |        0.689 |             2 |                4 |              3     |
| TCP_IP-DDoS-ICMP        |       0.621 |         0.82  |        0.637 |         0.642 |        0.68  |             5 |                6 |              3     |
| TCP_IP-DoS-SYN          |       0.831 |         0.772 |        0.482 |         0.539 |        0.656 |             7 |                1 |              6.667 |
| TCP_IP-DDoS-TCP         |       0.212 |         0.621 |        0.598 |         0.67  |        0.525 |             6 |                9 |              5.333 |
| TCP_IP-DoS-TCP          |       0.173 |         0.563 |        0.614 |         0.734 |        0.521 |             9 |               11 |              5     |
| TCP_IP-DDoS-SYN         |       0.645 |         0.497 |        0.428 |         0.407 |        0.494 |             5 |                5 |              8.667 |
| MQTT-DDoS-Connect_Flood |       0.292 |         0.163 |        0.617 |         0.164 |        0.309 |             8 |                7 |             10     |
| MQTT-DoS-Connect_Flood  |       0.173 |         0.159 |        0.459 |         0.259 |        0.262 |             5 |               12 |             11.333 |
| MQTT-Malformed_Data     |       0.194 |         0.251 |        0.247 |         0.281 |        0.243 |             1 |               10 |             10.333 |
| ARP_Spoofing            |       0.255 |         0.267 |        0.204 |         0.162 |        0.222 |             5 |                8 |             11.333 |
| Benign                  |       0.128 |         0.221 |        0.086 |         0.286 |        0.18  |             6 |               15 |             11.667 |
| Recon-OS_Scan           |       0.159 |         0.182 |        0.044 |         0.109 |        0.124 |             6 |               13 |             15     |
| Recon-Port_Scan         |       0.133 |         0.119 |        0.107 |         0.112 |        0.118 |             1 |               14 |             14.333 |
| Recon-VulScan           |       0.109 |         0.103 |        0.068 |         0.105 |        0.096 |             0 |               16 |             16     |
| Recon-Ping_Sweep        |       0.023 |         0.04  |        0.162 |         0.049 |        0.068 |             4 |               17 |             15.667 |
| MQTT-DoS-Publish_Flood  |       0.002 |         0.001 |        0.054 |         0.006 |        0.016 |             2 |               18 |             18     |
| MQTT-DDoS-Publish_Flood |       0.001 |         0.001 |        0.037 |         0.003 |        0.011 |             1 |               19 |             18.667 |

Most degraded (pooled mean): TCP_IP-DoS-UDP, TCP_IP-DoS-ICMP, TCP_IP-DDoS-UDP. Least degraded: Recon-Ping_Sweep, MQTT-DoS-Publish_Flood, MQTT-DDoS-Publish_Flood.
TRACE-MoE rank differs from the mean rank of the other three by >= 5 places: TCP_IP-DoS-SYN, TCP_IP-DoS-TCP.


# A3: routing instability versus per-class F1 drop

NOT COMPUTABLE per class: routing_instability.json stores instability_rate, jaccard and the expert distribution per seed and eps only, not per attack class (and only for clean-trained TRACE-MoE). The per-seed version (n = 5) was already reported as S3 (Spearman 0.80, exact p = 0.13). A per-class analysis needs the per-window router choices, which were not saved.


# A4: across-seed SD of class F1 at eps 0.50


## clean-trained

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean_sd |
|:------------------------|------------:|--------------:|-------------:|--------------:|----------:|
| MQTT-DDoS-Connect_Flood |       0.255 |         0.24  |        0.057 |         0.36  |     0.228 |
| TCP_IP-DoS-TCP          |       0.379 |         0.21  |        0.102 |         0.132 |     0.206 |
| TCP_IP-DoS-UDP          |       0.282 |         0.241 |        0.135 |         0.09  |     0.187 |
| TCP_IP-DDoS-TCP         |       0.349 |         0.155 |        0.103 |         0.127 |     0.184 |
| TCP_IP-DoS-SYN          |       0.077 |         0.308 |        0.153 |         0.189 |     0.182 |
| MQTT-DoS-Connect_Flood  |       0.122 |         0.248 |        0.159 |         0.163 |     0.173 |
| TCP_IP-DDoS-ICMP        |       0.23  |         0.039 |        0.068 |         0.35  |     0.172 |
| TCP_IP-DDoS-SYN         |       0.181 |         0.173 |        0.102 |         0.06  |     0.129 |
| TCP_IP-DoS-ICMP         |       0.141 |         0.041 |        0.07  |         0.194 |     0.112 |
| TCP_IP-DDoS-UDP         |       0.237 |         0.005 |        0.09  |         0.085 |     0.104 |
| MQTT-Malformed_Data     |       0.076 |         0.04  |        0.067 |         0.103 |     0.071 |
| Benign                  |       0.084 |         0.084 |        0.037 |         0.056 |     0.065 |
| ARP_Spoofing            |       0.065 |         0.088 |        0.05  |         0.028 |     0.058 |
| Recon-Port_Scan         |       0.064 |         0.029 |        0.076 |         0.051 |     0.055 |
| Recon-OS_Scan           |       0.067 |         0.053 |        0.01  |         0.05  |     0.045 |
| Recon-VulScan           |       0.039 |         0.04  |        0.015 |         0.072 |     0.042 |
| Recon-Ping_Sweep        |       0.018 |         0.025 |        0.063 |         0.038 |     0.036 |
| MQTT-DDoS-Publish_Flood |       0.022 |         0.023 |        0.03  |         0.022 |     0.024 |
| MQTT-DoS-Publish_Flood  |       0.003 |         0.005 |        0.045 |         0.007 |     0.015 |

Most variable (mean SD): MQTT-DDoS-Connect_Flood, TCP_IP-DoS-TCP, TCP_IP-DoS-UDP, TCP_IP-DDoS-TCP, TCP_IP-DoS-SYN. Least variable: Recon-OS_Scan, Recon-VulScan, Recon-Ping_Sweep, MQTT-DDoS-Publish_Flood, MQTT-DoS-Publish_Flood.
In the five most variable classes of every architecture: none; of at least two architectures: ['MQTT-DDoS-Connect_Flood', 'TCP_IP-DoS-TCP', 'TCP_IP-DoS-UDP', 'TCP_IP-DDoS-TCP', 'TCP_IP-DoS-SYN', 'MQTT-DoS-Connect_Flood'].
TCP/IP classes among the top-5 pooled: ['TCP_IP-DoS-TCP', 'TCP_IP-DoS-UDP', 'TCP_IP-DDoS-TCP', 'TCP_IP-DoS-SYN'].

## ft-trained

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean_sd |
|:------------------------|------------:|--------------:|-------------:|--------------:|----------:|
| TCP_IP-DoS-ICMP         |       0.272 |         0.253 |        0.097 |         0.328 |     0.238 |
| TCP_IP-DoS-UDP          |       0.3   |         0.272 |        0.152 |         0.205 |     0.232 |
| TCP_IP-DoS-TCP          |       0.038 |         0.418 |        0.159 |         0.26  |     0.219 |
| TCP_IP-DDoS-ICMP        |       0.151 |         0.288 |        0.086 |         0.336 |     0.215 |
| MQTT-DDoS-Connect_Flood |       0.236 |         0.188 |        0.248 |         0.181 |     0.213 |
| TCP_IP-DDoS-UDP         |       0.398 |         0.194 |        0.061 |         0.151 |     0.201 |
| TCP_IP-DDoS-SYN         |       0.333 |         0.212 |        0.123 |         0.135 |     0.201 |
| TCP_IP-DDoS-TCP         |       0.135 |         0.185 |        0.137 |         0.273 |     0.182 |
| TCP_IP-DoS-SYN          |       0.194 |         0.211 |        0.08  |         0.194 |     0.17  |
| MQTT-DoS-Connect_Flood  |       0.089 |         0.083 |        0.096 |         0.118 |     0.096 |
| Benign                  |       0.057 |         0.082 |        0.026 |         0.075 |     0.06  |
| ARP_Spoofing            |       0.033 |         0.07  |        0.066 |         0.042 |     0.053 |
| MQTT-Malformed_Data     |       0.032 |         0.032 |        0.023 |         0.104 |     0.048 |
| Recon-OS_Scan           |       0.059 |         0.059 |        0.024 |         0.033 |     0.044 |
| Recon-Ping_Sweep        |       0.011 |         0.063 |        0.044 |         0.052 |     0.043 |
| Recon-Port_Scan         |       0.033 |         0.034 |        0.042 |         0.049 |     0.04  |
| Recon-VulScan           |       0.059 |         0.038 |        0.014 |         0.036 |     0.037 |
| MQTT-DDoS-Publish_Flood |       0.023 |         0.023 |        0.046 |         0.024 |     0.029 |
| MQTT-DoS-Publish_Flood  |       0.006 |         0.005 |        0.034 |         0.006 |     0.012 |

Most variable (mean SD): TCP_IP-DoS-ICMP, TCP_IP-DoS-UDP, TCP_IP-DoS-TCP, TCP_IP-DDoS-ICMP, MQTT-DDoS-Connect_Flood. Least variable: Recon-Ping_Sweep, Recon-Port_Scan, Recon-VulScan, MQTT-DDoS-Publish_Flood, MQTT-DoS-Publish_Flood.
In the five most variable classes of every architecture: ['TCP_IP-DoS-UDP']; of at least two architectures: ['TCP_IP-DoS-ICMP', 'TCP_IP-DoS-UDP', 'TCP_IP-DoS-TCP', 'TCP_IP-DDoS-ICMP', 'MQTT-DDoS-Connect_Flood', 'TCP_IP-DDoS-SYN', 'TCP_IP-DDoS-TCP'].
TCP/IP classes among the top-5 pooled: ['TCP_IP-DoS-ICMP', 'TCP_IP-DoS-UDP', 'TCP_IP-DoS-TCP', 'TCP_IP-DDoS-ICMP'].


# A5: PCRP discrepancy distribution summary (attack windows, 5-step PGD, 1% subsample; saved: mean and SD per seed and eps)

| model       |   eps |   mean_D |   sd_D_pooled |   sd_of_seed_means |   shift_in_mean |   shift_over_window_sd |
|:------------|------:|---------:|--------------:|-------------------:|----------------:|-----------------------:|
| bilstm_dnn  |  0    |   11.213 |        44.971 |              0.494 |           0     |                  0     |
| bilstm_dnn  |  0.25 |   25.053 |        39.237 |              0.552 |          13.84  |                  0.353 |
| bilstm_dnn  |  0.5  |   42.912 |        56.446 |              0.388 |          31.699 |                  0.562 |
| tfsage_lite |  0    |   11.219 |        44.973 |              0.5   |           0     |                  0     |
| tfsage_lite |  0.25 |   25.169 |        47.433 |              2.023 |          13.95  |                  0.294 |
| tfsage_lite |  0.5  |   42.064 |        53.443 |              3.444 |          30.845 |                  0.577 |
| trace_moe   |  0    |   11.213 |        44.972 |              0.497 |           0     |                  0     |
| trace_moe   |  0.25 |   26.193 |        49.209 |              1.386 |          14.979 |                  0.304 |
| trace_moe   |  0.5  |   43.792 |        54.3   |              1.914 |          32.578 |                  0.6   |
| transformer |  0    |   11.212 |        44.97  |              0.495 |           0     |                  0     |
| transformer |  0.25 |   24.496 |        47.35  |              2.82  |          13.284 |                  0.281 |
| transformer |  0.5  |   41.46  |        51.392 |              5.32  |          30.248 |                  0.589 |

The saved files hold only mean and SD of D over windows, not the windows themselves, so overlap is characterised here by the shift in the mean relative to the window-level SD (last column); histograms need the per-window scores (the S6 export, once run).


# A6: transfer asymmetry (ASR of surrogate A on target B minus ASR of B on A, per seed; one-sided exact Wilcoxon and sign test, n = 5 seeds)


## eps 0.25

|   eps | higher_direction         |   mean_diff |   seeds_in_that_direction |   wilcoxon_p_one_sided |   sign_p_one_sided |
|------:|:-------------------------|------------:|--------------------------:|-----------------------:|-------------------:|
|  0.25 | tfsage_lite->bilstm_dnn  |      0.043  |                         5 |                 0.0312 |             0.0312 |
|  0.25 | trace_moe->bilstm_dnn    |      0.0389 |                         5 |                 0.0312 |             0.0312 |
|  0.25 | transformer->bilstm_dnn  |      0.0312 |                         5 |                 0.0312 |             0.0312 |
|  0.25 | tfsage_lite->transformer |      0.0121 |                         4 |                 0.0625 |             0.1875 |
|  0.25 | trace_moe->transformer   |      0.0063 |                         5 |                 0.0312 |             0.0312 |
|  0.25 | tfsage_lite->trace_moe   |      0.0058 |                         5 |                 0.0312 |             0.0312 |

## eps 0.5

|   eps | higher_direction         |   mean_diff |   seeds_in_that_direction |   wilcoxon_p_one_sided |   sign_p_one_sided |
|------:|:-------------------------|------------:|--------------------------:|-----------------------:|-------------------:|
|   0.5 | tfsage_lite->bilstm_dnn  |      0.0524 |                         5 |                 0.0312 |             0.0312 |
|   0.5 | trace_moe->bilstm_dnn    |      0.044  |                         5 |                 0.0312 |             0.0312 |
|   0.5 | transformer->bilstm_dnn  |      0.0384 |                         5 |                 0.0312 |             0.0312 |
|   0.5 | tfsage_lite->transformer |      0.0132 |                         4 |                 0.0625 |             0.1875 |
|   0.5 | tfsage_lite->trace_moe   |      0.0064 |                         5 |                 0.0312 |             0.0312 |
|   0.5 | trace_moe->transformer   |      0.0053 |                         5 |                 0.0312 |             0.0312 |

With five seeds the smallest attainable one-sided exact p is 0.031, and 6 pairs x 4 budgets are examined, so these are descriptive, not corrected.


# A7: clean F1 versus attacked F1 (eps 0.50) per class, 19 classes x 4 architectures


## clean-trained curve: Spearman rho (clean F1 vs attacked F1) pooled over 76 class-architecture points = -0.129 (p = 0.27); trace_moe: -0.18 (p=0.46); transformer: -0.52 (p=0.022); bilstm_dnn: 0.21 (p=0.39); tfsage_lite: -0.17 (p=0.48)

Robustly correct (clean F1 >= 0.90 and attacked F1 >= 0.80): 0 points, e.g. [].
Accurate but brittle (clean >= 0.90, attacked < 0.30): 22 points, e.g. ['MQTT-DDoS-Connect_Flood', 'MQTT-DoS-Connect_Flood', 'TCP_IP-DDoS-ICMP', 'TCP_IP-DDoS-TCP', 'TCP_IP-DDoS-UDP', 'TCP_IP-DoS-ICMP'].
Already weak and further hurt (clean < 0.30 and attacked below clean): 5 points, e.g. ['MQTT-DDoS-Publish_Flood', 'Recon-VulScan'].

## ft-trained curve: Spearman rho (clean F1 vs attacked F1) pooled over 76 class-architecture points = 0.113 (p = 0.33); trace_moe: 0.20 (p=0.41); transformer: -0.11 (p=0.65); bilstm_dnn: 0.33 (p=0.17); tfsage_lite: -0.02 (p=0.92)

Robustly correct (clean F1 >= 0.90 and attacked F1 >= 0.80): 5 points, e.g. ['MQTT-DDoS-Connect_Flood', 'MQTT-DoS-Connect_Flood', 'TCP_IP-DoS-TCP'].
Accurate but brittle (clean >= 0.90, attacked < 0.30): 9 points, e.g. ['TCP_IP-DDoS-ICMP', 'TCP_IP-DDoS-UDP', 'TCP_IP-DoS-ICMP', 'TCP_IP-DoS-SYN', 'TCP_IP-DoS-TCP', 'TCP_IP-DoS-UDP'].
Already weak and further hurt (clean < 0.30 and attacked below clean): 6 points, e.g. ['MQTT-DDoS-Publish_Flood', 'Recon-OS_Scan', 'Recon-VulScan'].

Thresholds (0.90, 0.80, 0.30) are descriptive cut-offs chosen here, not pre-registered. ASR per class is available per class in metrics.json (asr_per_class) but is a binary-evasion rate, not the 19-class F1 used here.