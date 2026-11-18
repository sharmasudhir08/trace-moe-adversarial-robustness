# S6a: exact conditional ASR (clean-trained models; evasion = 6-category head predicts Benign)

| model       |   eps |   asr_proxy |   asr_stored |   clean_missed_share |   cond_asr_given_clean_detected |   asr_given_clean_missed |
|:------------|------:|------------:|-------------:|---------------------:|--------------------------------:|-------------------------:|
| trace_moe   |  0.05 |       0.044 |        0.055 |                0.043 |                           0.002 |                    0.98  |
| trace_moe   |  0.1  |       0.045 |        0.06  |                0.043 |                           0.004 |                    0.96  |
| trace_moe   |  0.25 |       0.049 |        0.081 |                0.043 |                           0.008 |                    0.949 |
| trace_moe   |  0.5  |       0.059 |        0.208 |                0.043 |                           0.02  |                    0.939 |
| transformer |  0.05 |       0.04  |        0.062 |                0.041 |                           0.002 |                    0.946 |
| transformer |  0.1  |       0.04  |        0.065 |                0.041 |                           0.003 |                    0.912 |
| transformer |  0.25 |       0.041 |        0.14  |                0.041 |                           0.008 |                    0.806 |
| transformer |  0.5  |       0.09  |        0.352 |                0.041 |                           0.066 |                    0.668 |
| bilstm_dnn  |  0.05 |       0.049 |        0.082 |                0.047 |                           0.007 |                    0.917 |
| bilstm_dnn  |  0.1  |       0.05  |        0.084 |                0.047 |                           0.011 |                    0.856 |
| bilstm_dnn  |  0.25 |       0.066 |        0.104 |                0.047 |                           0.03  |                    0.801 |
| bilstm_dnn  |  0.5  |       0.08  |        0.127 |                0.047 |                           0.049 |                    0.701 |
| tfsage_lite |  0.05 |       0.041 |        0.05  |                0.039 |                           0.002 |                    0.997 |
| tfsage_lite |  0.1  |       0.043 |        0.054 |                0.039 |                           0.004 |                    0.989 |
| tfsage_lite |  0.25 |       0.048 |        0.075 |                0.039 |                           0.01  |                    0.968 |
| tfsage_lite |  0.5  |       0.065 |        0.231 |                0.039 |                           0.029 |                    0.947 |

asr_proxy is computed from the saved per-window predictions of the 6-category head; asr_stored is 1 - attack_detection_rate from metrics.json (binary head). A difference between them is the proxy error.



# S6b: PCRP score distributions, clean attack windows versus attacked windows (mean over five seeds)

pred = predicted-class variant, min = nearest-centroid variant. overlap_coefficient = sum of the minima of the two normalised histograms (1 = identical, 0 = disjoint).

| model       |   eps | score   |   clean_mean |   clean_sd |   clean_q99 |   attacked_mean |   attacked_sd |   attacked_median |   overlap_coefficient |   roc_auc |   tpr_at_1pct_fpr |
|:------------|------:|:--------|-------------:|-----------:|------------:|----------------:|--------------:|------------------:|----------------------:|----------:|------------------:|
| trace_moe   |  0.25 | pred    |       81.379 |    259.317 |     556.508 |         169.987 |       264.311 |           156.78  |                 0.443 |     0.786 |             0.012 |
| trace_moe   |  0.25 | min     |        4.367 |      4.538 |      22.011 |         122.556 |        92.925 |           112.332 |                 0.143 |     0.981 |             0.794 |
| trace_moe   |  0.5  | pred    |       81.379 |    259.317 |     556.508 |         278.044 |       312.298 |           254.671 |                 0.335 |     0.854 |             0.067 |
| trace_moe   |  0.5  | min     |        4.367 |      4.538 |      22.011 |         236.529 |       180.228 |           219.576 |                 0.091 |     0.992 |             0.867 |
| transformer |  0.25 | pred    |       82.963 |    265.705 |     584.729 |         172.248 |       296.201 |           155.953 |                 0.446 |     0.788 |             0.011 |
| transformer |  0.25 | min     |        4.367 |      4.538 |      22.011 |         120.836 |        92.983 |           108.117 |                 0.142 |     0.981 |             0.794 |
| transformer |  0.5  | pred    |       82.963 |    265.705 |     584.729 |         285.869 |       322.24  |           258.857 |                 0.337 |     0.857 |             0.055 |
| transformer |  0.5  | min     |        4.367 |      4.538 |      22.011 |         238.746 |       186.561 |           216.205 |                 0.094 |     0.992 |             0.865 |
| bilstm_dnn  |  0.25 | pred    |       84.594 |    312.97  |     539.086 |         171.087 |       307.757 |           152.509 |                 0.434 |     0.783 |             0.014 |
| bilstm_dnn  |  0.25 | min     |        4.367 |      4.538 |      22.011 |         121.401 |        89.345 |           110.1   |                 0.121 |     0.986 |             0.824 |
| bilstm_dnn  |  0.5  | pred    |       84.594 |    312.97  |     539.086 |         283.381 |       413.795 |           255.072 |                 0.315 |     0.861 |             0.076 |
| bilstm_dnn  |  0.5  | min     |        4.367 |      4.538 |      22.011 |         238.144 |       173.144 |           222.571 |                 0.072 |     0.994 |             0.902 |
| tfsage_lite |  0.25 | pred    |       84.386 |    293.632 |     579.137 |         170.02  |       309.52  |           155.123 |                 0.446 |     0.778 |             0.01  |
| tfsage_lite |  0.25 | min     |        4.367 |      4.538 |      22.011 |         118.825 |        91.762 |           107.303 |                 0.149 |     0.98  |             0.785 |
| tfsage_lite |  0.5  | pred    |       84.386 |    293.632 |     579.137 |         280.61  |       385.701 |           250.452 |                 0.342 |     0.847 |             0.052 |
| tfsage_lite |  0.5  | min     |        4.367 |      4.538 |      22.011 |         232.941 |       182.289 |           211.531 |                 0.094 |     0.992 |             0.861 |


# S6c: true-positive rate at 1% FPR by attack class (PCRP, mean over seeds)

Class names are taken from the label-19 order of the stored per-class metrics; checked below against the number of classes.


## nearest-centroid, eps 0.25: classes sorted by mean TPR over architectures (lowest first)

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean |
|:------------------------|------------:|--------------:|-------------:|--------------:|-------:|
| ARP_Spoofing            |       0.705 |         0.69  |        0.703 |         0.643 |  0.685 |
| Recon-Ping_Sweep        |       0.717 |         0.697 |        0.752 |         0.685 |  0.713 |
| MQTT-DoS-Publish_Flood  |       0.707 |         0.757 |        0.723 |         0.685 |  0.718 |
| Recon-Port_Scan         |       0.719 |         0.681 |        0.769 |         0.709 |  0.72  |
| MQTT-DDoS-Publish_Flood |       0.697 |         0.767 |        0.751 |         0.699 |  0.728 |
| Recon-VulScan           |       0.718 |         0.721 |        0.751 |         0.732 |  0.73  |
| MQTT-Malformed_Data     |       0.753 |         0.728 |        0.766 |         0.766 |  0.753 |
| Recon-OS_Scan           |       0.741 |         0.737 |        0.791 |         0.776 |  0.761 |
| MQTT-DDoS-Connect_Flood |       0.77  |         0.784 |        0.809 |         0.712 |  0.769 |
| MQTT-DoS-Connect_Flood  |       0.762 |         0.789 |        0.795 |         0.757 |  0.776 |
| TCP_IP-DoS-SYN          |       0.746 |         0.783 |        0.836 |         0.756 |  0.78  |
| TCP_IP-DDoS-SYN         |       0.812 |         0.753 |        0.821 |         0.76  |  0.787 |
| TCP_IP-DoS-TCP          |       0.881 |         0.864 |        0.938 |         0.863 |  0.886 |
| TCP_IP-DoS-ICMP         |       0.866 |         0.892 |        0.911 |         0.882 |  0.888 |
| TCP_IP-DDoS-TCP         |       0.867 |         0.871 |        0.926 |         0.902 |  0.891 |
| TCP_IP-DDoS-ICMP        |       0.915 |         0.892 |        0.925 |         0.913 |  0.911 |
| TCP_IP-DoS-UDP          |       0.94  |         0.923 |        0.919 |         0.935 |  0.929 |
| TCP_IP-DDoS-UDP         |       0.945 |         0.935 |        0.929 |         0.924 |  0.933 |

## nearest-centroid, eps 0.5: classes sorted by mean TPR over architectures (lowest first)

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean |
|:------------------------|------------:|--------------:|-------------:|--------------:|-------:|
| ARP_Spoofing            |       0.744 |         0.732 |        0.801 |         0.709 |  0.747 |
| Recon-Ping_Sweep        |       0.772 |         0.739 |        0.827 |         0.763 |  0.775 |
| MQTT-Malformed_Data     |       0.777 |         0.751 |        0.813 |         0.795 |  0.784 |
| MQTT-DDoS-Publish_Flood |       0.763 |         0.81  |        0.837 |         0.742 |  0.788 |
| MQTT-DoS-Publish_Flood  |       0.777 |         0.816 |        0.811 |         0.76  |  0.791 |
| Recon-VulScan           |       0.808 |         0.808 |        0.856 |         0.805 |  0.819 |
| Recon-Port_Scan         |       0.851 |         0.824 |        0.885 |         0.849 |  0.852 |
| Recon-OS_Scan           |       0.848 |         0.853 |        0.886 |         0.877 |  0.866 |
| MQTT-DoS-Connect_Flood  |       0.85  |         0.878 |        0.891 |         0.852 |  0.868 |
| MQTT-DDoS-Connect_Flood |       0.874 |         0.879 |        0.916 |         0.859 |  0.882 |
| TCP_IP-DDoS-SYN         |       0.893 |         0.848 |        0.932 |         0.865 |  0.884 |
| TCP_IP-DoS-SYN          |       0.871 |         0.897 |        0.931 |         0.868 |  0.892 |
| TCP_IP-DDoS-TCP         |       0.947 |         0.925 |        0.961 |         0.943 |  0.944 |
| TCP_IP-DoS-TCP          |       0.952 |         0.957 |        0.976 |         0.945 |  0.957 |
| TCP_IP-DDoS-UDP         |       0.964 |         0.951 |        0.969 |         0.951 |  0.959 |
| TCP_IP-DDoS-ICMP        |       0.966 |         0.948 |        0.97  |         0.956 |  0.96  |
| TCP_IP-DoS-ICMP         |       0.951 |         0.958 |        0.971 |         0.959 |  0.96  |
| TCP_IP-DoS-UDP          |       0.963 |         0.96  |        0.973 |         0.973 |  0.968 |

## predicted-class, eps 0.25: classes sorted by mean TPR over architectures (lowest first)

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean |
|:------------------------|------------:|--------------:|-------------:|--------------:|-------:|
| MQTT-DDoS-Connect_Flood |       0     |         0     |        0     |         0     |  0     |
| MQTT-DDoS-Publish_Flood |       0     |         0     |        0     |         0     |  0     |
| MQTT-DoS-Connect_Flood  |       0     |         0     |        0     |         0     |  0     |
| MQTT-DoS-Publish_Flood  |       0     |         0     |        0     |         0     |  0     |
| TCP_IP-DDoS-ICMP        |       0     |         0     |        0     |         0     |  0     |
| TCP_IP-DDoS-UDP         |       0     |         0     |        0     |         0     |  0     |
| TCP_IP-DDoS-SYN         |       0     |         0     |        0     |         0     |  0     |
| TCP_IP-DoS-TCP          |       0     |         0     |        0     |         0     |  0     |
| TCP_IP-DoS-UDP          |       0     |         0     |        0     |         0     |  0     |
| TCP_IP-DoS-SYN          |       0     |         0.001 |        0     |         0     |  0     |
| TCP_IP-DoS-ICMP         |       0     |         0     |        0.002 |         0     |  0     |
| Recon-Port_Scan         |       0     |         0.001 |        0.001 |         0     |  0     |
| MQTT-Malformed_Data     |       0.003 |         0.002 |        0.006 |         0.001 |  0.003 |
| TCP_IP-DDoS-TCP         |       0.004 |         0.004 |        0.003 |         0.004 |  0.004 |
| ARP_Spoofing            |       0.008 |         0.011 |        0.002 |         0.008 |  0.007 |
| Recon-VulScan           |       0.006 |         0.011 |        0.016 |         0.006 |  0.01  |
| Recon-OS_Scan           |       0.016 |         0.014 |        0.013 |         0.018 |  0.015 |
| Recon-Ping_Sweep        |       0.261 |         0.218 |        0.309 |         0.218 |  0.251 |

## predicted-class, eps 0.5: classes sorted by mean TPR over architectures (lowest first)

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |   mean |
|:------------------------|------------:|--------------:|-------------:|--------------:|-------:|
| MQTT-DoS-Connect_Flood  |       0.023 |         0.024 |        0.03  |         0.008 |  0.021 |
| MQTT-DDoS-Connect_Flood |       0.024 |         0.023 |        0.041 |         0.006 |  0.023 |
| TCP_IP-DDoS-SYN         |       0.046 |         0.013 |        0.037 |         0.016 |  0.028 |
| TCP_IP-DoS-SYN          |       0.033 |         0.021 |        0.045 |         0.024 |  0.031 |
| MQTT-DDoS-Publish_Flood |       0.027 |         0.049 |        0.025 |         0.025 |  0.032 |
| MQTT-DoS-Publish_Flood  |       0.023 |         0.059 |        0.045 |         0.028 |  0.039 |
| Recon-Port_Scan         |       0.045 |         0.017 |        0.053 |         0.044 |  0.04  |
| MQTT-Malformed_Data     |       0.045 |         0.024 |        0.068 |         0.034 |  0.043 |
| TCP_IP-DoS-TCP          |       0.04  |         0.031 |        0.065 |         0.037 |  0.043 |
| TCP_IP-DoS-ICMP         |       0.049 |         0.033 |        0.062 |         0.04  |  0.046 |
| TCP_IP-DDoS-TCP         |       0.051 |         0.033 |        0.068 |         0.043 |  0.049 |
| Recon-VulScan           |       0.068 |         0.043 |        0.067 |         0.045 |  0.056 |
| TCP_IP-DDoS-ICMP        |       0.078 |         0.035 |        0.071 |         0.043 |  0.057 |
| Recon-OS_Scan           |       0.059 |         0.055 |        0.055 |         0.059 |  0.057 |
| ARP_Spoofing            |       0.077 |         0.051 |        0.061 |         0.043 |  0.058 |
| TCP_IP-DDoS-UDP         |       0.109 |         0.089 |        0.127 |         0.073 |  0.099 |
| TCP_IP-DoS-UDP          |       0.113 |         0.124 |        0.144 |         0.097 |  0.12  |
| Recon-Ping_Sweep        |       0.428 |         0.365 |        0.427 |         0.386 |  0.401 |