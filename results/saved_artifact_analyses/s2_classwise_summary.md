# S2: class-wise robustness after adversarial fine-tuning (mean F1 over five seeds)


## TRACE-MoE, eps 0.25: six lowest classes

| cls                     |   mean_f1 |   sd_f1 |   mean_support |
|:------------------------|----------:|--------:|---------------:|
| MQTT-DDoS-Publish_Flood |     0.18  |   0.024 |            263 |
| Recon-VulScan           |     0.253 |   0.048 |            263 |
| Recon-OS_Scan           |     0.359 |   0.061 |            263 |
| Recon-Port_Scan         |     0.596 |   0.017 |            263 |
| ARP_Spoofing            |     0.613 |   0.018 |            263 |
| Recon-Ping_Sweep        |     0.621 |   0.007 |            171 |

## TRACE-MoE, eps 0.5: six lowest classes

| cls                     |   mean_f1 |   sd_f1 |   mean_support |
|:------------------------|----------:|--------:|---------------:|
| TCP_IP-DoS-SYN          |     0.169 |   0.194 |            263 |
| MQTT-DDoS-Publish_Flood |     0.178 |   0.023 |            263 |
| Recon-VulScan           |     0.215 |   0.059 |            263 |
| TCP_IP-DoS-ICMP         |     0.233 |   0.272 |            263 |
| Recon-OS_Scan           |     0.314 |   0.059 |            263 |
| TCP_IP-DoS-UDP          |     0.327 |   0.3   |            263 |

## Class-wise F1 at eps 0.50 after fine-tuning, all models

| cls                     |   trace_moe |   transformer |   bilstm_dnn |   tfsage_lite |
|:------------------------|------------:|--------------:|-------------:|--------------:|
| ARP_Spoofing            |       0.476 |         0.444 |        0.267 |         0.565 |
| Benign                  |       0.609 |         0.517 |        0.641 |         0.445 |
| MQTT-DDoS-Connect_Flood |       0.708 |         0.837 |        0.372 |         0.835 |
| MQTT-DDoS-Publish_Flood |       0.178 |         0.178 |        0.176 |         0.162 |
| MQTT-DoS-Connect_Flood  |       0.827 |         0.841 |        0.528 |         0.739 |
| MQTT-DoS-Publish_Flood  |       0.687 |         0.689 |        0.643 |         0.681 |
| MQTT-Malformed_Data     |       0.681 |         0.527 |        0.388 |         0.563 |
| Recon-OS_Scan           |       0.314 |         0.305 |        0.078 |         0.339 |
| Recon-Ping_Sweep        |       0.61  |         0.584 |        0.394 |         0.578 |
| Recon-Port_Scan         |       0.541 |         0.537 |        0.356 |         0.559 |
| Recon-VulScan           |       0.215 |         0.258 |        0.128 |         0.261 |
| TCP_IP-DDoS-ICMP        |       0.379 |         0.18  |        0.363 |         0.358 |
| TCP_IP-DDoS-SYN         |       0.351 |         0.5   |        0.57  |         0.59  |
| TCP_IP-DDoS-TCP         |       0.785 |         0.378 |        0.4   |         0.33  |
| TCP_IP-DDoS-UDP         |       0.353 |         0.193 |        0.357 |         0.339 |
| TCP_IP-DoS-ICMP         |       0.233 |         0.191 |        0.364 |         0.386 |
| TCP_IP-DoS-SYN          |       0.169 |         0.228 |        0.516 |         0.461 |
| TCP_IP-DoS-TCP          |       0.826 |         0.437 |        0.381 |         0.266 |
| TCP_IP-DoS-UDP          |       0.327 |         0.224 |        0.313 |         0.188 |