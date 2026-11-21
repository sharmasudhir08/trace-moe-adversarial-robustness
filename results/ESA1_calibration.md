# ESA1 — Reject-threshold calibration quality

**Target coverage**: 0.95 → target `known_false_alarm_rate` = 0.05.

**Report** (direction-only; no formal verdict per Charter):
- Overall mean known_false_alarm_rate across 75 runs: 0.0456

## Per (model × class)

| Model | Class | far_s42 | far_s43 | far_s44 | far_s45 | far_s46 | mean | dev vs 0.05 | CI 95% |
|---|---|---|---|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | 0.0302 | 0.0440 | 0.0382 | 0.0451 | 0.0448 | 0.0404 | -0.0096 | [-0.0152, -0.0053] |
| trace_moe | Recon-VulScan | 0.0455 | 0.0363 | 0.0401 | 0.0421 | 0.0406 | 0.0409 | -0.0091 | [-0.0118, -0.0065] |
| trace_moe | MQTT-Malformed_Data | 0.0412 | 0.0365 | 0.0358 | 0.0330 | 0.0425 | 0.0378 | -0.0122 | [-0.0152, -0.0092] |
| trace_moe | TCP_IP-DoS-UDP | 0.0447 | 0.0409 | 0.0695 | 0.0382 | 0.0396 | 0.0466 | -0.0034 | [-0.0107, +0.0083] |
| trace_moe | MQTT-DDoS-Publish_Flood | 0.0451 | 0.0387 | 0.0536 | 0.0418 | 0.0380 | 0.0434 | -0.0066 | [-0.0110, -0.0012] |
| transformer | ARP_Spoofing | 0.0436 | 0.0435 | 0.0430 | 0.0619 | 0.0549 | 0.0494 | -0.0006 | [-0.0067, +0.0068] |
| transformer | Recon-VulScan | 0.0420 | 0.0423 | 0.0422 | 0.0428 | 0.0422 | 0.0423 | -0.0077 | [-0.0079, -0.0074] |
| transformer | MQTT-Malformed_Data | 0.0432 | 0.0414 | 0.0421 | 0.0426 | 0.0428 | 0.0424 | -0.0076 | [-0.0082, -0.0071] |
| transformer | TCP_IP-DoS-UDP | 0.0429 | 0.0434 | 0.0432 | 0.0443 | 0.0419 | 0.0431 | -0.0069 | [-0.0076, -0.0062] |
| transformer | MQTT-DDoS-Publish_Flood | 0.0429 | 0.0434 | 0.0425 | 0.0415 | 0.0428 | 0.0426 | -0.0074 | [-0.0080, -0.0069] |
| bilstm_dnn | ARP_Spoofing | 0.0454 | 0.0435 | 0.0409 | 0.0452 | 0.0427 | 0.0435 | -0.0065 | [-0.0079, -0.0051] |
| bilstm_dnn | Recon-VulScan | 0.0304 | 0.0900 | 0.0740 | 0.0239 | 0.1172 | 0.0671 | +0.0171 | [-0.0135, +0.0477] |
| bilstm_dnn | MQTT-Malformed_Data | 0.1287 | 0.0482 | 0.0467 | 0.0535 | 0.0384 | 0.0631 | +0.0131 | [-0.0066, +0.0462] |
| bilstm_dnn | TCP_IP-DoS-UDP | 0.0306 | 0.0406 | 0.0427 | 0.0422 | 0.0363 | 0.0385 | -0.0115 | [-0.0159, -0.0079] |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | 0.0522 | 0.0384 | 0.0574 | 0.0322 | 0.0340 | 0.0428 | -0.0072 | [-0.0159, +0.0017] |