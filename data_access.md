# Data Access

## CICIoMT2024 Wi-Fi/MQTT Dataset

The dataset used in this study is publicly available from the Canadian Institute
for Cybersecurity (CIC):

> https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

**Reference:**
Dadkhah, S., Pinto Neto, E.C., Ferreira, R., Molokwu, R.C., Sadeghi, S.,
Ghorbani, A.A.: CICIoMT2024: A benchmark dataset for multi-protocol security
assessment in IoMT. Internet of Things 28, 101351 (2024).
https://doi.org/10.1016/j.iot.2024.101351

## Split and preprocessing

The official file-level split is used throughout: PCAP files comprising 80% of
captures form the training set, the remaining 20% the test set. Flow features are
robust-scaled using statistics fitted on training rows only; the last 15% of each
training file is held out for validation and temperature calibration.
Exact-duplicate training windows (144,682 removed) and test windows appearing in
training (29,478 removed for seed 42) are excluded before evaluation.

## What is NOT included in this release

Raw traffic captures (PCAP files), processed flow feature files, model
checkpoints, training history, and configuration files are not included due to
size and local-path constraints.
