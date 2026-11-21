# U5 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch2.py --date 20261121 --tasks U5
Script SHA-256: f664f9ee387f9f0e6dff3eeedc9ae53e734cbbe00269a42d1147eb7f21f43aaf
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz, part2/meta_cache (rebuilt test-window metadata). Runs read: 75. rng seed 42, bootstrap 10000 over seeds.
Started 2026-11-21 19:37:01, ended 2026-11-21 19:37:20.

Groups are the protocol token of the last row of each window (the token the router receives). The data contain no per-window Wi-Fi or MQTT flag (Benign is a single file that mixes both), so the Wi-Fi versus MQTT split of the brief is replaced by the transport token; MQTT runs over TCP. Checks: pooled known FAR equals ESA1 (max difference 9.71e-17) and pooled held-out caught-any equals A2_caught_any (max difference 1.11e-16); criterion 1e-6.

Known false-alarm rate by token, mean over classes and seeds (seed interval):

| model       | protocol_token   |   far_energy_known |   far_energy_known_lo |   far_energy_known_hi |   far_either_known |   far_either_known_lo |   far_either_known_hi |   n_seeds |
|:------------|:-----------------|-------------------:|----------------------:|----------------------:|-------------------:|----------------------:|----------------------:|----------:|
| bilstm_dnn  | ICMP             |             0.0201 |                0.0003 |                0.056  |             1      |                1      |                1      |         5 |
| bilstm_dnn  | TCP              |             0.0806 |                0.0625 |                0.1004 |             0.9797 |                0.9766 |                0.9829 |         5 |
| bilstm_dnn  | UDP              |             0.0155 |                0.0066 |                0.0326 |             0.9992 |                0.9989 |                0.9995 |         5 |
| bilstm_dnn  | mixed            |             0.2667 |                0.2322 |                0.3113 |             0.8788 |                0.8613 |                0.897  |         5 |
| bilstm_dnn  | other_IP         |             0.3798 |                0.3496 |                0.3997 |             0.9374 |                0.9233 |                0.9515 |         5 |
| trace_moe   | ICMP             |             0.0109 |                0.0077 |                0.0141 |             1      |                1      |                1      |         5 |
| trace_moe   | TCP              |             0.0674 |                0.0612 |                0.0746 |             0.9957 |                0.9934 |                0.9981 |         5 |
| trace_moe   | UDP              |             0.0075 |                0.0067 |                0.0084 |             0.9994 |                0.9987 |                1      |         5 |
| trace_moe   | mixed            |             0.2887 |                0.2647 |                0.3104 |             0.9703 |                0.9545 |                0.986  |         5 |
| trace_moe   | other_IP         |             0.4023 |                0.3548 |                0.4514 |             0.9547 |                0.9337 |                0.9782 |         5 |
| transformer | ICMP             |             0.0002 |                0      |                0.0004 |             1      |                1      |                1      |         5 |
| transformer | TCP              |             0.0781 |                0.075  |                0.0817 |             0.9999 |                0.9999 |                0.9999 |         5 |
| transformer | UDP              |             0.0082 |                0.008  |                0.0084 |             0.9999 |                0.9998 |                0.9999 |         5 |
| transformer | mixed            |             0.3067 |                0.3045 |                0.3083 |             0.999  |                0.9988 |                0.9992 |         5 |
| transformer | other_IP         |             0.5156 |                0.4925 |                0.5285 |             0.9996 |                0.9989 |                1      |         5 |

Window counts by token (summed over the 75 runs' windows, one model counted once per class and seed x3):

| protocol_token   |   n_known |   n_heldout |   share_known |
|:-----------------|----------:|------------:|--------------:|
| ICMP             |  31100601 |           0 |        0.2668 |
| TCP              |  46941099 |      162270 |        0.4028 |
| UDP              |  34702017 |     2017455 |        0.2977 |
| mixed            |   3799383 |       76485 |        0.0326 |
| other_IP         |      8145 |          30 |        0.0001 |
