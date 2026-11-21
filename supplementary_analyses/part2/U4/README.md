# U4 (post hoc, exploratory, CPU, no training)

Command: python scripts/p4_rev2_batch2.py --date 20261121 --tasks U4
Script SHA-256: f664f9ee387f9f0e6dff3eeedc9ae53e734cbbe00269a42d1147eb7f21f43aaf
Inputs: results_v3_paper3/score_export, zero_day/<run>/predictions.npz, part2/meta_cache (rebuilt test-window metadata). Runs read: 25. rng seed 42, bootstrap 10000 over seeds.
Started 2026-11-21 19:37:27, ended 2026-11-21 19:37:39.

TRACE-MoE only (25 runs). Routing = expert chosen by the router on each test window (saved `route`); protocol = token of the last row (rebuilt). No validation routing is saved, so p(expert | protocol) is estimated on KNOWN test windows only, with two-fold cross-fitting (fold = (position // 10000) % 2, a known window is scored with the other fold's estimate; held-out windows are scored with the estimate from all known windows), Laplace smoothing 1; no held-out window enters the estimate. Rarity = -log p(route | protocol); AUROC held-out against known.

Per class (mean over seeds, seed interval):

| held_out_class          |   routing_rarity_auroc |   routing_rarity_auroc_lo |   routing_rarity_auroc_hi |   energy_auroc |   energy_auroc_lo |   energy_auroc_hi |   mean_rarity_heldout |   mean_rarity_heldout_lo |   mean_rarity_heldout_hi |   mean_rarity_known |   mean_rarity_known_lo |   mean_rarity_known_hi |   n_seeds |
|:------------------------|-----------------------:|--------------------------:|--------------------------:|---------------:|------------------:|------------------:|----------------------:|-------------------------:|-------------------------:|--------------------:|-----------------------:|-----------------------:|----------:|
| ARP_Spoofing            |                 0.8289 |                    0.7506 |                    0.8981 |         0.9658 |            0.9367 |            0.9855 |                2.5227 |                   2.04   |                   3.0119 |              0.782  |                 0.7323 |                 0.84   |         5 |
| MQTT-DDoS-Publish_Flood |                 0.8868 |                    0.8268 |                    0.9475 |         0.3256 |            0.2169 |            0.4343 |                2.0562 |                   1.4996 |                   2.6127 |              0.7543 |                 0.6302 |                 0.8784 |         5 |
| MQTT-Malformed_Data     |                 0.8222 |                    0.7179 |                    0.9265 |         0.981  |            0.9746 |            0.9856 |                2.0246 |                   1.5214 |                   2.5278 |              0.8825 |                 0.8261 |                 0.9322 |         5 |
| Recon-VulScan           |                 0.7384 |                    0.6458 |                    0.8361 |         0.9692 |            0.9572 |            0.9782 |                1.561  |                   1.1546 |                   1.9927 |              0.8608 |                 0.7514 |                 0.9761 |         5 |
| TCP_IP-DoS-UDP          |                 0.5439 |                    0.2147 |                    0.8731 |         0.3139 |            0.1412 |            0.5283 |                1.5512 |                   0.2727 |                   2.8297 |              0.7684 |                 0.6699 |                 0.8723 |         5 |

Spearman correlation across the five classes between routing-rarity AUROC and energy AUROC (descriptive, n = 5): rho = 0.100, p = 0.873.
