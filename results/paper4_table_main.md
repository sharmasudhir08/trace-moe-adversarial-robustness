> **LEGACY LAYOUT, READ FIRST (added 2026-11-18).** Regenerated on the v1.4 run set (all 75 runs, 50-epoch protocol; Charter v1.4 is a DRAFT, not locked). The builder script `paper4_build_table.py` was written for 3 seeds: its panel headings say 3 seeds and its per-seed columns show seeds 42 to 44 only, while the mean, CI and verdict columns come from the 5-seed CSVs. Do not cite this file. Use the CSVs in this folder, `S1_before_after.md` and `figures_v1_4/`. Previous version: `../h8_verdict_v1_3_5seed_epoch_mixed_superseded/paper4_table_main.md`.

# Paper 4 Main Verdict Table (4-panel, 5-seed aggregation)

**Charter**: v1.3 (SHA-verified); statistical recipe v1.1; seed scope extended per v1.2; SHA256 `130767067f36261dca0370a12c418da70bf081a249ce380f6df613508f675a42`

**Source CSVs** (read-only inputs; byte-identical to the aggregator output recorded in verdict.json, generated 2026-11-18T05:44:11.255124+00:00):

- `H1_safety_view.csv` (5 pairs)
- `H2_energy_vs_msp.csv` (15 pairs)
- `H3_trace_moe_vs_baselines.csv` (10 pairs)
- `H4_closed_set_degradation.csv` (15 pairs)
- `ESA1_calibration.csv` (15 pairs, direction-only)

**Total verdict cells rendered**: 45 (= 5 + 15 + 10 + 15).

**Watch-item TCP_IP-DoS-UDP check**: 2 of 6 INCONCLUSIVE verdicts across H1-H4 involve TCP_IP-DoS-UDP as the held-out attack.

---

## Panel 1 of 4 — H1 safety-view detection rate (TRACE-MoE caught_any across 5 held-out classes x 3 seeds)

Rule: SUPPORTED (strong) if mean caught_any >= 0.95 AND CI lower bound >= 0.90.

| Held-out class | seed42 | seed43 | seed44 | mean | 95% CI | width | Verdict |
|---|---|---|---|---|---|---|---|
| ARP_Spoofing | 0.6408 | 0.9988 | 1.0000 | 0.9239 | [0.7808, 0.9992] | 0.2184 | INCONCLUSIVE |
| Recon-VulScan | 0.9990 | 1.0000 | 1.0000 | 0.9998 | [0.9994, 1.0000] | 0.0006 | SUPPORTED (strong) |
| MQTT-Malformed_Data | 1.0000 | 0.9913 | 1.0000 | 0.9941 | [0.9858, 1.0000] | 0.0142 | SUPPORTED (strong) |
| TCP_IP-DoS-UDP | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] | 0.0000 | SUPPORTED (strong) |
| MQTT-DDoS-Publish_Flood | 1.0000 | 0.9999 | 1.0000 | 1.0000 | [0.9999, 1.0000] | 0.0001 | SUPPORTED (strong) |

H1 breakdown over 5 class pairs: SUPPORTED (strong) 4, SUPPORTED (preliminary) 0, INCONCLUSIVE 1, NOT SUPPORTED 0.

## Panel 2 of 4 — H2 energy AUROC minus MSP AUROC (3 models x 5 held-out classes = 15 pairs)

Rule: paired bootstrap 95% CI of (energy_auroc minus msp_auroc) across 3 seeds. Expected effect 0.05 AUROC, CI width margin 0.10 (= 2 x expected). 4-tier verdict per Charter v1.1.

| Model | Held-out class | diff s42 | diff s43 | diff s44 | mean_diff | 95% CI | width | Verdict |
|---|---|---|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | -0.0478 | 0.0014 | 0.0030 | -0.0027 | [-0.0276, 0.0174] | 0.0450 | NOT SUPPORTED (narrow null) |
| trace_moe | Recon-VulScan | -0.0029 | -0.0103 | -0.0045 | -0.0023 | [-0.0070, 0.0032] | 0.0101 | NOT SUPPORTED (narrow null) |
| trace_moe | MQTT-Malformed_Data | 0.0014 | -0.0066 | 0.0067 | -0.0006 | [-0.0067, 0.0055] | 0.0121 | NOT SUPPORTED (narrow null) |
| trace_moe | TCP_IP-DoS-UDP | -0.3558 | -0.0716 | -0.1207 | -0.1286 | [-0.2563, -0.0238] | 0.2325 | NOT SUPPORTED (baseline wins) |
| trace_moe | MQTT-DDoS-Publish_Flood | 0.1269 | -0.2888 | -0.1322 | -0.0537 | [-0.1933, 0.0858] | 0.2790 | INCONCLUSIVE |
| transformer | ARP_Spoofing | 0.0075 | 0.0358 | 0.0241 | 0.0207 | [0.0109, 0.0304] | 0.0195 | SUPPORTED |
| transformer | Recon-VulScan | 0.0885 | 0.0247 | 0.0100 | 0.0398 | [0.0175, 0.0681] | 0.0505 | SUPPORTED |
| transformer | MQTT-Malformed_Data | 0.0032 | 0.0279 | 0.0734 | 0.0261 | [0.0080, 0.0510] | 0.0431 | SUPPORTED |
| transformer | TCP_IP-DoS-UDP | 0.0042 | -0.0208 | 0.1196 | 0.0888 | [0.0010, 0.2091] | 0.2082 | SUPPORTED |
| transformer | MQTT-DDoS-Publish_Flood | -0.0681 | 0.0040 | -0.2520 | -0.1811 | [-0.2938, -0.0684] | 0.2254 | NOT SUPPORTED (baseline wins) |
| bilstm_dnn | ARP_Spoofing | -0.0715 | -0.2122 | -0.0447 | -0.1174 | [-0.1697, -0.0701] | 0.0996 | NOT SUPPORTED (baseline wins) |
| bilstm_dnn | Recon-VulScan | -0.0503 | -0.1765 | -0.1797 | -0.1236 | [-0.1774, -0.0676] | 0.1097 | NOT SUPPORTED (baseline wins) |
| bilstm_dnn | MQTT-Malformed_Data | -0.0727 | -0.0351 | 0.0056 | -0.0649 | [-0.1228, -0.0188] | 0.1040 | NOT SUPPORTED (baseline wins) |
| bilstm_dnn | TCP_IP-DoS-UDP | -0.3448 | -0.1041 | 0.1755 | -0.0775 | [-0.2272, 0.0714] | 0.2987 | INCONCLUSIVE |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | 0.0446 | -0.2937 | 0.1386 | -0.1082 | [-0.2703, 0.0540] | 0.3243 | INCONCLUSIVE |

H2 breakdown for trace_moe: SUPPORTED (strong) 0, SUPPORTED (preliminary) 0, SUPPORTED 0, INCONCLUSIVE 1, NOT SUPPORTED (narrow null) 3, NOT SUPPORTED (baseline wins) 1.
H2 breakdown for transformer: SUPPORTED (strong) 0, SUPPORTED (preliminary) 0, SUPPORTED 4, INCONCLUSIVE 0, NOT SUPPORTED (narrow null) 0, NOT SUPPORTED (baseline wins) 1.
H2 breakdown for bilstm_dnn: SUPPORTED (strong) 0, SUPPORTED (preliminary) 0, SUPPORTED 0, INCONCLUSIVE 2, NOT SUPPORTED (narrow null) 0, NOT SUPPORTED (baseline wins) 3.

## Panel 3 of 4 — H3 TRACE-MoE energy AUROC minus baseline energy AUROC (2 baselines x 5 held-out classes = 10 pairs)

Rule: paired bootstrap 95% CI of (trace_moe.energy_auroc minus baseline.energy_auroc) across 3 seeds. Expected effect 0.05 AUROC, CI width margin 0.10. 4-tier verdict per Charter v1.1.

| Baseline | Held-out class | TRACE-MoE s42 | TRACE-MoE s43 | TRACE-MoE s44 | baseline s42 | baseline s43 | baseline s44 | mean_diff | 95% CI | width | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer | ARP_Spoofing | 0.9093 | 0.9780 | 0.9862 | 0.9852 | 0.9767 | 0.9850 | -0.0163 | [-0.0467, 0.0009] | 0.0475 | NOT SUPPORTED (narrow null) |
| transformer | Recon-VulScan | 0.9686 | 0.9460 | 0.9714 | 0.9812 | 0.9766 | 0.9773 | -0.0051 | [-0.0191, 0.0082] | 0.0273 | NOT SUPPORTED (narrow null) |
| transformer | MQTT-Malformed_Data | 0.9870 | 0.9809 | 0.9828 | 0.9821 | 0.9817 | 0.9704 | -0.0006 | [-0.0102, 0.0071] | 0.0173 | NOT SUPPORTED (narrow null) |
| transformer | TCP_IP-DoS-UDP | 0.1971 | 0.2464 | 0.0573 | 0.8433 | 0.6467 | 0.6859 | -0.3535 | [-0.6009, 0.0117] | 0.6126 | INCONCLUSIVE |
| transformer | MQTT-DDoS-Publish_Flood | 0.4904 | 0.1571 | 0.2192 | 0.0880 | 0.0251 | 0.1797 | 0.2097 | [0.1120, 0.3130] | 0.2010 | SUPPORTED |
| bilstm_dnn | ARP_Spoofing | 0.9093 | 0.9780 | 0.9862 | 0.9017 | 0.7675 | 0.9202 | 0.1084 | [0.0484, 0.1700] | 0.1215 | SUPPORTED |
| bilstm_dnn | Recon-VulScan | 0.9686 | 0.9460 | 0.9714 | 0.8101 | 0.6981 | 0.6360 | 0.2413 | [0.1660, 0.3166] | 0.1507 | SUPPORTED |
| bilstm_dnn | MQTT-Malformed_Data | 0.9870 | 0.9809 | 0.9828 | 0.8988 | 0.9473 | 0.9724 | 0.0753 | [0.0255, 0.1424] | 0.1169 | SUPPORTED |
| bilstm_dnn | TCP_IP-DoS-UDP | 0.1971 | 0.2464 | 0.0573 | 0.4744 | 0.7358 | 0.7260 | -0.3511 | [-0.5995, -0.0301] | 0.5694 | NOT SUPPORTED (baseline wins) |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | 0.4904 | 0.1571 | 0.2192 | 0.3590 | 0.2219 | 0.5942 | -0.0085 | [-0.2076, 0.1361] | 0.3437 | INCONCLUSIVE |

H3 breakdown vs transformer: SUPPORTED (strong) 0, SUPPORTED (preliminary) 0, SUPPORTED 1, INCONCLUSIVE 1, NOT SUPPORTED (narrow null) 3, NOT SUPPORTED (baseline wins) 0.
H3 breakdown vs bilstm_dnn: SUPPORTED (strong) 0, SUPPORTED (preliminary) 0, SUPPORTED 3, INCONCLUSIVE 1, NOT SUPPORTED (narrow null) 0, NOT SUPPORTED (baseline wins) 1.

## Panel 4 of 4 — H4 closed-set macro-F1 degradation (3 models x 5 held-out classes = 15 pairs, 10 pt threshold)

Rule: full-model baseline (recomputed over the SAME 18 non-held-out classes per pair) minus Phase 3(c) known_classes_macro_f1. 3-tier verdict: SUPPORTED if CI upper <= 0.10, NOT SUPPORTED if CI lower > 0.10, else INCONCLUSIVE. 10 pt threshold tightened from 15 pt in Charter v1.1.

| Model | Held-out class | degr s42 | degr s43 | degr s44 | mean_degr | 95% CI | width | Verdict |
|---|---|---|---|---|---|---|---|---|
| trace_moe | ARP_Spoofing | -0.0096 | -0.0235 | -0.0166 | -0.0184 | [-0.0247, -0.0125] | 0.0122 | SUPPORTED |
| trace_moe | Recon-VulScan | 0.0028 | 0.0027 | 0.0118 | 0.0005 | [-0.0066, 0.0069] | 0.0136 | SUPPORTED |
| trace_moe | MQTT-Malformed_Data | 0.0090 | 0.0142 | 0.0173 | 0.0052 | [-0.0049, 0.0144] | 0.0192 | SUPPORTED |
| trace_moe | TCP_IP-DoS-UDP | 0.0164 | 0.0140 | 0.0045 | 0.0065 | [-0.0001, 0.0131] | 0.0132 | SUPPORTED |
| trace_moe | MQTT-DDoS-Publish_Flood | -0.0018 | -0.0116 | -0.0107 | -0.0140 | [-0.0222, -0.0070] | 0.0152 | SUPPORTED |
| transformer | ARP_Spoofing | -0.0020 | -0.0121 | -0.0147 | -0.0238 | [-0.0428, -0.0086] | 0.0342 | SUPPORTED |
| transformer | Recon-VulScan | 0.0021 | -0.0179 | -0.0014 | -0.0094 | [-0.0164, -0.0024] | 0.0141 | SUPPORTED |
| transformer | MQTT-Malformed_Data | 0.0128 | 0.0005 | -0.0021 | -0.0077 | [-0.0242, 0.0054] | 0.0296 | SUPPORTED |
| transformer | TCP_IP-DoS-UDP | 0.0109 | -0.0102 | 0.0025 | -0.0054 | [-0.0223, 0.0077] | 0.0300 | SUPPORTED |
| transformer | MQTT-DDoS-Publish_Flood | 0.0007 | -0.0048 | -0.0082 | -0.0164 | [-0.0329, -0.0033] | 0.0297 | SUPPORTED |
| bilstm_dnn | ARP_Spoofing | 0.0061 | -0.0046 | -0.0079 | 0.0006 | [-0.0047, 0.0053] | 0.0100 | SUPPORTED |
| bilstm_dnn | Recon-VulScan | 0.0029 | -0.0124 | -0.0131 | -0.0051 | [-0.0114, 0.0013] | 0.0127 | SUPPORTED |
| bilstm_dnn | MQTT-Malformed_Data | -0.0020 | -0.0065 | -0.0019 | -0.0014 | [-0.0041, 0.0011] | 0.0052 | SUPPORTED |
| bilstm_dnn | TCP_IP-DoS-UDP | 0.0034 | 0.0028 | 0.0025 | 0.0019 | [0.0008, 0.0030] | 0.0021 | SUPPORTED |
| bilstm_dnn | MQTT-DDoS-Publish_Flood | -0.0128 | -0.0185 | -0.0162 | -0.0175 | [-0.0197, -0.0148] | 0.0049 | SUPPORTED |

H4 breakdown for trace_moe: SUPPORTED 5, INCONCLUSIVE 0, NOT SUPPORTED 0.
H4 breakdown for transformer: SUPPORTED 5, INCONCLUSIVE 0, NOT SUPPORTED 0.
H4 breakdown for bilstm_dnn: SUPPORTED 5, INCONCLUSIVE 0, NOT SUPPORTED 0.

## ESA1 calibration summary (direction-only report, per Charter v1.1)

| | Target (95% coverage) | Observed overall mean across 15 (model x class) pairs |
|---|---|---|
| known_false_alarm_rate | 0.0500 | 0.0456 |

Per-pair observed FAR is in `ESA1_calibration.csv`. Overall calibration drift from target is 0.0044 absolute, i.e. empirical reject layer is slightly conservative.

## Verdict label disambiguation audit (presentation layer, no Charter rule change)

Cells relabeled from raw 'NOT SUPPORTED' (as stored in the aggregator CSV) to the finer presentation label (as rendered in Panels 2 and 3 above):

| Hypothesis | Pair | Raw (CSV) | Final (table) |
|---|---|---|---|
| H2 | trace_moe x ARP_Spoofing | NOT SUPPORTED | NOT SUPPORTED (narrow null) |
| H2 | trace_moe x Recon-VulScan | NOT SUPPORTED | NOT SUPPORTED (narrow null) |
| H2 | trace_moe x MQTT-Malformed_Data | NOT SUPPORTED | NOT SUPPORTED (narrow null) |
| H2 | trace_moe x TCP_IP-DoS-UDP | NOT SUPPORTED | NOT SUPPORTED (baseline wins) |
| H2 | transformer x MQTT-DDoS-Publish_Flood | NOT SUPPORTED | NOT SUPPORTED (baseline wins) |
| H2 | bilstm_dnn x ARP_Spoofing | NOT SUPPORTED | NOT SUPPORTED (baseline wins) |
| H2 | bilstm_dnn x Recon-VulScan | NOT SUPPORTED | NOT SUPPORTED (baseline wins) |
| H2 | bilstm_dnn x MQTT-Malformed_Data | NOT SUPPORTED | NOT SUPPORTED (baseline wins) |
| H3 | vs transformer x ARP_Spoofing | NOT SUPPORTED | NOT SUPPORTED (narrow null) |
| H3 | vs transformer x Recon-VulScan | NOT SUPPORTED | NOT SUPPORTED (narrow null) |
| H3 | vs transformer x MQTT-Malformed_Data | NOT SUPPORTED | NOT SUPPORTED (narrow null) |
| H3 | vs bilstm_dnn x TCP_IP-DoS-UDP | NOT SUPPORTED | NOT SUPPORTED (baseline wins) |

**Count across all 45 cells**: NOT SUPPORTED (narrow null) 6, NOT SUPPORTED (baseline wins) 6.

This audit list is the reviewer-citable artefact proving the sub-tagging is deterministic from the CSV data and the Charter v1.1 narrow-null rule, not post-hoc reinterpretation.

## Legend

**4-tier verdict scheme per Charter v1.1** (locked SHA `e1d29fc2b3278515579599bd5895f62e1706b173169db46b5978d80c43f6caec`):

- **SUPPORTED (strong)**: CI excludes zero AND tight (width <= margin for H2/H3) OR mean meets threshold with CI tight (for H1).
- **SUPPORTED (preliminary)**: mean is favourable but CI width exceeds margin; honest small-sample reporting.
- **SUPPORTED**: CI excludes zero (plain tier used for H4 where no strong/prelim split applies).
- **INCONCLUSIVE**: CI straddles the decision boundary AND is wide (> margin). Evidence insufficient at 3-seed scope.
- **NOT SUPPORTED (narrow null)**: CI contains zero AND width <= margin (0.10 for H2/H3). The effect is precise enough to say it is below the practical-significance threshold even though the point estimate may be of either sign.
- **NOT SUPPORTED (baseline wins)**: CI excludes zero in the negative direction (CI upper < 0) AND mean < 0. The baseline strictly outperforms TRACE-MoE on this pair.

Charter-faithful disambiguation note: v1.1 §Verdict tiers already distinguishes these two cases in text. The aggregator CSV collapses them to a single 'NOT SUPPORTED' label; this table restores the distinction at render time without altering any verdict rule or re-running aggregation. No Charter SHA bump is required.

**Per-hypothesis margins (locked in Charter v1.1)**:

| Hypothesis | Decision quantity | Threshold or margin | Direction |
|---|---|---|---|
| H1 | caught_any (TRACE-MoE) | mean >= 0.95 AND CI lower >= 0.90 for strong | higher better |
| H2 | energy_auroc minus msp_auroc | expected effect 0.05 AUROC; CI width margin 0.10 (= 2 x expected) | higher better |
| H3 | trace_moe.energy_auroc minus baseline.energy_auroc | expected effect 0.05 AUROC; CI width margin 0.10 | higher better |
| H4 | Full-model baseline minus Phase 3(c) known_classes_macro_f1 | 10 pt threshold (tightened from 15 pt in v1.1) | lower better |

All CI columns are bootstrap 95% CIs over 5 seeds (10,000 resamples, fixed RNG seed per Charter).
