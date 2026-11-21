# Supplementary exploratory analyses (post hoc, 21 November 2026)

All analyses read saved model outputs (T3 loads the saved models for inference only); none trains a model and none changes a verdict of H1 to H5. Each folder has a README with the command, the script SHA-256, the inputs and the checks.

| Folder | Content |
|---|---|
| T1 | per-seed decomposition of the safety view, ARP seed diagnosis |
| T2 | FPR at 95% TPR, TPR at fixed FPR, AUPR, OSCR (temperature columns are not valid; see T2/TEMPERATURE_NOTE.md) |
| T3 | feature-space scores and the float32 temperature grid |
| T4 | PCRP estimator and aggregation variants |
| T5 | calibration and risk-coverage |
| T6 | window-level bootstrap |
| T8 | FLOPs |
| T9 | alert rates and event-level rules |
| T10 | flood-class clusters |
| T11 | Spearman correlation of energy and 1 - MSP |
| U3 | fusion of energy and PCRP |
| part2/U1, U2, U4 to U12 | absorption, head disagreement, routing rarity, protocol-token breakdown, false-alarm sources, seed ensembles, time to detection, error detection, logit scores, per-class degradation, threshold quantiles |
| from_20261120/V4, T9 | Benign alert rates and the event-rule run used for the 0.998 figure |

Scripts: `scripts_supplementary/` (the scripts named in each task README, copied unmodified; the input score exports are too large to include). Large caches (window metadata, embeddings, PCRP caches) are not included. In the public copies of `p4_rev2_batch1.py` and `p4_rev_t8_flops.py` the wording of comments and one output label (`f1_reference_baseline`, U11) were edited; no computation changed. SHA-256 of the scripts as they were run: `p4_rev2_batch1.py` `33a6796f8072cf9ef314e9252ab43b07abcb46491386464814058b49a34582d7`, `p4_rev_t8_flops.py` `181e0b66abc420c5d9ef83519ae9cadd5e2fabe32793fbed20d9acf6ddea2cf0`.
