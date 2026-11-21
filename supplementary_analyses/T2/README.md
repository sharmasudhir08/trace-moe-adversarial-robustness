# T2 operating points (post-hoc exploratory)

Script `scripts/p4_rev_saved_scores.py` SHA-256 5189b8f69c91518855135ad90430046203c8e50cd04ff83bc4a7b7c20153d0b1; Python 3.12.4, numpy 2.5.3, pandas 3.0.6; rng 42; runs read: 75 (expected 75); 2026-11-21 11:54:41.

Reproduction of saved values (max abs difference vs per_run.csv):

- energy_auroc_T1: 1.110e-16
- msp_auroc_T1: 1.110e-16
- maxlogit_auroc: 1.110e-16
- energy_T0.5: 7.869e-04
- energy_T2: 1.910e-04
- energy_T5: 8.507e-07

No temperature is selected or recommended. Higher score = more unknown; OSCR with argmax over known classes.
