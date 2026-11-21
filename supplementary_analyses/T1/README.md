# T1 decomposition and ARP diagnosis (post-hoc exploratory, descriptive)

Script `scripts/p4_rev_saved_scores.py` SHA-256 5189b8f69c91518855135ad90430046203c8e50cd04ff83bc4a7b7c20153d0b1; Python 3.12.4, numpy 2.5.3, pandas 3.0.6; rng 42; runs read: 75 (expected 75); 2026-11-21 11:54:41.

Checks: max |sum of four cells - 1| = 3.33e-16; max |caught_any - (energy_only+binary_only+both)| = 2.22e-16; max |TRACE-MoE caught_any - H1 Table per seed| = 0.00e+00.

|   seed |    thr |   val_energy_p5 |   median_heldout_energy |   median_known_energy |   n_heldout |
|-------:|-------:|----------------:|------------------------:|----------------------:|------------:|
|     42 | 6.0628 |          6.0628 |                  6.3486 |               13.8594 |        1729 |
|     43 | 6.3157 |          6.3157 |                  4.2342 |               20.3281 |        1729 |
|     44 | 6.5161 |          6.5161 |                  4.2252 |               11.0391 |        1729 |
|     45 | 6.6837 |          6.6837 |                  5.5269 |               13.1328 |        1729 |
|     46 | 6.1012 |          6.1012 |                  3.4045 |               28.5469 |        1729 |
