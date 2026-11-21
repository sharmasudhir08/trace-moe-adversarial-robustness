# T9 safety-view false alarms and event-level alerting (post-hoc exploratory, descriptive)

Script `scripts/p4_rev_t5_t9.py` SHA-256 2bcdea256397197b8b334a33746a9be813f97e531cf4e06af1c42d56a1edc418; Python 3.12.4, numpy 2.5.3; rng 42; runs: 75 (expected 75); 2026-11-21 12:31:56.

Checks: max |held-out alert rate - A2_caught_any (Table II per seed)| = 1.11e-16; max |binary-only benign rate - A1_binary_fpr_on_benign| = 9.71e-17; event-level runs: 75; non-contiguous file blocks found (must be 0): 0.
