# T9: safety-view false alarms and event-level alerting

**Status:** Exploratory, post hoc analysis; no change to H1-H5, their decision rules or verdicts. Test windows used for reporting only.

- Command: `python scripts/p4_rev_9_event_level.py`
- Script: `p4_rev_9_event_level.py`, SHA-256 `35836aace65361166e5e2f774496a54be4cb71048d9420d168cfba5320a3808f`
- Inputs: score_export/*.npz (energy, thr, p_attack, y19, hid)
- Canonical runs read: 75
- rng seed: 42
- Versions: Python 3.12.4, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
- Start 2026-11-20 11:38:17, end 2026-11-20 11:38:33, duration 16 s

## Result
Pass criteria: held-out caught-any at n = 1 equals Table II / A2 (max |difference| 1.1e-16); binary-only plus both on benign equals A1_binary_fpr_on_benign (max |difference| 9.7e-17).

Safety-view alert rates, mean over classes and seeds:
```
             alert_rate_benign  energy_only_benign  binary_only_benign  alert_rate_known_attack  alert_rate_heldout
model                                                                                                              
bilstm_dnn              0.5015              0.4380              0.0392                   0.9997              0.9172
trace_moe               0.8833              0.8586              0.0085                   0.9999              0.9836
transformer             0.9973              0.9530              0.0165                   1.0000              0.9988
```

Event-level rules (mean over classes and seeds; events per 10,000 Benign windows; window-level held-out detection):
```
                   benign_events_per_10k_windows  heldout_window_detection
model       n  m                                                          
bilstm_dnn  4  2                         212.445                     0.963
               4                         234.973                     0.808
            16 8                          64.071                     0.953
               16                         36.267                     0.662
            64 32                         18.674                     0.948
               64                          4.498                     0.508
trace_moe   4  2                          25.944                     0.997
               4                          47.854                     0.962
            16 8                           8.580                     0.996
               16                         19.714                     0.937
            64 32                          2.302                     0.998
               64                          5.063                     0.894
transformer 4  2                           1.085                     1.000
               4                          17.377                     0.995
            16 8                           0.277                     1.000
               16                          9.027                     0.984
            64 32                          0.266                     1.000
               64                          3.464                     0.950
```
No rule is recommended from these test results.

## Notes
File identity = contiguous label block (one test file per class), verified by the number of constant-label runs.
