# V4 support: Benign flag rate, test and validation (exploratory, post hoc)

Script: p4_rev_v4_benign_disclosure.py. Source of the numbers quoted in v4_proposed_changes. Per-run file: benign_flag_rate_per_run.csv; summary: benign_flag_rate_summary.csv.

```
                                             0          1            2
model                               bilstm_dnn  trace_moe  transformer
test_flag_rate                          0.4623     0.8748       0.9808
test_flag_rate_lo                       0.3816     0.8089       0.9766
test_flag_rate_hi                       0.5472     0.9408       0.9863
val_flag_rate                           0.4606     0.8798       0.9929
val_flag_rate_lo                        0.3393     0.8177         0.99
val_flag_rate_hi                        0.5833     0.9442       0.9954
benign_share_of_known                   0.0242     0.0242       0.0242
benign_share_of_known_lo                0.0242     0.0242       0.0242
benign_share_of_known_hi                0.0242     0.0242       0.0242
test_flag_rate_pooled_windows           0.4623     0.8748       0.9808
val_flag_rate_pooled_windows            0.4606     0.8798       0.9929
val_overall_flag_rate                   0.0495       0.05         0.05
safety_view_alert_rate_benign_test      0.5015     0.8833       0.9973
sv_lo                                   0.4289     0.8202       0.9966
sv_hi                                   0.5795     0.9464        0.998
```
