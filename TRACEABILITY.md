# Traceability — Paper to Release Artefacts

Maps every reported number, table and figure in the paper to the result
file it comes from and the script that produced it.
All paths are relative to the repository root.

| Paper item | Claim / number | Source file(s) | Producing script |
|---|---|---|---|
| Abstract — AUC-robustness 0.694 | TRACE-MoE best AUC-rob. after adv-FT | results/robustness/trace_moe_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Abstract — worst-case F1 0.488 | TRACE-MoE worst-case F1 at ε=0.5 | results/robustness/trace_moe_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Abstract — H4 not supported | TRACE-MoE never ranks first at ε=0.25 | results/robustness/trace_moe_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Abstract — H5 not supported | Two-seed ablation does not attribute anomaly to routing | results/ablation/dense_expert_seed{43,44}/metrics.json | scripts/analyze_h5.py |
| Table 1 — MFLOPs/window | Architecture compute cost (11.54, 8.84, 3.07, 2.51) | Computed analytically (see paper §3.1) | — |
| Table 1 — parameters | Architecture parameter counts | Computed analytically (see paper §3.1) | — |
| Table 2 — TRACE-MoE components | Architecture description | Manuscript | — |
| Table 3 — Hypothesis definitions | H1–H6 pre-registered criteria | Manuscript / PAPER2_CHARTER | — |
| Table 4 (H1) — mean change, Std, largest drop | Adv-FT cost on clean accuracy | results/robustness/{model}_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Table 5 (H2+H3) — AUC-robustness, worst-case F1 | Mean ± SD over 5 seeds | results/robustness/{model}_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Table 6 — bootstrap 95% CIs | 10,000 paired resamples, seed 42 | results/aggregated/bootstrap_cis.json | scripts/06_robustness.py |
| Fig 2 — workflow | Evaluation design | Manuscript | — |
| Fig 3 — robustness curves | Macro-F1 vs ε, mean ± 1SD over seeds 42–46 | results/robustness/{model}_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Fig 4 — PCRP trajectories | Mean Mahalanobis discrepancy D̄ vs ε | results/h6_pcrp/{model}_seed{42..46}_pcrp.json | scripts/06_robustness.py |
| Fig 5 — cost vs robustness | MFLOPs vs AUC-robustness (clean and adv-FT) | Table 1 (MFLOPs) + Table 5 (AUC) | — |
| §6.1 — gap 0.009 | TRACE-MoE 0.694 minus TF-SAGE-lite 0.685 | results/aggregated/bootstrap_cis.json + Table 5 | scripts/06_robustness.py |
| §6.1 — spread 0.162 to 0.086 | Inter-model AUC-rob. spread before / after adv-FT | results/robustness/{model}_seed{42..46}/metrics.json | scripts/06_robustness.py |
| §6.1 — TF-SAGE-lite 4.6× lower cost | 11.54 / 2.51 = 4.60 | Table 1 MFLOPs | — |
| §6.1 — ablation raises AUC-rob. | +0.034 to +0.044 above TRACE-MoE full | results/ablation/*/metrics.json | scripts/analyze_h5.py |
| H5 verdict (Table 12 / §5.4) | NOT SUPPORTED — gap widens rather than closes | results/aggregated/h5_verdict.json | scripts/analyze_h5.py |
| H6 verdict (Table 11 / §5.6) | Spearman ρ = 1.000 all 20 pairs | results/aggregated/twin_h6_verdict.csv | scripts/06_robustness.py |
