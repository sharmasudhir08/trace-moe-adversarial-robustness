# Traceability — Paper to Release Artefacts

Maps every reported number, table and figure in the paper to the result
file it comes from and the script that produced it.
All paths are relative to the repository root.

## Release PDF

| File | SHA-256 | Date | Notes |
|---|---|---|---|
| `Paper 2.pdf` | `f76f7af1714848bef539d0c6e14614f989bc3a286d2065a029602190f1031054` | 2026-11-17 | Final compiled manuscript; pdfLaTeX via Overleaf (sn-jnl class) |

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
| Table 6 — bootstrap 95% CIs | 10,000 paired resamples, seed 42 | results/bootstrap_cis.json | scripts/06_robustness.py |
| Fig 2 — workflow | Evaluation design | Manuscript | — |
| Fig 3 — robustness curves | Macro-F1 vs ε, mean ± 1SD over seeds 42–46 | results/robustness/{model}_seed{42..46}/metrics.json | scripts/06_robustness.py |
| Fig 4 — PCRP trajectories | Mean Mahalanobis discrepancy D̄ vs ε | results/h6_pcrp/{model}_seed{42..46}_pcrp.json | scripts/06_robustness.py |
| Fig 5 — cost vs robustness | MFLOPs vs AUC-robustness (clean and adv-FT) | Table 1 (MFLOPs) + Table 5 (AUC) | — |
| §6.1 — gap 0.009 | TRACE-MoE 0.694 minus TF-SAGE-lite 0.685 | results/bootstrap_cis.json + Table 5 | scripts/06_robustness.py |
| §6.1 — spread 0.162 to 0.086 | Inter-model AUC-rob. spread before / after adv-FT | results/robustness/{model}_seed{42..46}/metrics.json | scripts/06_robustness.py |
| §6.1 — TF-SAGE-lite 4.6× lower cost | 11.54 / 2.51 = 4.60 | Table 1 MFLOPs | — |
| §6.1 — ablation raises AUC-rob. | +0.034 to +0.044 above TRACE-MoE full | results/ablation/*/metrics.json | scripts/analyze_h5.py |
| H5 verdict (Table 12 / §5.4) | NOT SUPPORTED — gap widens rather than closes | results/aggregated/h5_verdict.json | scripts/analyze_h5.py |
| H6 verdict (Table 11 / §5.6) | Spearman ρ = 1.000 all 20 pairs | results/twin_h6/twin_h6_verdict.csv | scripts/06_robustness.py |

## Figures (`figures/`)

| Paper figure | Published file | Generator |
|---|---|---|
| Fig 1 (data flow) | `figures/Fig1_dataflow.pdf` (+ `.tif`) | `figures/gen_fig1_dataflow.py` |
| Fig 2 (workflow) | `figures/Fig2_workflow.pdf` (+ `.tif`) | `figures/gen_fig2_workflow.py` |
| Fig 3 (robustness curves) | `figures/Fig3_robustness.pdf` (+ `.tif`) | `figures/gen_fig3_robustness.py` |
| Fig 4 (PCRP trajectories) | `figures/Fig4_pcrp.pdf` | PDF as compiled into the manuscript; generator not bundled |
| Fig 5 (cost vs robustness) | `figures/Fig5_cost_robustness.pdf` (+ `.tif`) | `figures/gen_fig5_cost_robustness.py` |

## Bundled CSV / summary files (`results/`)

| File | Content |
|---|---|
| `results/twin_h6/twin_h6_per_seed.csv` | H6 per-seed PCRP monotonicity results |
| `results/twin_h6/twin_h6_verdict.csv` / `.json` / `.md` | H6 verdict (Spearman rho, all 20 model-seed pairs) |
| `results/bootstrap_cis.json` / `.md` | Paired bootstrap 95% CIs (Table 6) |
| `results/robustness_variant/summary_adv.csv` | Adversarial fine-tuning robustness summary |
| `results/abstain_post_fix_summary.csv` | Post-fix abstention summary (2026-11-13 rescore) |
| `results/abstain_rescore_log_20261113_092548.csv` | Rescore log |
| `results/ablation_history/*_history.csv` | Per-epoch training history of the 22 ablation runs (seeds 43, 44) |

The per-run `metrics.json` files cited above (`results/robustness/`, `results/ablation/`) are not bundled in this release.
