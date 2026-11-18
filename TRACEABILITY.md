# Traceability: paper tables and figures to release artefacts

Maps each table and figure of the paper to the result file it comes from and the script that produced it.
All paths are relative to the root of this release. Per-run `metrics.json` files and model checkpoints are not bundled;
rows that depend on them are marked (not bundled) and can be checked only against the summaries listed here.

| Paper item | Content | Source file(s) | Producing script |
|---|---|---|---|
| Table 1 | Architectures: parameters, MFLOPs per window | Analytic count (see paper) | none |
| Table 2 | TRACE-MoE hyperparameters and training settings | `src/trace_iomt/config.py`, `scripts/04_train_deep.py` | none |
| Table 3 | Hypotheses and decision rules | Manuscript | none |
| Table 4 (H1) | Clean macro-F1 change after adversarial fine-tuning | per-run `metrics.json` (not bundled) | `scripts/06_robustness.py` |
| Table 5 (H2, H3) | AUC-robustness and worst-case F1, five seeds | per-run `metrics.json` (not bundled) | `scripts/06_robustness.py` |
| Table 6 | Paired bootstrap 95% CIs | `results/bootstrap_cis.json`, `results/bootstrap_cis.md` | `scripts/06_robustness.py` |
| Table 7 (H4) | Macro-F1 at eps 0.25 per seed after fine-tuning, with TRACE-MoE rank | per-run `metrics.json` (not bundled) | `scripts/06_robustness.py` |
| Table 8 | Mean macro-F1 rank per budget after fine-tuning | per-run `metrics.json` (not bundled) | `scripts/06_robustness.py` |
| Table 9 | Router instability under PGD (clean-trained TRACE-MoE) | `results/e4_routing_stability/routing_instability.json` | `scripts/e4_routing_stability.py` |
| Table 10 | Transfer-attack matrix at eps 0.25 | per-run `transfer_matrix.json` (not bundled) | `scripts/06_robustness.py` |
| Table 11 | White-box and cross-architecture ASR by budget | `results/e6_transfer_multi_eps/transfer_multi_eps.json` | `scripts/e6_transfer_multi_eps.py` |
| Table 12, Fig 4 | PCRP discrepancy trajectories | `results/twin_h6/*_twin_h6_track.json`, `results/twin_h6/twin_h6_per_seed.csv` | `scripts/06_robustness.py` |
| Table 13 | H6 verdict | `results/twin_h6/twin_h6_verdict.csv`, `.json`, `.md` | `scripts/06_robustness.py` |
| Table 14 | Component ablation (seeds 43, 44) | `results/ablation_history/`, per-run `metrics.json` (not bundled) | `scripts/05_ablation.py`, `scripts/analyze_h5.py` |
| Table 15 | PCRP variants, ROC-AUC | `results/e5_pcrp_variants/pcrp_variants.json` | `scripts/e5_pcrp_variants.py` |
| Table 16 | PCRP true-positive rate at fixed false-positive rate | `results/saved_artifact_analyses/s1_pcrp_tpr_summary.csv`, `s1_pcrp_tpr_table.tex` | `scripts/p2_s1_pcrp_tpr.py` |
| Table 17 | PGD with restarts | `results/e1_pgd_restarts/restarts5_seed42.json` | `scripts/e1_pgd_restarts.py` |
| Table 18 | Class-wise F1 after fine-tuning | `results/saved_artifact_analyses/s2_classwise_long.csv` | `scripts/p2_s2_classwise.py` |
| Table 19 | Hypothesis verdict summary | Tables 4 to 6, 13, 14 | none |
| Fig 1 | Data-flow architecture | none (drawing) | `figures/gen_fig1_dataflow.py` |
| Fig 2 | Evaluation workflow | none (drawing) | `figures/gen_fig2_workflow.py` |
| Fig 3 | Robustness curves, mean +/- 1 SD over five seeds | per-run `metrics.json` (not bundled) | `figures/gen_fig3_robustness.py` |
| Fig 4 | PCRP discrepancy versus epsilon | `results/twin_h6/*_twin_h6_track.json` | `figures/gen_fig4_pcrp.py` |
| Fig 5 | Cost versus robustness | Tables 1 and 5 | `figures/gen_fig5_cost_robustness.py` |

## Statements in the text that rest on saved-artifact analyses

| Text item | Source file(s) | Producing script |
|---|---|---|
| Router instability versus eps 0.25 deficit (Spearman, five seeds) | `results/saved_artifact_analyses/s3_routing_vs_deficit.csv` | `scripts/p2_s3_routing_vs_deficit.py` |
| Conditional ASR bounds, per-class degradation, class variance, transfer asymmetry, clean-versus-attacked F1 | `results/saved_artifact_analyses/a_suite/` | `scripts/p2_a_suite_saved_artifacts.py` |
| PCRP score distributions and detection by attack class | `results/saved_artifact_analyses/s6*.csv`, `s6_analysis.md`, `results/e5_pcrp_perwindow/` | `scripts/p2_s6_pcrp_perwindow_export.py`, `scripts/p2_s6_analysis.py` |

## Checksums

`SHASUMS` lists the SHA-256 of every file in this release except itself. Verify with `sha256sum -c SHASUMS` from the release root.
