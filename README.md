# Unseen-Attack Detection in the Internet of Medical Things: leave-one-attack-out evaluation

Companion repository of the manuscript "Unseen-Attack Detection in the Internet of Medical Things: A Leave-One-Attack-Out Evaluation of Sequence Models, Energy Scores and Feature-Space Profiles" (S. K. Sharma and S. Vasishth; submitted to the IEEE Open Journal of the Communications Society, OJCOMS-07851-2026).

## What this study does
TRACE-MoE (a mixture-of-experts sequence model), a Transformer and a BiLSTM-DNN are evaluated under a leave-one-attack-out (LOAO) open-set protocol on the Wi-Fi and MQTT subset of CICIoMT2024: five held-out attack classes, three architectures, five seeds (42 to 46), 75 independently trained models. Five hypotheses (H1 to H5) are evaluated with paired bootstrap 95% confidence intervals over seeds, plus exploratory supplementary analyses.

## For reviewers: verify a number
1. Find a claim in the manuscript (for example the H1 detection rate for ARP spoofing, mean 0.9239).
2. Open `results/H1_safety_view.csv` (or `H2_energy_vs_msp.csv`, `H3_trace_moe_vs_baselines.csv`, `H4_closed_set_degradation.csv`, `twin_h5_verdict.csv`, `ESA1_calibration.csv`) and read the row for that class and model.
3. Compare with the `mean`, `ci_lo` and `ci_hi` columns.
4. `TRACEABILITY.md` maps every table and figure to its source file and reports the automated check of all table values; `SHASUMS` lists the SHA-256 of every file in this folder.

## Layout
| Path | Content |
|---|---|
| `results/` | final hypothesis tables (CSV and markdown), per-seed values, verdict files, `S1_before_after` |
| `figures/` | figures of the manuscript and their source files (`fig1` to `fig4` and `fig_y_h5`, PDF and PNG) |
| `supplementary_analyses/` | post hoc exploratory analyses with a README each (see `SUPPLEMENTARY_ANALYSES.md`) |
| `scripts_supplementary/` | scripts of the supplementary analyses (and the score-export script they read) |
| `ANALYSIS_PLAN.md` | hypotheses, decision rules and timeline of the analysis plan |
| `TRACEABILITY.md` | table-by-table and figure-by-figure source map, check results, reference list |
| `graphical_abstract.png`, `graphical_abstract.pdf` | graphical abstract |
| `src/trace_iomt/` | model, data-pipeline, training and abstention code imported by the scripts (17 modules) |
| `scripts/` | `07_unknown_attacks.py` (leave-one-attack-out training and evaluation, GPU), `24_twin_score_paper4.py` and `25_twin_verdict_paper4.py` (H5 profile score and verdict), `paper4_h8_verdict.py` (H1 to H4 and ESA1 aggregation and verdicts), `paper4_build_table.py`, `paper4_figures_v1_4.py`, `paper4_figure_y_h5_twin.py`, `_common.py` |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | citation metadata |

## Dataset
CICIoMT2024 (Wi-Fi and MQTT subset), Dadkhah et al. 2024, Internet of Things 28, DOI 10.1016/j.iot.2024.101351; also via IEEE DataPort, DOI 10.21227/tq0p-ag21. The raw data are not redistributed.

## Not included
Model checkpoints, per-run predictions and training logs, preprocessed caches and the large per-window exports are not included because of their size; they are reproducible from the code. The training settings are given in Section IV-C of the manuscript.

## Citation and license
See `CITATION.cff` and `LICENSE`.
