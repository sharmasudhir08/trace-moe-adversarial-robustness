# Traceability for the final manuscript (v6l, 21 November 2026)

Manuscript: "Unseen-Attack Detection in the Internet of Medical Things: A Leave-One-Attack-Out Evaluation of Sequence Models, Energy Scores and Feature-Space Profiles" (IEEE OJ-COMS, OJCOMS-07851-2026).

## 1. Fingerprints

| File | SHA-256 |
|---|---|
| paper4_ojcoms.tex (v6l) | `33deeed9f1202bd28261e9bf1b404320776e19fd70977185af2fa826d5c39902` |
| paper4_refs.bib | `780db18dc78e238316eeec843a833963a4b4b7273170eb76ebc72f72f6744bff` |
| Paper 4 v7.pdf (v6l compiled by the author on Overleaf) | `10a38731f69af3e611229ed2431f3e5cec12b4241ea55d93a58d7e802fbf4e4e` |
| paper4_ojcoms_overleaf_20261121_v6l.zip | `4f3120b758b0a2c8a6b37766c474e4fc65880906ae9c892ede79ba5c70b8a9f4` |

SHA-256 of every file of this folder: `SHASUMS`. Supplementary files: listed in `SUPPLEMENTARY_ANALYSES.md`.

## 2. Tables (automated row-wise check)

Each body row of Tables 1 to 5 of v6l was matched to its row in the CSV (key: model or baseline and held-out class; Table 1 and Table 5 by class). Every decimal number of the row was compared with the named CSV column (tolerance: half a unit of the last printed digit) and the verdict word was compared with the CSV verdict (the table adds the subtype in brackets).

| Table (tex label) | Source in `results/` | Rows | Decimal numbers compared | Verdict mismatches |
|---|---|---|---|---|
| tab:h1 | H1_safety_view.csv | 5 | 40 | 0 |
| tab:h3 | H3_trace_moe_vs_baselines.csv | 10 | 30 | 0 |
| tab:h2 | H2_energy_vs_msp.csv | 15 | 45 | 0 |
| tab:h4 | H4_closed_set_degradation.csv | 15 | 45 | 0 |
| tab:h5 | twin_h5_verdict.csv | 5 | 20 | 0 |

Total: 50 rows, 180 decimal numbers compared, **0 mismatches** (none).


Integer cells (the number of held-out windows N in Table 1) are not covered by this check; the window counts were compared separately with `per_run.csv` (identical across seeds and architectures). All tables of v6l come from `results/` CSVs; the exploratory tables of earlier versions are written as text in v6l.

## 3. Figures

| Figure label | File in the manuscript package | SHA-256 (start) | Identical to `results/`-side source |
|---|---|---|---|
| fig:arch | sharm1.png | 4ecbf90156d9fe7b | n/a |
| fig:h3 | figures_v1_4/fig2_h3_architecture_comparison.pdf | 588828c83f0b3a40 | True |

Figure 1 (architecture schematic, `sharm1.png` in the manuscript package, not in this folder) is a drawing, not a result; its content was compared with `src/trace_iomt/models/networks.py` in the 21 November audit. Its origin file is not recorded here.

## 4. Prose numbers

Numbers in the prose were checked on 21 November in two manual audits (about 60 numbers recomputed from `results/`, `supplementary_analyses/` and the saved score exports). These audits are not exhaustive. Where a prose number comes from a supplementary analysis, the source is listed here:

| Statement in the text | Source |
|---|---|
| Benign alert rates (0.88, 0.997, 0.50; 0.46, 0.88, 0.98) | `supplementary_analyses/from_20261120/V4/benign_flag_rate_summary.csv` |
| Event rule: 2.3 events per 10,000 Benign windows, 0.998 (windows with a full 64-window history), 0.983 (all held-out windows) | `supplementary_analyses/from_20261120/T9/` (0.998) and `supplementary_analyses/T9/event_level.csv` (0.983) |
| Decomposition of the safety view | `supplementary_analyses/T1/decomposition_per_seed.csv`, `arp_seed_diagnosis.csv` |
| Energy temperature sensitivity | `supplementary_analyses/T3/per_run_feature_scores.csv` (float32 columns) |
| Spearman correlation of energy and 1 - MSP (ESA2) | `supplementary_analyses/T11/esa2_spearman.csv` |
| FPR@95 pooled values | `supplementary_analyses/T2/per_run_operating_points.csv` |
| Further open-set scores (ESA3) | `supplementary_analyses/T3/`, `supplementary_analyses/part2/U10/`, `part2/U2/`, `part2/pooled_table_for_manuscript.csv` |
| Absorption of held-out windows by closely related classes | `supplementary_analyses/part2/U1/absorption_19.csv` |
| Parameter counts, checkpoint sizes, FLOPs | `results_v3_paper3/zero_day/<run>/metrics.json` and `model.pt` sizes (not in this repository), `supplementary_analyses/T8/` |
| Inference timing | `results_v3_paper3/inference_benchmark_20261118/` (not in this repository; summary in the manuscript) |

## 5. Supplementary analyses

All are post hoc and exploratory; none changes a verdict. Task folders under `supplementary_analyses/`: T1, T2, T3, T4, T5, T6, T8, T9, T10, T11, U3, and `part2/` (U1, U2, U4 to U12). Each has a README with command, script SHA-256, inputs and checks. Claims and evidence: `supplementary_analyses/part2/claims_part2.md`.

## 6. References

27 cite keys are used in v6l; all are present in `paper4_refs.bib` (static check). Metadata of keys other than ref34 to ref39 was verified on 16 November (see the earlier verification record); metadata of ref34 to ref39 (added on 21 November) has NOT been verified against the sources.

| Key | Title (start) | Added 21 Nov, not verified |
|---|---|---|
| ref1 | CICIoMT2024: Attack Vectors in Healthcare Devices, a Multi-Protocol Dataset for Assessing IoMT  |  |
| ref2 | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer |  |
| ref3 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity |  |
| ref4 | Attention Is All You Need |  |
| ref5 | Long Short-Term Memory |  |
| ref6 | A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks |  |
| ref7 | Energy-Based Out-of-Distribution Detection |  |
| ref9 | A Novel Open Set Energy-based Flow Classifier for Network Intrusion Detection |  |
| ref10 | Enhancing Trustworthiness in ML-Based Network Intrusion Detection with Uncertainty Quantificati |  |
| ref12 | A Comprehensive and Systematic Literature Review on Intrusion Detection Systems in the Internet |  |
| ref13 | Enhancing IoMT Network Threat Detection with Data Balancing for Multi-Class Attack Classificati |  |
| ref14 | HBiLD-IDS: An Efficient Hybrid BiLSTM-DNN Model for Real-Time Intrusion Detection in IoMT Netwo |  |
| ref16 | Explainable Zero-Day Attack Detection in IoMT Using Transformer-Based Time-Series Modeling |  |
| ref20 | Evaluating and Enhancing Intrusion Detection Systems in IoMT: The Importance of Domain-Specific |  |
| ref24 | Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation |  |
| ref26 | Focal Loss for Dense Object Detection |  |
| ref27 | Long-Tail Learning via Logit Adjustment |  |
| ref29 | Dos and Don'ts of Machine Learning in Computer Security |  |
| ref31 | A Hierarchical IDS for Zero-Day Attack Detection in Internet of Medical Things Networks |  |
| ref32 | FEVER-OOD: Free Energy Vulnerability Elimination for Robust Out-of-Distribution Detection |  |
| ref33 | Energy-Based Hopfield Boosting for Out-of-Distribution Detection |  |
| ref34 | A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks | yes |
| ref35 | A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices | yes |
| ref36 | Out-of-Distribution Detection with Deep Nearest Neighbors | yes |
| ref37 | ReAct: Out-of-Distribution Detection with Rectified Activations | yes |
| ref38 | ViM: Out-of-Distribution with Virtual-Logit Matching | yes |
| ref39 | GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection | yes |

## 7. Not covered

- Each held-out class has one test file; a file-level cluster bootstrap is undefined (see `supplementary_analyses/T6/`).
- The device-level audit of the raw files was not done (T7 not approved).
- Training runs themselves are not reproduced here. The hypotheses, decision rules and timeline are in `ANALYSIS_PLAN.md`.

## 8. Files held back from `results/`

The following files of the internal verdict folder were not copied because they describe earlier charter versions or deferred analyses: `ESA2_DEFERRED.md`, `SURPRISES_v1_0_to_v1_1.md`, `SURPRISES_v1_1_3seed_to_v1_3_5seed.md`, `VERDICT_SUMMARY_v1_1.md`, `VERDICT_SUMMARY_v1_3_5seed.md`. Older copies of `results/` and `figures/` are not part of this folder.

## 9. Code

`src/trace_iomt/` holds the 17 modules imported by the Paper 4 scripts (models in `models/networks.py`, `heads.py`, `factory.py`; data pipeline in `data/`; training in `train.py`; abstention in `abstain.py`). `scripts/` holds the run, verdict, table and figure scripts (`07_unknown_attacks.py` training and evaluation; `24_` and `25_twin_*` the H5 profile; `paper4_h8_verdict.py` the H1 to H4 and ESA1 aggregation; `paper4_build_table.py`; `paper4_figures_v1_4.py`). They were copied with text-only edits (comments, messages and generated markdown; no computation changed); SHA-256 values of the files in this folder are in `SHASUMS`. The H4 reference (full 19-class models) comes from a separate earlier run set that is not part of this folder.
