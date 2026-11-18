# Adversarial Robustness of a Mixture-of-Experts Network Intrusion Detector: A Multi-Seed Evaluation with Protocol-Cluster Reference Profiles on Medical IoT Traffic

Code, figures, summary result files and saved-artifact analyses for the paper above (Sudhir Kumar Sharma and Sneha Vasishth, 2026). The compiled manuscript is not part of this repository.

## Overview

This repository contains the training and evaluation code for a study that benchmarks
four IoMT intrusion detectors under 20-step PGD attack on the CICIoMT2024 Wi-Fi/MQTT
dataset.

**Five seeds (42–46), six hypotheses (H1–H6).**

## Hypotheses

| ID | Topic | Verdict |
|---|---|---|
| H1 | Adversarial fine-tuning costs negligible clean accuracy (≤1 pt, all architectures) | Supported |
| H2 | TRACE-MoE has highest AUC-robustness after adversarial fine-tuning | Mixed |
| H3 | TRACE-MoE has highest worst-case macro-F1 at ε = 0.5 after adversarial fine-tuning | Mixed |
| H4 | TRACE-MoE ranks first at the single operating point ε = 0.25 | Not Supported |
| H5 | The ε = 0.25 rank anomaly is attributable to MoE routing | Not Supported |
| H6 | PCRP discrepancy rises strictly monotonically with PGD budget | Supported |

## Repository structure

```
├── figures
│   ├── Fig1_dataflow.pdf / .tif
│   ├── Fig2_workflow.pdf / .tif
│   ├── Fig3_robustness.pdf / .tif
│   ├── Fig4_pcrp.pdf / .tif
│   ├── Fig5_cost_robustness.pdf / .tif
│   └── gen_fig1_dataflow.py, gen_fig2_workflow.py, gen_fig3_robustness.py, gen_fig4_pcrp.py, gen_fig5_cost_robustness.py
├── results
│   ├── twin_h6/                      (H6 per-seed and verdict CSV/JSON/MD)
│   ├── robustness_variant/summary_adv.csv
│   ├── ablation_history/             (22 per-epoch history CSVs, seeds 43 and 44)
│   ├── abstain_post_fix_summary.csv
│   ├── abstain_rescore_log_20261113_092548.csv
│   ├── bootstrap_cis.json / .md
│   ├── e1_pgd_restarts/, e4_routing_stability/, e5_pcrp_variants/, e6_transfer_multi_eps/   (summaries behind Tables 9, 11, 15 and 17)
│   ├── e5_pcrp_perwindow/            (per-window PCRP scores, labels and predictions; 20 .npz files with consistency checks)
│   └── saved_artifact_analyses/      (S1 to S3, S6 and the A-suite: CSV and Markdown outputs)
├── scripts
│   ├── 04_train_deep.py
│   ├── 05_ablation.py
│   ├── 06_robustness.py
│   ├── _common.py
│   ├── analyze_h5.py
│   ├── e1_pgd_restarts.py, e4_routing_stability.py, e5_pcrp_variants.py, e6_transfer_multi_eps.py
│   └── p2_s1 to p2_s3, p2_s6 (two scripts), p2_a_suite (analyses on saved artifacts)
├── src
│   └── trace_iomt
│       ├── data
│       │   ├── __init__.py
│       │   ├── labels.py
│       │   ├── loaders.py
│       │   ├── preprocess.py
│       │   └── windows.py
│       ├── edge_iiotset_native
│       │   ├── __init__.py
│       │   ├── labels.py
│       │   ├── loaders.py
│       │   ├── preprocess.py
│       │   └── schema.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── classical.py
│       │   ├── factory.py
│       │   ├── heads.py
│       │   └── networks.py
│       ├── __init__.py
│       ├── abstain.py
│       ├── attacks.py
│       ├── audit.py
│       ├── compare.py
│       ├── config.py
│       ├── efficiency.py
│       ├── experiment.py
│       ├── metrics.py
│       ├── reporting.py
│       ├── result_doc.py
│       ├── train.py
│       └── utils.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── NOTICE
├── SHASUMS
├── TRACEABILITY.md
├── data_access.md
└── requirements.txt
```

## Training settings

| Parameter | Value |
|---|---|
| Seeds | 42, 43, 44, 45, 46 |
| Window length T | 16 consecutive flows |
| Training windows per class | Up to 60,000 (cap per class) |
| PGD steps | 20 |
| PGD step size α (evaluation) | 0.1ε |
| PGD norm | ℓ∞ |
| Mutable features | 18 |
| Adversarial fine-tuning epochs | 3 |
| Adversarial fine-tuning budget | ε drawn uniformly from [0, 0.25] per batch |
| Adversarial fine-tuning sample | Class-balanced, up to 15,000 training windows per class per epoch |
| Adversarial fine-tuning PGD | 5 steps, α = 0.2ε |
| TRACE-MoE d (encoder dim) | 128 |
| TRACE-MoE expert_hidden | 256 |
| TRACE-MoE experts E | 4 |
| TRACE-MoE routing | Top-1 (Switch-style) |
| Ablation seeds | 43, 44 |

## Steps to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the CICIoMT2024 dataset — see `data_access.md`.

3. Train clean models (requires a config file — see note below):
   ```bash
   python scripts/04_train_deep.py --regime official
   ```

4. Run adversarial robustness evaluation and fine-tuning:
   ```bash
   python scripts/06_robustness.py
   ```

5. Run component ablation (seeds 43 and 44):
   ```bash
   python scripts/05_ablation.py --seeds 43 44
   ```

6. Aggregate H5 (ablation) results against the pre-computed baselines:
   ```bash
   python scripts/analyze_h5.py
   ```

> **Note:** Scripts 04–06 require a YAML configuration file specifying dataset paths,
> hyperparameters, and output directories. Configuration files are not included in
> this release (they contain local paths). The key hyperparameters are listed in the
> *Training settings* table above and in the paper (§3–§4).

## Saved-artifact analyses

The scripts `p2_s1` to `p2_s3`, `p2_s6_analysis.py` and `p2_a_suite_saved_artifacts.py` read stored results only (no training, no attack), except `p2_s6_pcrp_perwindow_export.py`, which loads the saved clean checkpoints, re-runs PGD on the evaluation windows and writes per-window PCRP scores; it reproduced the stored detector metrics exactly (maximum absolute difference 0.0 over 36 metrics per seed file; see `results/e5_pcrp_perwindow/*/seed*_check.json`). The scripts expect the working layout used for the study (`results_v3_paper2/` with the per-run `metrics.json` files), which is not bundled; their outputs are bundled in `results/saved_artifact_analyses/`.

## Verifying checksums

```bash
sha256sum -c SHASUMS
```

`SHASUMS` lists every file except itself.

## What is included

- Full source of the `trace_iomt` package: model definitions (TRACE-MoE, BiLSTM-DNN,
  Transformer, TF-SAGE-lite), training loop, PGD attack, metrics, and utilities
- Training scripts: `04_train_deep.py`, `05_ablation.py`, `06_robustness.py`
- H5 aggregation script: `analyze_h5.py`
- Paper figures (PDF/TIFF) and all five generator scripts in `figures/`
- Summary result CSVs in `results/`: H6 verdict and per-seed table, adversarial fine-tuning summary, abstention summary and rescore log, bootstrap CIs, and ablation training histories
- Summaries and per-window PCRP scores behind the exploratory studies, and the read-only analysis scripts that produce the saved-artifact analyses
- `TRACEABILITY.md`, which maps each paper table and figure to its source file

## What is NOT included

- Raw PCAP captures and processed flow feature files (dataset files)
- Model checkpoints and per-run `metrics.json` result files (`results/robustness/`, `results/ablation/`); several `TRACEABILITY.md` rows cite these and cannot be checked from this release alone
- Configuration files (contain local paths)
- Manuscript source and compiled PDF
- Preprocessing scripts (contain local paths to raw data)

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
