# Adversarial Robustness of Mixture-of-Experts IoMT Intrusion Detection with Protocol-Cluster Reference-Profile Analysis

Code for the paper submitted to *Cluster Computing* (Springer Nature, 2026).

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
├── scripts
│   ├── 04_train_deep.py
│   ├── 05_ablation.py
│   ├── 06_robustness.py
│   ├── _common.py
│   └── analyze_h5.py
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
├── TRACEABILITY.md
├── data_access.md
└── requirements.txt
```

## Training settings

| Parameter | Value |
|---|---|
| Seeds | 42, 43, 44, 45, 46 |
| Window length T | 16 consecutive flows |
| Training windows per class | 60,000 (balanced sampling) |
| PGD steps | 20 |
| PGD step size α | 0.1ε |
| PGD norm | ℓ∞ |
| Mutable features | 18 |
| Adversarial fine-tuning epochs | 3 |
| Adversarial fine-tuning budget εtrain | 0.25 |
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

## Verifying checksums

```bash
sha256sum -c SHASUMS
```

## What is included

- Full source of the `trace_iomt` package: model definitions (TRACE-MoE, BiLSTM-DNN,
  Transformer, TF-SAGE-lite), training loop, PGD attack, metrics, and utilities
- Training scripts: `04_train_deep.py`, `05_ablation.py`, `06_robustness.py`
- H5 aggregation script: `analyze_h5.py`

## What is NOT included

- Raw PCAP captures and processed flow feature files (dataset files)
- Model checkpoints, training history, and per-run result files
- Configuration files (contain local paths)
- Figure-generation code and manuscript LaTeX source
- Preprocessing scripts (contain local paths to raw data)

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
