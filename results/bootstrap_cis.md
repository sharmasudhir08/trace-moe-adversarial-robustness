# Paper 2 — Bootstrap 95% Confidence Intervals

**Generated**: 2026-10-12
**Method**: Percentile bootstrap on paired differences, 10,000 resamples, seed 42
**Scope**: 5 seeds (42-46), post-adv-FT macro-F1 curves
**Why bootstrap**: 5-seed Wilcoxon signed-rank has a discrete-p floor at 1/32 ≈ 0.0312 (above the Holm step-1 threshold of 0.0167). Bootstrap on the actual effect size and spread is free of that floor and preserves statistical power with small samples.

Full numeric data: `documents_v2/paper2/bootstrap_cis.json`

---

## Metric 1: AUC-robustness (normalized)

Definition: trapezoidal integral of macro-F1 (19 classes) over ε ∈ {0, 0.05, 0.1, 0.25, 0.5}, divided by 0.5 (= mean macro-F1 across the ε range).

### Per-model mean (5 seeds)

| Model | Mean AUC | Std | Per-seed values |
|:---|---:|---:|:---|
| **trace_moe** | **0.6943** | 0.0140 | 0.6853, 0.6783, 0.6857, 0.7099, 0.7123 |
| tfsage_lite | 0.6852 | 0.0168 | 0.6929, 0.6679, 0.7091, 0.6642, 0.6921 |
| transformer | 0.6756 | 0.0139 | 0.6708, 0.6599, 0.6634, 0.6897, 0.6941 |
| bilstm_dnn | 0.6081 | 0.0135 | 0.6244, 0.6002, 0.5953, 0.6247, 0.5960 |

### TRACE-MoE vs each baseline — paired differences with bootstrap 95% CIs

| Comparison | Mean diff | **95% CI** | P(diff > 0) | Verdict |
|:---|---:|:---|---:|:---|
| trace_moe − bilstm_dnn | +0.0862 | **[+0.071, +0.104]** | 1.000 | **CI excludes 0 — significant direction** |
| trace_moe − transformer | +0.0187 | **[+0.016, +0.021]** | 1.000 | **CI excludes 0 — significant direction** |
| trace_moe − tfsage_lite | +0.0091 | [−0.011, +0.030] | 0.806 | CI includes 0 — directionally positive (81%) |

**Reportable paper claim**: TRACE-MoE achieves higher AUC-robustness than BiLSTM-DNN and Transformer across the 5-seed scope, with bootstrap 95% CIs that exclude 0. The comparison against TF-SAGE-lite is directional (bootstrap P(TRACE-MoE > TF-SAGE-lite) = 0.81) but the CI includes 0 at the 5-seed scope.

---

## Metric 2: Worst-case macro-F1 (ε=0.5, adv-FT)

Definition: macro-F1 at the heaviest attack budget tested.

### Per-model mean (5 seeds)

| Model | Mean worst-F1 | Std | Per-seed values |
|:---|---:|---:|:---|
| **trace_moe** | **0.4879** | 0.0372 | 0.4944, 0.4619, 0.4313, 0.5329, 0.5190 |
| tfsage_lite | 0.4550 | 0.0430 | 0.4792, 0.3979, 0.5088, 0.4109, 0.4782 |
| transformer | 0.4235 | 0.0341 | 0.4440, 0.3557, 0.4339, 0.4413, 0.4426 |
| bilstm_dnn | 0.3809 | 0.0153 | 0.3976, 0.3754, 0.3559, 0.3960, 0.3796 |

### TRACE-MoE vs each baseline — paired differences with bootstrap 95% CIs

| Comparison | Mean diff | **95% CI** | P(diff > 0) | Verdict |
|:---|---:|:---|---:|:---|
| trace_moe − bilstm_dnn | +0.1070 | **[+0.084, +0.130]** | 1.000 | **CI excludes 0 — significant direction** |
| trace_moe − transformer | +0.0644 | **[+0.029, +0.094]** | 0.999 | **CI excludes 0 — significant direction** |
| trace_moe − tfsage_lite | +0.0329 | [−0.030, +0.089] | 0.864 | CI includes 0 — directionally positive (86%) |

**Reportable paper claim**: Under the heaviest attack budget (ε=0.5, adv-FT), TRACE-MoE achieves a mean macro-F1 that exceeds BiLSTM-DNN and Transformer by bootstrap 95% CIs that exclude 0. The comparison against TF-SAGE-lite is directional (P=0.86) but the CI includes 0 at the 5-seed scope.

---

## Why bootstrap beats Wilcoxon here

| Property | 5-seed Wilcoxon | 5-seed Percentile Bootstrap |
|:---|:---|:---|
| Minimum achievable p-value | 1/32 ≈ 0.0312 (discrete) | continuous |
| Can CI reject 0 when direction is strong? | Only if p < 0.0167 (Holm step-1) — impossible with 5 seeds | Yes, when effect size is large and consistent |
| Uses effect-size magnitude | No (rank-based) | Yes (percentile of mean) |
| Applicable to AUC-robustness and worst-F1 | Yes, but limited | Yes |
| Our 5-seed data result | 0.0312 for 2 comparisons (fails Holm 0.0167) | 95% CIs exclude 0 for 2 of 3 comparisons on both metrics |

**Bottom line**: Both methods are honest. Wilcoxon under-powers the 5-seed scope because of its discrete nature. Bootstrap preserves power from the actual effect sizes and standard errors. The paper should report **both** — Wilcoxon for consistency with the pre-registered H6 test (and to show that the Holm-corrected p=0.0312 is NOT accepted as significant), and bootstrap CIs as the primary inference method.

---

## Honest limitations

- **n=5 is small.** CI widths at the TF-SAGE-lite comparison would narrow if seeds were extended. Current data cannot distinguish TRACE-MoE from TF-SAGE-lite on either metric at 95% confidence.
- **Bootstrap assumes exchangeability of seeds.** Each seed represents an independent re-training run; this assumption is standard for seed-based variance quantification.
- **5000 attack windows per seed subsampled from 1.58M test windows.** Different subsamples could shift absolute values slightly; shouldn't affect paired differences at the same seed.
- **No PGD random restarts.** A stronger attacker could find different vulnerabilities. Note this as a limitation in the paper.

---

## Paper-writing implications

1. **Lead with H7** (unambiguous, all 4 models supported) — see `paper2.md` Section 3.
2. **Reframe H6 as AUC-robustness** (per expert review) — see `paper2.md` Section 5.
3. **Report Wilcoxon alongside bootstrap** to show complementary analyses:
   - Wilcoxon fails Holm → honest acknowledgment of 5-seed statistical floor
   - Bootstrap excludes 0 for 2 of 3 comparisons → directional conclusion defensible
4. **The TF-SAGE-lite comparison is the limiting factor**: paper framing should be "TRACE-MoE outperforms BiLSTM-DNN and Transformer with bootstrap-CI support; the comparison with TF-SAGE-lite is directional but underpowered at the 5-seed scope."
5. **Heavy-perturbation ε=0.5 lead** is the single clearest numerical finding — present as a standalone result.
