# Paper 2 H6 verdict -- twin-discrepancy monotonicity under PGD

**Pre-registered**: Paper 2 Charter v1.6 (SHA256 `6dc0bd10f8f4d9c5467ee96e4932f3569570e4c0d9b08635c6ca807e132a36d5`).

**Score**: Spearman rank correlation rho(epsilon, mean_D_attack) across 5 epsilon values
in [0.0, 0.05, 0.1, 0.25, 0.5], computed per (model, seed) and averaged across 5 seeds per model with
a bootstrap 95% CI (10,000 resamples, RNG seed 42).

**Scope**: 20/20 (model, seed) pairs scored.

## Per-architecture verdicts

| Model | n seeds | Mean rho | 95% CI | Verdict |
|---|---|---|---|---|
| trace_moe | 5 | 1.0000 | [1.0000, 1.0000] | SUPPORTED |
| transformer | 5 | 1.0000 | [1.0000, 1.0000] | SUPPORTED |
| bilstm_dnn | 5 | 1.0000 | [1.0000, 1.0000] | SUPPORTED |
| tfsage_lite | 5 | 1.0000 | [1.0000, 1.0000] | SUPPORTED |

**Overall verdict**: **SUPPORTED (strong, overall)**

## Decision rule (pre-registered per Charter v1.6)

- **SUPPORTED**: rho_arch >= 0.70 AND CI lower >= 0.50.
- **SUPPORTED (preliminary)**: rho_arch >= 0.50 AND CI lower > 0 AND not SUPPORTED.
- **INCONCLUSIVE**: CI straddles 0.
- **NOT SUPPORTED**: rho_arch < 0 OR CI upper < 0.30.
- **Overall SUPPORTED (strong)**: SUPPORTED on all 4 architectures.
- **Overall SUPPORTED (preliminary)**: SUPPORTED on 3 of 4 OR SUPPORTED+PRELIM on all 4.
- **Overall INCONCLUSIVE**: SUPPORTED on 1 or 2.
- **Overall NOT SUPPORTED**: SUPPORTED on 0.