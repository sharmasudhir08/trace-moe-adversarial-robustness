"""Pooled exploratory table for the manuscript (CPU, reads T3 per-run scores and part-2 CSVs; writes one new CSV). Mean over the five held-out classes per seed, then mean and 95% bootstrap interval over the five seeds (10,000, rng 42)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1] / "results_v3_paper3" / "p4_review_analyses_20261121"
rng = np.random.default_rng(42)
def ci(v):
    v = np.asarray(v, float); bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
t3 = pd.read_csv(R / "T3" / "per_run_feature_scores.csv"); u10 = pd.read_csv(R / "part2" / "U10" / "logit_scores_per_run.csv"); u2 = pd.read_csv(R / "part2" / "U2" / "head_disagreement_per_run.csv")
rows = []
for m in ["trace_moe", "transformer", "bilstm_dnn"]:
    def per_seed(df, col): return df[df.model == m].groupby("seed")[col].mean().sort_index().to_numpy()
    for name, df, col in [("energy (reference)", t3, "energy_auroc"), ("Mahalanobis (Ledoit-Wolf), embeddings", t3, "mahalanobis_lw_auroc"), ("kNN-50, embeddings", t3, "knn50_auroc"), ("ReAct (90th pct), embeddings", t3, "react_auroc"),
                          ("ViM, embeddings", t3, "vim_auroc"), ("generalized entropy (gamma 0.1), logits", u10, "gen_gamma0.1_auroc"), ("logit norm", u10, "logit_norm_auroc"), ("max logit minus mean logit", u10, "maxlogit_minus_mean_auroc"),
                          ("head disagreement, KL score A", u2, "scoreA_auroc"), ("head disagreement, indicator B", u2, "scoreB_auroc")]:
        v = per_seed(df, col); a, lo, hi = ci(v)
        base = per_seed(t3, "energy_auroc"); d, dlo, dhi = ci(v - base) if "reference" not in name else (0, 0, 0)
        rows.append(dict(model=m, score=name, auroc=a, auroc_lo=lo, auroc_hi=hi, diff_vs_energy=d, diff_lo=dlo, diff_hi=dhi))
out = R / "part2" / "pooled_table_for_manuscript.csv"
assert not out.exists(); pd.DataFrame(rows).round(4).to_csv(out, index=False); print(pd.DataFrame(rows).round(3).to_string())
