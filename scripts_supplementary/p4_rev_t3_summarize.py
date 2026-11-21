"""Paper 4, T3 summary (post-hoc exploratory; no verdict, no change to H1-H5). CPU only, reads T3/per_run_feature_scores.csv.

Per (model, held-out class): mean over seeds and percentile bootstrap CI over seeds (10,000 resamples, rng 42) of AUROC and FPR@95
for energy, Mahalanobis (Ledoit-Wolf), kNN-50, ReAct and ViM; plus the paired seed bootstrap of each score minus energy.
Writes summary_feature_scores.csv, paired_vs_energy.csv and README.md next to the input. No 'best score' is declared.
Usage: python scripts/p4_rev_t3_summarize.py --date 20261121
"""
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORES = ["energy", "mahalanobis_lw", "knn50", "react", "vim"]
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
D = ROOT / "results_v3_paper3" / f"p4_review_analyses_{a.date}" / "T3"
df = pd.read_csv(D / "per_run_feature_scores.csv")
rng = np.random.default_rng(42)
B = 10_000


def ci(v):
    v = np.asarray(v, float)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


rows, prow = [], []
for (m, c), g in df.groupby(["model", "held_out_class"]):
    g = g.sort_values("seed")
    for s in SCORES:
        for met in ("auroc", "fpr95"):
            mean, lo, hi = ci(g[f"{s}_{met}"])
            rows.append(dict(model=m, held_out_class=c, score=s, metric=met, n_seeds=len(g), mean=mean, ci_lo=lo, ci_hi=hi))
        if s != "energy":
            for met in ("auroc", "fpr95"):
                mean, lo, hi = ci(g[f"{s}_{met}"].values - g[f"energy_{met}"].values)
                prow.append(dict(model=m, held_out_class=c, score=s, metric=met, n_seeds=len(g), mean_diff_vs_energy=mean, ci_lo=lo, ci_hi=hi))
pd.DataFrame(rows).to_csv(D / "summary_feature_scores.csv", index=False)
pd.DataFrame(prow).to_csv(D / "paired_vs_energy.csv", index=False)
bad = df[~df["consistency_ok"].astype(bool)]
(D / "README.md").write_text(
    "# T3: feature-space scores on learned embeddings (post-hoc exploratory)\n\n"
    f"Summary written {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/p4_rev_t3_summarize.py` (SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}).\n"
    f"Input: `per_run_feature_scores.csv` ({len(df)} runs; expected 75). Scoring script `scripts/p4_rev_t3_feature_scores.py`.\n"
    f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}; bootstrap rng 42, {B} resamples over seeds.\n\n"
    f"Consistency with score_export (energy within 1e-3, equal labels): {len(df) - len(bad)} of {len(df)} runs pass"
    + ("" if bad.empty else "; FAILING runs: " + ", ".join(bad["run"])) + ".\n\n"
    "Fixed settings: fit on known-class training windows (class-stratified subsample of at most 200,000, rng 42); Mahalanobis class-conditional with "
    "Ledoit-Wolf covariance, minimum over known classes; kNN on L2-normalised embeddings, k = 50; ReAct clip at the 90th percentile of training activations; "
    "ViM subspace dimension d/2. None was changed after seeing test results. All scores: higher = more unknown; held-out class positive.\n\n"
    "Status: exploratory, no verdict. Files: `summary_feature_scores.csv`, `paired_vs_energy.csv`.\n", encoding="utf-8")
print("rows", len(df), "consistency failures", len(bad))
