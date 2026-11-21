"""Paper 4 exploratory analyses on the exported per-window scores (reviewer items). NOT pre-registered (ESA3 onward):
no verdicts are issued and no Charter rule is applied. Reads results_v3_paper3/score_export/*.npz, writes
results_v3_paper3/saved_output_analyses_20261118/{per_run.csv, summary_by_model_class.csv, consistency_check.csv, SUMMARY.md}.

Quantities per run (unknown = held-out class windows, known = all other test windows):
  A1  binary head on known windows: false-positive rate on Benign, precision and recall for Attack.
  A2  safety-view decomposition of held-out windows: energy only / binary head only / both / neither (caught_any = union).
  A7  TPR at FPR 1% and 0.1%, AUROC and average precision for energy, MSP and MaxLogit as unknown-vs-known scores.
  A18 energy AUROC under temperature T in {0.5, 2, 5} (T=1 is the primary score).
Summary: mean over the five seeds with a percentile bootstrap 95% CI (10,000 resamples of the five seed values, rng seed 42).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results_v3_paper3" / "score_export"
OUT = ROOT / "results_v3_paper3" / "saved_output_analyses_20261118"
ZD = ROOT / "results_v3_paper3" / "zero_day"
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]


def tpr_at(fpr_target, fpr, tpr):
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def score_block(prefix, s, unk):
    """s: higher = more unknown."""
    y = unk.astype(np.int8)
    fpr, tpr, _ = roc_curve(y, s)
    return {f"{prefix}_auroc": float(roc_auc_score(y, s)), f"{prefix}_tpr_at_fpr1pct": tpr_at(0.01, fpr, tpr),
            f"{prefix}_tpr_at_fpr0p1pct": tpr_at(0.001, fpr, tpr), f"{prefix}_ap": float(average_precision_score(y, s))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cons = [], []
    for m in MODELS:
        for c in CLASSES:
            for sd in SEEDS:
                run = f"{m}_loao_T16_{c}_seed{sd}"
                f = EXP / f"{run}.npz"
                if not f.exists():
                    print("missing", run); continue
                z = np.load(f, allow_pickle=False)
                y19, hid, thr = z["y19"], int(z["hid"]), float(z["thr"])
                names = json.loads((ZD / run / "metrics.json").read_text(encoding="utf-8"))["hierarchy"]["classes"]["label19"]
                ben = names.index("Benign")
                unk = y19 == hid
                known = ~unk
                en, msp, ml, pa = z["energy"], z["msp"], z["maxlogit"], z["p_attack"]
                binp = pa > 0.5
                flagged = en < thr
                u = unk.sum()
                r = dict(model=m, held_out_class=c, seed=sd, n_unknown=int(u), n_known=int(known.sum()))
                # A1
                kb = known & (y19 == ben)
                ka = known & (y19 != ben)
                r["A1_binary_fpr_on_benign"] = float(binp[kb].mean()) if kb.any() else np.nan
                tp = int((binp & ka).sum()); fp = int((binp & kb).sum())
                r["A1_binary_precision_attack_known"] = tp / (tp + fp) if tp + fp else np.nan
                r["A1_binary_recall_attack_known"] = float(binp[ka].mean()) if ka.any() else np.nan
                # A2
                fu, bu = flagged[unk], binp[unk]
                r["A2_energy_only"] = float((fu & ~bu).mean()); r["A2_binary_only"] = float((~fu & bu).mean())
                r["A2_both"] = float((fu & bu).mean()); r["A2_neither"] = float((~fu & ~bu).mean())
                r["A2_caught_any"] = float((fu | bu).mean())
                r["A2_energy_flag_rate"] = float(fu.mean()); r["A2_binary_rate"] = float(bu.mean())
                # A7 scores
                r.update(score_block("A7_energy", -en, unk))
                r.update(score_block("A7_msp", -msp, unk))
                r.update(score_block("A7_maxlogit", -ml, unk))
                # A18 temperature sweep
                lg = z["logits19"].astype(np.float32)
                for T in (0.5, 2.0, 5.0):
                    eT = T * logsumexp(lg / T, axis=1)
                    r[f"A18_energy_T{T:g}_auroc"] = float(roc_auc_score(unk.astype(np.int8), -eT))
                rows.append(r)
                cj = json.loads(str(z["consistency"]))
                cons.append(dict(run=run, **cj, d_auroc=abs(cj["auroc_export"] - cj["auroc_metrics"]), d_caught=abs(cj["caught_export"] - cj["caught_metrics"])))
                print("done", run, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_run.csv", index=False)
    pd.DataFrame(cons).to_csv(OUT / "consistency_check.csv", index=False)
    rng = np.random.default_rng(42)
    metric_cols = [c for c in df.columns if c.split("_")[0] in ("A1", "A2", "A7", "A18")]
    out = []
    for (m, c), g in df.groupby(["model", "held_out_class"]):
        rec = dict(model=m, held_out_class=c, n_seeds=len(g))
        for col in metric_cols:
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            bs = rng.choice(v, size=(10000, len(v)), replace=True).mean(1)
            rec[col] = v.mean(); rec[col + "_lo"] = np.percentile(bs, 2.5); rec[col + "_hi"] = np.percentile(bs, 97.5)
        out.append(rec)
    sm = pd.DataFrame(out)
    sm.to_csv(OUT / "summary_by_model_class.csv", index=False)
    cdf = pd.DataFrame(cons)
    L = ["# Exploratory analyses on exported per-window scores (not pre-registered; no verdicts)\n",
         f"Runs analysed: {len(df)} of 75. Consistency with the runs' own metrics.json: max |AUROC difference| = {cdf.d_auroc.max():.2e}, "
         f"max |caught_any difference| = {cdf.d_caught.max():.2e}.\n",
         "Summary table: `summary_by_model_class.csv` (mean over five seeds with percentile bootstrap 95% CI per metric).\n"]
    for col in ("A2_energy_only", "A2_binary_only", "A2_both", "A2_neither"):
        L.append(f"\n## {col} (mean over seeds)\n")
        L.append(sm.pivot(index="held_out_class", columns="model", values=col).round(4).to_markdown())
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
