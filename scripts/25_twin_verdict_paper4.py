"""Step 25 — Aggregate Paper 4 H5 twin-alone AUROC verdict across seeds.

Reads results_v3_paper3/zero_day/<model>_loao_T16_<class>_seed<s>/twin_h5_score.json
for each (model x held_out_class x seed). For each (model, held_out_class) pair,
bootstraps the AUROC across seeds (3 or 5) and applies the pre-registered four-tier
rule locked by Paper 4 Charter v1.2.

Verdict rule per (model, held_out_class) pair (pre-registered BEFORE aggregation):
  SUPPORTED:              mean AUROC >= 0.80 AND CI lower bound >= 0.75
  SUPPORTED (preliminary): mean AUROC >= 0.80 AND CI lower bound >= 0.60 (relaxed)
  INCONCLUSIVE:            CI straddles 0.80 or AUROC between 0.50 and 0.80
  NOT SUPPORTED:           CI upper bound < 0.65 OR mean AUROC < 0.65

Overall H5 verdict per model (pre-registered):
  SUPPORTED:              SUPPORTED on at least 4 of 5 held-out classes
  SUPPORTED (preliminary): SUPPORTED on at least 3 of 5
  INCONCLUSIVE:           2 of 5
  NOT SUPPORTED:           <= 1 of 5

Outputs:
  results_v3_paper3/h8_verdict/twin_h5_verdict.csv
  results_v3_paper3/h8_verdict/twin_h5_verdict.md
  results_v3_paper3/h8_verdict/twin_h5_verdict.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_iomt.config import load_config  # noqa: E402

MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = [
    "ARP_Spoofing",
    "Recon-VulScan",
    "MQTT-Malformed_Data",
    "TCP_IP-DoS-UDP",
    "MQTT-DDoS-Publish_Flood",
]
BOOT_N = 10_000
RNG_SEED = 42


def bootstrap_ci(values, n=BOOT_N, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(valid), size=(n, len(valid)))
    boots = valid[idx].mean(axis=1)
    return float(valid.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def verdict_h5_pair(mean, lo, hi):
    if np.isnan(mean):
        return "NO DATA"
    if hi < 0.65 or mean < 0.65:
        return "NOT SUPPORTED"
    if mean >= 0.80 and lo >= 0.75:
        return "SUPPORTED"
    if mean >= 0.80 and lo >= 0.60:
        return "SUPPORTED (preliminary)"
    return "INCONCLUSIVE"


def verdict_h5_overall(per_class_verdicts):
    sup = sum(1 for v in per_class_verdicts if v == "SUPPORTED")
    pre = sum(1 for v in per_class_verdicts if v == "SUPPORTED (preliminary)")
    inc = sum(1 for v in per_class_verdicts if v == "INCONCLUSIVE")
    nsp = sum(1 for v in per_class_verdicts if v == "NOT SUPPORTED")
    strong = sup
    if strong >= 4:
        return "SUPPORTED (strong, overall)"
    if sup + pre >= 3:
        return "SUPPORTED (preliminary, overall)"
    if strong + pre + inc >= 2 and sup + pre < 3:
        return "INCONCLUSIVE (overall)"
    return "NOT SUPPORTED (overall)"


def main():
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    res_dir = Path(cfg["paths"]["results_dir"])
    zd_dir = res_dir / "zero_day"

    per_pair_rows = []
    for model in MODELS:
        for held in CLASSES:
            safe = held.replace("/", "_")
            aurocs = []
            fprs = []
            seeds_used = []
            for seed_dir in [zd_dir / f"{model}_loao_T16_{safe}_seed{sd}" for sd in (42, 43, 44, 45, 46)]:  # exact names: a glob also matched *_1epoch_backup, __v2 and __v3 folders
                sj = seed_dir / "twin_h5_score.json"
                if not sj.exists():
                    continue
                d = json.loads(sj.read_text())
                aurocs.append(d["auroc"])
                fprs.append(d["fpr_at_95tpr"])
                seeds_used.append(d["seed"])
            if not aurocs:
                per_pair_rows.append({"model": model, "held_out_class": held, "n_seeds": 0,
                                       "mean_auroc": float("nan"), "ci_lo": float("nan"),
                                       "ci_hi": float("nan"), "mean_fpr_at_95tpr": float("nan"),
                                       "verdict": "NO DATA", "seeds_used": []})
                continue
            mean_auroc, lo, hi = bootstrap_ci(np.array(aurocs))
            mean_fpr = float(np.nanmean(fprs))
            v = verdict_h5_pair(mean_auroc, lo, hi)
            per_pair_rows.append({
                "model": model, "held_out_class": held, "n_seeds": len(aurocs),
                "mean_auroc": mean_auroc, "ci_lo": lo, "ci_hi": hi,
                "mean_fpr_at_95tpr": mean_fpr,
                "per_seed_auroc": [float(x) for x in aurocs],
                "seeds_used": seeds_used,
                "verdict": v,
            })

    # Overall per-model
    per_model_overall = {}
    for model in MODELS:
        pair_verdicts = [r["verdict"] for r in per_pair_rows if r["model"] == model and r["verdict"] != "NO DATA"]
        overall = verdict_h5_overall(pair_verdicts) if pair_verdicts else "NO DATA"
        per_model_overall[model] = overall

    # Save CSV
    out_dir = res_dir / "h8_verdict"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_lines = ["model,held_out_class,n_seeds,mean_auroc,ci_lo,ci_hi,mean_fpr_at_95tpr,verdict"]
    for r in per_pair_rows:
        csv_lines.append(f"{r['model']},{r['held_out_class']},{r['n_seeds']},"
                          f"{r['mean_auroc']:.6f},{r['ci_lo']:.6f},{r['ci_hi']:.6f},"
                          f"{r['mean_fpr_at_95tpr']:.6f},{r['verdict']}")
    csv_lines.append("")
    csv_lines.append("# overall per model")
    csv_lines.append("model,overall_verdict")
    for model, v in per_model_overall.items():
        csv_lines.append(f"{model},{v}")
    (out_dir / "twin_h5_verdict.csv").write_text("\n".join(csv_lines))

    # Save MD
    md = [
        "# Paper 4 H5 verdict — twin-alone unknown-attack AUROC",
        "",
        "**Pre-registered**: Paper 4 Charter v1.2 (DRAFT), H5 twin.",
        "**Twin design (A2)**: 5 LOAO-specific protocol-cluster profiles; held-out class never in train.",
        "**Score**: nearest-known-cluster Mahalanobis `D_min(x) = min_c D(x | c)`.",
        "**Verdict rule (per pair)**: SUPPORTED if mean AUROC >= 0.80 AND CI lower >= 0.75.",
        "",
        "## Per-pair verdicts",
        "",
        "| Model | Held-out class | n seeds | Mean AUROC | CI_lo | CI_hi | Mean FPR@95 | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in per_pair_rows:
        md.append(f"| {r['model']} | {r['held_out_class']} | {r['n_seeds']} | "
                  f"{r['mean_auroc']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} | "
                  f"{r['mean_fpr_at_95tpr']:.4f} | {r['verdict']} |")
    md += [
        "",
        "## Overall per-model",
        "",
    ]
    for model, v in per_model_overall.items():
        md.append(f"- **{model}**: {v}")
    (out_dir / "twin_h5_verdict.md").write_text("\n".join(md))

    # Save JSON
    out = {
        "charter_version": "Paper 4 Charter v1.2 (DRAFT)",
        "per_pair": per_pair_rows,
        "per_model_overall": per_model_overall,
        "bootstrap_n": BOOT_N,
        "rng_seed": RNG_SEED,
    }
    (out_dir / "twin_h5_verdict.json").write_text(json.dumps(out, indent=2))

    # Print summary
    print("\n=== Paper 4 H5 verdict summary ===")
    for r in per_pair_rows:
        print(f"  {r['model']:14s} {r['held_out_class']:28s} n={r['n_seeds']} "
              f"AUROC={r['mean_auroc']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] -> {r['verdict']}")
    print("\n=== Overall per model ===")
    for model, v in per_model_overall.items():
        print(f"  {model}: {v}")


if __name__ == "__main__":
    main()
