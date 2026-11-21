"""Paper 4 Figure Y — H5 twin-alone unknown-attack AUROC per held-out class.

Reads results_v3_paper3/h8_verdict/twin_h5_verdict.csv.

Since the twin discrepancy D_min is computed from the raw windowed feature
vector (not from model predictions), the per-class AUROC is identical across
the three architectures. The figure therefore shows one bar per held-out
class with bootstrap 95% CI error bars, and horizontal reference lines at
the SUPPORTED threshold (0.80) and the CI-lower-bound threshold (0.75) from
Charter v1.3.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VERDICT_DIR = ROOT / "results_v3_paper3" / "h8_verdict"
OUT_DIR = VERDICT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
AMBER = "#eb6834"
MUTED = "#9aa3ad"
INK = "#0b0b0b"


def read_h5():
    rows = []
    with (VERDICT_DIR / "twin_h5_verdict.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("model") or r.get("mean_auroc") in (None, "", "overall_verdict"):
                continue
            if r["model"] != "trace_moe":
                continue
            rows.append({
                "cls": r["held_out_class"],
                "mean": float(r["mean_auroc"]),
                "lo": float(r["ci_lo"]),
                "hi": float(r["ci_hi"]),
                "verdict": r["verdict"],
                "n": int(r["n_seeds"]),
            })
    return rows


def main():
    data = read_h5()
    labels = [r["cls"].replace("_", " ").replace("MQTT-", "MQTT ").replace("TCP IP-", "TCP/IP ")
              for r in data]
    means = [r["mean"] for r in data]
    err_lo = [r["mean"] - r["lo"] for r in data]
    err_hi = [r["hi"] - r["mean"] for r in data]
    colors = [BLUE if r["verdict"] == "SUPPORTED" else AMBER for r in data]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    x = range(len(data))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], color=colors,
                  edgecolor=INK, linewidth=0.6, capsize=4, width=0.6,
                  error_kw={"elinewidth": 1.1, "ecolor": INK})
    ax.axhline(0.80, color=MUTED, linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(0.75, color=MUTED, linestyle=":", linewidth=0.9, zorder=0)
    ax.text(len(data) - 0.5, 0.80, "0.80 mean threshold",
            ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.text(len(data) - 0.5, 0.75, "0.75 CI-lower threshold",
            ha="right", va="bottom", fontsize=8, color=MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Twin-alone AUROC (D_min = min nearest-cluster Mahalanobis)",
                  fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Figure Y: H5 pre-registered twin-alone unknown-attack AUROC\n"
                 "5 seeds (42-46), bootstrap 95% CI; model-invariant (D_min from raw features)",
                 fontsize=10)
    for b, r in zip(bars, data):
        ax.text(b.get_x() + b.get_width() / 2, r["mean"] + 0.015,
                f"{r['mean']:.3f}", ha="center", va="bottom", fontsize=8, color=INK)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_y_h5_twin_auroc.{ext}"
        fig.tight_layout()
        fig.savefig(out)
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
