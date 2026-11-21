"""
Paper 4 figure generation -- v1.4 (all 75 runs under the 50-epoch protocol). Copy of paper4_figures_5seed.py; only the input/output folders and the fig1 y-limit changed.

Reads from documents_v2/paper4/repo_staging/results/ (5-seed CSVs).
Writes PNG + PDF to documents_v2/paper4/repo_staging/figures/.

Produces:
  fig1_h1_safety_view.{png,pdf}
  fig2_h3_architecture_comparison.{png,pdf}
  fig3_h2_energy_vs_msp.{png,pdf}
  fig4_h4_closed_set_degradation.{png,pdf}
  fig_y_h5_twin_auroc.{png,pdf}

Changes from paper4_figures.py:
  - Reads 5-seed staging CSVs, not 3-seed h8_verdict CSVs.
  - Y-axis labels say "5 seeds"; fig2 min-max spans all 5 seeds.
  - No fig.text() footnote annotations inside the images.
  - Fig4 BiLSTM-DNN verdict corrected (ARP SUPPORTED at 5-seed).
  - Fig3 TRACE-MoE MQTT-DDoS correctly shown as inconclusive (CI straddles zero).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "results_v3_paper3"
RESULTS = STAGING / "h8_verdict"
OUT_DIR = STAGING / "h8_verdict" / "figures_v1_4"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": BASELINE_AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

CLASS_ORDER = [
    "ARP_Spoofing",
    "Recon-VulScan",
    "MQTT-Malformed_Data",
    "TCP_IP-DoS-UDP",
    "MQTT-DDoS-Publish_Flood",
]
CLASS_LABELS = {
    "ARP_Spoofing": "ARP\nSpoofing",
    "Recon-VulScan": "Recon\nVulScan",
    "MQTT-Malformed_Data": "MQTT\nMalformed",
    "TCP_IP-DoS-UDP": "TCP/IP\nDoS-UDP",
    "MQTT-DDoS-Publish_Flood": "MQTT-DDoS\nPublish Flood",
}

SEEDS = (42, 43, 44, 45, 46)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig1_h1_safety_view() -> Path:
    rows = {r["class"]: r for r in read_csv(RESULTS / "H1_safety_view.csv")}
    classes = CLASS_ORDER
    means = [float(rows[c]["mean"]) for c in classes]
    ci_lo = [float(rows[c]["ci_lo"]) for c in classes]
    ci_hi = [float(rows[c]["ci_hi"]) for c in classes]
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    x = range(len(classes))
    ax.bar(x, means, width=0.55, color=BLUE, edgecolor="none", zorder=3, linewidth=0)
    ax.errorbar(
        x, means, yerr=[err_lo, err_hi], fmt="none", ecolor=INK,
        elinewidth=1.4, capsize=4, capthick=1.4, zorder=4,
    )
    ax.axhline(0.95, color=ORANGE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(
        len(classes) - 0.42, 0.952, "95% target",
        color=ORANGE, fontsize=9, va="bottom", ha="right",
    )
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.006, f"{m:.4f}", ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
    ax.set_ylim(max(0.0, min(ci_lo) - 0.04), 1.015)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.set_ylabel("Caught-any detection rate\n(TRACE-MoE, mean of 5 seeds)", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(
        "H1: TRACE-MoE safety-view detection rate by held-out attack class",
        fontsize=11, pad=12, color=INK,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(length=0)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig1_h1_safety_view.png"
    out_pdf = OUT_DIR / "fig1_h1_safety_view.pdf"
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def fig2_h3_architecture_comparison() -> Path:
    rows = read_csv(RESULTS / "H3_trace_moe_vs_baselines.csv")
    by_class_transformer = {r["class"]: r for r in rows if r["baseline"] == "transformer"}
    by_class_bilstm = {r["class"]: r for r in rows if r["baseline"] == "bilstm_dnn"}

    classes = CLASS_ORDER

    def seed_vals(row: dict, prefix: str) -> list[float]:
        return [float(row[f"{prefix}_s{s}"]) for s in SEEDS]

    series: dict[str, list[float]] = {"TRACE-MoE": [], "Transformer": [], "BiLSTM-DNN": []}
    errs: dict[str, list[tuple[float, float]]] = {"TRACE-MoE": [], "Transformer": [], "BiLSTM-DNN": []}
    for c in classes:
        tr = by_class_transformer[c]
        bl = by_class_bilstm[c]
        tm_vals = seed_vals(tr, "tm")
        tf_vals = seed_vals(tr, "bl")
        bi_vals = seed_vals(bl, "bl")
        for name, vals in (("TRACE-MoE", tm_vals), ("Transformer", tf_vals), ("BiLSTM-DNN", bi_vals)):
            m = sum(vals) / len(vals)
            series[name].append(m)
            errs[name].append((m - min(vals), max(vals) - m))

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=300)
    width = 0.26
    x = list(range(len(classes)))
    colors = {"TRACE-MoE": BLUE, "Transformer": ORANGE, "BiLSTM-DNN": AQUA}
    offsets = {"TRACE-MoE": -width, "Transformer": 0, "BiLSTM-DNN": width}

    for name in ("TRACE-MoE", "Transformer", "BiLSTM-DNN"):
        xs = [xi + offsets[name] for xi in x]
        means = series[name]
        err_lo = [e[0] for e in errs[name]]
        err_hi = [e[1] for e in errs[name]]
        ax.bar(xs, means, width=width * 0.92, color=colors[name], label=name, zorder=3, linewidth=0)
        ax.errorbar(xs, means, yerr=[err_lo, err_hi], fmt="none", ecolor=INK,
                    elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Energy-score AUROC\n(mean of 5 seeds)", fontsize=9.5)
    ax.set_title(
        "H3: open-set energy AUROC by model and held-out attack class",
        fontsize=11, pad=12, color=INK,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig2_h3_architecture_comparison.png"
    out_pdf = OUT_DIR / "fig2_h3_architecture_comparison.pdf"
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def fig3_h2_energy_vs_msp() -> Path:
    rows = read_csv(RESULTS / "H2_energy_vs_msp.csv")
    by_model = {m: {r["class"]: r for r in rows if r["model"] == m}
                for m in ("trace_moe", "transformer", "bilstm_dnn")}
    classes = CLASS_ORDER
    labels = {"trace_moe": "TRACE-MoE", "transformer": "Transformer", "bilstm_dnn": "BiLSTM-DNN"}
    colors = {"trace_moe": BLUE, "transformer": ORANGE, "bilstm_dnn": AQUA}
    offsets = {"trace_moe": -0.26, "transformer": 0.0, "bilstm_dnn": 0.26}

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=300)
    x = list(range(len(classes)))
    for m in ("trace_moe", "transformer", "bilstm_dnn"):
        means = [float(by_model[m][c]["mean_diff"]) for c in classes]
        ci_lo = [float(by_model[m][c]["ci_lo"]) for c in classes]
        ci_hi = [float(by_model[m][c]["ci_hi"]) for c in classes]
        err_lo = [mn - lo for mn, lo in zip(means, ci_lo)]
        err_hi = [hi - mn for mn, hi in zip(means, ci_hi)]
        xs = [xi + offsets[m] for xi in x]
        ax.bar(xs, means, width=0.24, color=colors[m], label=labels[m], zorder=3, linewidth=0)
        ax.errorbar(xs, means, yerr=[err_lo, err_hi], fmt="none", ecolor=INK,
                    elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)

    ax.axhline(0.0, color=BASELINE_AXIS, linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
    ax.set_ylabel("Energy AUROC minus MSP AUROC\n(mean, bootstrap 95% CI)", fontsize=9.5)
    ax.set_title("H2: energy-score advantage over MSP by model and held-out class",
                 fontsize=11, pad=12, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig3_h2_energy_vs_msp.png"
    out_pdf = OUT_DIR / "fig3_h2_energy_vs_msp.pdf"
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def fig4_h4_closed_set_degradation() -> Path:
    rows = read_csv(RESULTS / "H4_closed_set_degradation.csv")
    by_model = {m: {r["class"]: r for r in rows if r["model"] == m}
                for m in ("trace_moe", "transformer", "bilstm_dnn")}
    classes = CLASS_ORDER
    labels = {"trace_moe": "TRACE-MoE", "transformer": "Transformer", "bilstm_dnn": "BiLSTM-DNN"}
    colors = {"trace_moe": BLUE, "transformer": ORANGE, "bilstm_dnn": AQUA}
    offsets = {"trace_moe": -0.26, "transformer": 0.0, "bilstm_dnn": 0.26}

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=300)
    x = list(range(len(classes)))
    for m in ("trace_moe", "transformer", "bilstm_dnn"):
        means = [float(by_model[m][c]["mean_degr"]) for c in classes]
        ci_lo = [float(by_model[m][c]["ci_lo"]) for c in classes]
        ci_hi = [float(by_model[m][c]["ci_hi"]) for c in classes]
        err_lo = [mn - lo for mn, lo in zip(means, ci_lo)]
        err_hi = [hi - mn for mn, hi in zip(means, ci_hi)]
        xs = [xi + offsets[m] for xi in x]
        ax.bar(xs, means, width=0.24, color=colors[m], label=labels[m], zorder=3, linewidth=0)
        ax.errorbar(xs, means, yerr=[err_lo, err_hi], fmt="none", ecolor=INK,
                    elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)

    ax.axhline(0.10, color=ORANGE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(len(classes) - 0.42, 0.102, "10 pt bound",
            color=ORANGE, fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
    ax.set_ylabel("Closed-set macro-F1 degradation\n(mean, bootstrap 95% CI)", fontsize=9.5)
    ax.set_title("H4: closed-set macro-F1 degradation by model and held-out class",
                 fontsize=11, pad=12, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig4_h4_closed_set_degradation.png"
    out_pdf = OUT_DIR / "fig4_h4_closed_set_degradation.pdf"
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def fig_y_h5_twin_auroc() -> Path:
    rows = read_csv(RESULTS / "twin_h5_verdict.csv")
    # Use trace_moe rows only (values are model-invariant)
    data = []
    for r in rows:
        if r.get("model") != "trace_moe":
            continue
        if not r.get("mean_auroc") or r.get("mean_auroc") in ("", "overall_verdict"):
            continue
        data.append({
            "cls": r["held_out_class"],
            "mean": float(r["mean_auroc"]),
            "lo": float(r["ci_lo"]),
            "hi": float(r["ci_hi"]),
            "verdict": r["verdict"],
        })

    cls_list = [r["cls"] for r in data]
    labels = [CLASS_LABELS.get(r["cls"], r["cls"]) for r in data]
    means = [r["mean"] for r in data]
    err_lo = [r["mean"] - r["lo"] for r in data]
    err_hi = [r["hi"] - r["mean"] for r in data]
    colors = [BLUE if "SUPPORTED" in r["verdict"] and "INCONCLUSIVE" not in r["verdict"]
              else ORANGE for r in data]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    x = range(len(data))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], color=colors,
                  edgecolor="none", linewidth=0, capsize=4, width=0.55,
                  error_kw={"elinewidth": 1.4, "ecolor": INK, "capthick": 1.4})

    ax.axhline(0.80, color=ORANGE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(len(data) - 0.42, 0.802, "0.80 mean threshold",
            ha="right", va="bottom", fontsize=9, color=ORANGE)

    for b, r in zip(bars, data):
        cx = b.get_x() + b.get_width() / 2
        if "Recon" in r["cls"]:
            ax.text(cx, r["mean"] - 0.025, f"{r['mean']:.4f}",
                    ha="center", va="top", fontsize=9, color=INK)
        else:
            ax.text(cx, r["mean"] + 0.015, f"{r['mean']:.4f}",
                    ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_ylabel("Protocol-cluster reference profile AUROC\n(TRACE-MoE, mean of 5 seeds)", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(
        "H5: protocol-cluster reference profile (PCRP) unknown-attack AUROC by held-out class",
        fontsize=11, pad=12, color=INK,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(length=0)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig_y_h5_twin_auroc.png"
    out_pdf = OUT_DIR / "fig_y_h5_twin_auroc.pdf"
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn, name in [
        (fig1_h1_safety_view, "Fig 1 H1"),
        (fig2_h3_architecture_comparison, "Fig 2 H3"),
        (fig3_h2_energy_vs_msp, "Fig 3 H2"),
        (fig4_h4_closed_set_degradation, "Fig 4 H4"),
        (fig_y_h5_twin_auroc, "Fig Y H5"),
    ]:
        p = fn()
        print(f"{name}: {p}")


if __name__ == "__main__":
    main()
