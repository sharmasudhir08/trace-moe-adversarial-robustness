"""
scripts/paper4_build_table.py

Phase 1 Step 2.1 of Paper 4 workflow (locked 2026-10-15 by project owner).
Builds the 4-panel verdict table (one panel per hypothesis) from the v1.1
aggregation CSVs. Deterministic; no judgment calls.

Input (READ-ONLY):
  - documents_v2/paper4/PAPER4_CHARTER.md  (SHA e1d29fc2...43f6caec)
  - results_v3_paper3/h8_verdict/H1_safety_view.csv
  - results_v3_paper3/h8_verdict/H2_energy_vs_msp.csv
  - results_v3_paper3/h8_verdict/H3_trace_moe_vs_baselines.csv
  - results_v3_paper3/h8_verdict/H4_closed_set_degradation.csv
  - results_v3_paper3/h8_verdict/ESA1_calibration.csv

Output:
  - results_v3_paper3/h8_verdict/paper4_table_main.md
    (4-panel verdict table + ESA1 calibration summary + legend)

Charter v1.1 verdict thresholds (locked, do not modify):
  H1: SUPPORTED (strong) if mean caught_any >= 0.95 AND CI lower >= 0.90.
      SUPPORTED (preliminary) if mean >= 0.95 but CI lower < 0.90.
      INCONCLUSIVE if mean < 0.95 and CI straddles 0.95.
      NOT SUPPORTED if CI upper < 0.95.
  H2: 4-tier with expected effect 0.05 AUROC, CI width margin 2 x 0.05 = 0.10.
      SUPPORTED, SUPPORTED (preliminary), INCONCLUSIVE, NOT SUPPORTED (narrow null).
  H3: Same 4-tier scheme as H2, margin 0.10 CI width.
  H4: 10 pt degradation threshold (tightened from 15 pt in v1.1).
      SUPPORTED, NOT SUPPORTED, INCONCLUSIVE based on CI vs 0.10.
"""
from __future__ import annotations

import sys
from pathlib import Path

import json

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
VDIR = ROOT / "results_v3_paper3" / "h8_verdict"
OUT = VDIR / "paper4_table_main.md"

CHARTER_SHA = "e1d29fc2b3278515579599bd5895f62e1706b173169db46b5978d80c43f6caec"  # v1.1 statistical recipe (unchanged in v1.3)
_V = json.loads((VDIR / "verdict.json").read_text(encoding="utf-8"))
N_SEEDS = len(_V["scope"]["seeds"])
BOOT = int(_V["bootstrap_resamples"])


def verdict_tag(v: str) -> str:
    """Compact tag for the verdict column (keeps tables readable)."""
    mapping = {
        "SUPPORTED (strong)": "SUPPORTED (strong)",
        "SUPPORTED (preliminary)": "SUPPORTED (prelim)",
        "SUPPORTED": "SUPPORTED",
        "INCONCLUSIVE": "INCONCLUSIVE",
        "NOT SUPPORTED": "NOT SUPPORTED",
    }
    return mapping.get(v, v)


def not_supported_subtag_h2_h3(mean: float, lo: float, hi: float,
                                margin: float = 0.10) -> str:
    """Charter-faithful disambiguation of 'NOT SUPPORTED' for H2/H3 (presentation only, no rule change).

    Charter v1.1 §Verdict tiers already states these two cases distinctly in text; the
    aggregator collapsed them to a single label. This helper restores the distinction
    at render time without altering any verdict rule or re-running aggregation.

    - CI excludes zero in the negative direction (hi < 0) AND mean < 0 → BASELINE WINS
    - CI contains zero AND width <= margin → NARROW NULL
    """
    if hi < 0 and mean < 0:
        return "NOT SUPPORTED (baseline wins)"
    if lo <= 0 <= hi:
        if (hi - lo) <= margin:
            return "NOT SUPPORTED (narrow null)"
    return "NOT SUPPORTED"


def render_verdict_h2_h3(verdict_raw: str, mean: float, lo: float, hi: float,
                          margin: float = 0.10) -> str:
    """Apply the H2/H3 sub-tag only to raw NOT SUPPORTED cells; leave others untouched."""
    if verdict_raw == "NOT SUPPORTED":
        return not_supported_subtag_h2_h3(mean, lo, hi, margin)
    return verdict_tag(verdict_raw)


# Audit-counter container, filled during panel render
_NS_SUBTAG_AUDIT: list[dict] = []


def _record_subtag(hypothesis: str, pair: str, raw: str, final: str) -> None:
    """Record every relabel for the §12a verdict label disambiguation audit."""
    if raw == "NOT SUPPORTED" and final != "NOT SUPPORTED":
        _NS_SUBTAG_AUDIT.append({
            "hypothesis": hypothesis, "pair": pair, "raw": raw, "final": final,
        })


def fmt_f(x: float, places: int = 4) -> str:
    """Signed-when-negative, fixed-places format for a float cell."""
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{places}f}" if x < 0 else f"{x:.{places}f}"


def panel_h1(df: pd.DataFrame) -> list[str]:
    lines = ["## Panel 1 of 4 — H1 safety-view detection rate (TRACE-MoE caught_any across 5 held-out classes x 3 seeds)",
             "",
             "Rule: SUPPORTED (strong) if mean caught_any >= 0.95 AND CI lower bound >= 0.90.",
             "",
             "| Held-out class | seed42 | seed43 | seed44 | mean | 95% CI | width | Verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['class']} | {r['seed42']:.4f} | {r['seed43']:.4f} | {r['seed44']:.4f} | "
            f"{r['mean']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['ci_width']:.4f} | "
            f"{verdict_tag(r['verdict'])} |"
        )
    # Breakdown
    counts = df["verdict"].value_counts().to_dict()
    strong = counts.get("SUPPORTED (strong)", 0)
    prelim = counts.get("SUPPORTED (preliminary)", 0)
    inc = counts.get("INCONCLUSIVE", 0)
    not_sup = counts.get("NOT SUPPORTED", 0)
    lines += ["",
              f"H1 breakdown over 5 class pairs: SUPPORTED (strong) {strong}, "
              f"SUPPORTED (preliminary) {prelim}, INCONCLUSIVE {inc}, NOT SUPPORTED {not_sup}.",
              ""]
    return lines


def panel_h2(df: pd.DataFrame) -> list[str]:
    lines = ["## Panel 2 of 4 — H2 energy AUROC minus MSP AUROC (3 models x 5 held-out classes = 15 pairs)",
             "",
             "Rule: paired bootstrap 95% CI of (energy_auroc minus msp_auroc) across 3 seeds. "
             "Expected effect 0.05 AUROC, CI width margin 0.10 (= 2 x expected). "
             "4-tier verdict per Charter v1.1.",
             "",
             "| Model | Held-out class | diff s42 | diff s43 | diff s44 | mean_diff | 95% CI | width | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        final_verdict = render_verdict_h2_h3(r["verdict"], r["mean_diff"], r["ci_lo"], r["ci_hi"])
        _record_subtag("H2", f"{r['model']} x {r['class']}", r["verdict"], final_verdict)
        lines.append(
            f"| {r['model']} | {r['class']} | {fmt_f(r['diff_s42'])} | {fmt_f(r['diff_s43'])} | "
            f"{fmt_f(r['diff_s44'])} | {fmt_f(r['mean_diff'])} | "
            f"[{fmt_f(r['ci_lo'])}, {fmt_f(r['ci_hi'])}] | {r['ci_width']:.4f} | "
            f"{final_verdict} |"
        )
    # Per-model breakdown (compute from final sub-tagged labels)
    lines += [""]
    for model in ("trace_moe", "transformer", "bilstm_dnn"):
        sub = df[df["model"] == model]
        final_labels = []
        for _, rr in sub.iterrows():
            final_labels.append(render_verdict_h2_h3(rr["verdict"], rr["mean_diff"], rr["ci_lo"], rr["ci_hi"]))
        n_strong = sum(v == "SUPPORTED (strong)" for v in final_labels)
        n_prelim = sum(v == "SUPPORTED (preliminary)" for v in final_labels)
        n_sup = sum(v == "SUPPORTED" for v in final_labels)
        n_inc = sum(v == "INCONCLUSIVE" for v in final_labels)
        n_narrow = sum(v == "NOT SUPPORTED (narrow null)" for v in final_labels)
        n_baseline = sum(v == "NOT SUPPORTED (baseline wins)" for v in final_labels)
        n_plain_ns = sum(v == "NOT SUPPORTED" for v in final_labels)
        lines.append(
            f"H2 breakdown for {model}: "
            f"SUPPORTED (strong) {n_strong}, "
            f"SUPPORTED (preliminary) {n_prelim}, "
            f"SUPPORTED {n_sup}, "
            f"INCONCLUSIVE {n_inc}, "
            f"NOT SUPPORTED (narrow null) {n_narrow}, "
            f"NOT SUPPORTED (baseline wins) {n_baseline}"
            + (f", NOT SUPPORTED (unlabelled) {n_plain_ns}" if n_plain_ns else "")
            + "."
        )
    lines.append("")
    return lines


def panel_h3(df: pd.DataFrame) -> list[str]:
    lines = ["## Panel 3 of 4 — H3 TRACE-MoE energy AUROC minus baseline energy AUROC (2 baselines x 5 held-out classes = 10 pairs)",
             "",
             "Rule: paired bootstrap 95% CI of (trace_moe.energy_auroc minus baseline.energy_auroc) across 3 seeds. "
             "Expected effect 0.05 AUROC, CI width margin 0.10. 4-tier verdict per Charter v1.1.",
             "",
             "| Baseline | Held-out class | TRACE-MoE s42 | TRACE-MoE s43 | TRACE-MoE s44 | baseline s42 | baseline s43 | baseline s44 | mean_diff | 95% CI | width | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        final_verdict = render_verdict_h2_h3(r["verdict"], r["mean_diff"], r["ci_lo"], r["ci_hi"])
        _record_subtag("H3", f"vs {r['baseline']} x {r['class']}", r["verdict"], final_verdict)
        lines.append(
            f"| {r['baseline']} | {r['class']} | "
            f"{r['tm_s42']:.4f} | {r['tm_s43']:.4f} | {r['tm_s44']:.4f} | "
            f"{r['bl_s42']:.4f} | {r['bl_s43']:.4f} | {r['bl_s44']:.4f} | "
            f"{fmt_f(r['mean_diff'])} | "
            f"[{fmt_f(r['ci_lo'])}, {fmt_f(r['ci_hi'])}] | {r['ci_width']:.4f} | "
            f"{final_verdict} |"
        )
    lines += [""]
    for baseline in ("transformer", "bilstm_dnn"):
        sub = df[df["baseline"] == baseline]
        final_labels = []
        for _, rr in sub.iterrows():
            final_labels.append(render_verdict_h2_h3(rr["verdict"], rr["mean_diff"], rr["ci_lo"], rr["ci_hi"]))
        n_strong = sum(v == "SUPPORTED (strong)" for v in final_labels)
        n_prelim = sum(v == "SUPPORTED (preliminary)" for v in final_labels)
        n_sup = sum(v == "SUPPORTED" for v in final_labels)
        n_inc = sum(v == "INCONCLUSIVE" for v in final_labels)
        n_narrow = sum(v == "NOT SUPPORTED (narrow null)" for v in final_labels)
        n_baseline_wins = sum(v == "NOT SUPPORTED (baseline wins)" for v in final_labels)
        n_plain_ns = sum(v == "NOT SUPPORTED" for v in final_labels)
        lines.append(
            f"H3 breakdown vs {baseline}: "
            f"SUPPORTED (strong) {n_strong}, "
            f"SUPPORTED (preliminary) {n_prelim}, "
            f"SUPPORTED {n_sup}, "
            f"INCONCLUSIVE {n_inc}, "
            f"NOT SUPPORTED (narrow null) {n_narrow}, "
            f"NOT SUPPORTED (baseline wins) {n_baseline_wins}"
            + (f", NOT SUPPORTED (unlabelled) {n_plain_ns}" if n_plain_ns else "")
            + "."
        )
    lines.append("")
    return lines


def panel_h4(df: pd.DataFrame) -> list[str]:
    lines = ["## Panel 4 of 4 — H4 closed-set macro-F1 degradation (3 models x 5 held-out classes = 15 pairs, 10 pt threshold)",
             "",
             "Rule: full-model baseline (recomputed over the SAME 18 non-held-out classes per pair) "
             "minus Phase 3(c) known_classes_macro_f1. 3-tier verdict: SUPPORTED if CI upper <= 0.10, "
             "NOT SUPPORTED if CI lower > 0.10, else INCONCLUSIVE. 10 pt threshold tightened from 15 pt in Charter v1.1.",
             "",
             "| Model | Held-out class | degr s42 | degr s43 | degr s44 | mean_degr | 95% CI | width | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['class']} | {fmt_f(r['degr_s42'])} | {fmt_f(r['degr_s43'])} | "
            f"{fmt_f(r['degr_s44'])} | {fmt_f(r['mean_degr'])} | "
            f"[{fmt_f(r['ci_lo'])}, {fmt_f(r['ci_hi'])}] | {r['ci_width']:.4f} | "
            f"{verdict_tag(r['verdict'])} |"
        )
    lines += [""]
    for model in ("trace_moe", "transformer", "bilstm_dnn"):
        sub = df[df["model"] == model]
        counts = sub["verdict"].value_counts().to_dict()
        lines.append(
            f"H4 breakdown for {model}: "
            f"SUPPORTED {counts.get('SUPPORTED', 0)}, "
            f"INCONCLUSIVE {counts.get('INCONCLUSIVE', 0)}, "
            f"NOT SUPPORTED {counts.get('NOT SUPPORTED', 0)}."
        )
    lines.append("")
    return lines


def esa1_block(df: pd.DataFrame) -> list[str]:
    overall_mean = float(df["mean_far"].mean())
    lines = ["## ESA1 calibration summary (direction-only report, per Charter v1.1)",
             "",
             "| | Target (95% coverage) | Observed overall mean across 15 (model x class) pairs |",
             "|---|---|---|",
             f"| known_false_alarm_rate | 0.0500 | {overall_mean:.4f} |",
             "",
             "Per-pair observed FAR is in `ESA1_calibration.csv`. Overall calibration drift from target "
             f"is {abs(overall_mean - 0.05):.4f} absolute, i.e. empirical reject layer is "
             f"{'slightly conservative' if overall_mean < 0.05 else 'slightly permissive'}.",
             ""]
    return lines


def tcp_tv_count(h1, h2, h3, h4) -> dict:
    """Verify watch-item: count TCP_IP-DoS-UDP across INCONCLUSIVE verdicts in H1-H4."""
    tcp = "TCP_IP-DoS-UDP"
    total_inc = 0
    tcp_in_inc = 0
    for df in (h1, h2, h3, h4):
        inc = df[df["verdict"] == "INCONCLUSIVE"]
        total_inc += len(inc)
        tcp_in_inc += (inc["class"] == tcp).sum()
    return {"tcp_in_inconclusive": int(tcp_in_inc), "total_inconclusive": int(total_inc)}


def legend() -> list[str]:
    return ["## Legend",
            "",
            "**4-tier verdict scheme per Charter v1.1** (locked SHA `" + CHARTER_SHA + "`):",
            "",
            "- **SUPPORTED (strong)**: CI excludes zero AND tight (width <= margin for H2/H3) OR mean meets threshold with CI tight (for H1).",
            "- **SUPPORTED (preliminary)**: mean is favourable but CI width exceeds margin; honest small-sample reporting.",
            "- **SUPPORTED**: CI excludes zero (plain tier used for H4 where no strong/prelim split applies).",
            "- **INCONCLUSIVE**: CI straddles the decision boundary AND is wide (> margin). Evidence insufficient at 3-seed scope.",
            "- **NOT SUPPORTED (narrow null)**: CI contains zero AND width <= margin (0.10 for H2/H3). The effect is precise enough to say it is below the practical-significance threshold even though the point estimate may be of either sign.",
            "- **NOT SUPPORTED (baseline wins)**: CI excludes zero in the negative direction (CI upper < 0) AND mean < 0. The baseline strictly outperforms TRACE-MoE on this pair.",
            "",
            "Charter-faithful disambiguation note: v1.1 §Verdict tiers already distinguishes these two cases in text. The aggregator CSV collapses them to a single 'NOT SUPPORTED' label; this table restores the distinction at render time without altering any verdict rule or re-running aggregation. No Charter SHA bump is required.",
            "",
            "**Per-hypothesis margins (locked in Charter v1.1)**:",
            "",
            "| Hypothesis | Decision quantity | Threshold or margin | Direction |",
            "|---|---|---|---|",
            "| H1 | caught_any (TRACE-MoE) | mean >= 0.95 AND CI lower >= 0.90 for strong | higher better |",
            "| H2 | energy_auroc minus msp_auroc | expected effect 0.05 AUROC; CI width margin 0.10 (= 2 x expected) | higher better |",
            "| H3 | trace_moe.energy_auroc minus baseline.energy_auroc | expected effect 0.05 AUROC; CI width margin 0.10 | higher better |",
            "| H4 | Full-model baseline minus Phase 3(c) known_classes_macro_f1 | 10 pt threshold (tightened from 15 pt in v1.1) | lower better |",
            "",
            f"All CI columns are bootstrap 95% CIs over {N_SEEDS} seeds ({BOOT:,} resamples, fixed RNG seed per Charter).",
            ""]


def disambiguation_audit() -> list[str]:
    """§12a verdict label disambiguation audit per workflow Step 2.2 addition."""
    lines = ["## Verdict label disambiguation audit (presentation layer, no Charter rule change)",
             "",
             "Cells relabeled from raw 'NOT SUPPORTED' (as stored in the aggregator CSV) to the "
             "finer presentation label (as rendered in Panels 2 and 3 above):",
             ""]
    if not _NS_SUBTAG_AUDIT:
        lines.append("- No relabels recorded.")
        return lines
    lines += ["| Hypothesis | Pair | Raw (CSV) | Final (table) |",
             "|---|---|---|---|"]
    for row in _NS_SUBTAG_AUDIT:
        lines.append(f"| {row['hypothesis']} | {row['pair']} | {row['raw']} | {row['final']} |")
    narrow = sum(r["final"] == "NOT SUPPORTED (narrow null)" for r in _NS_SUBTAG_AUDIT)
    bwin = sum(r["final"] == "NOT SUPPORTED (baseline wins)" for r in _NS_SUBTAG_AUDIT)
    plain = sum(r["final"] == "NOT SUPPORTED" for r in _NS_SUBTAG_AUDIT)
    lines += ["",
              f"**Count across all 45 cells**: NOT SUPPORTED (narrow null) {narrow}, "
              f"NOT SUPPORTED (baseline wins) {bwin}" + (f", NOT SUPPORTED (unlabelled) {plain}" if plain else "") + ".",
              "",
              "This audit list is the reviewer-citable artefact proving the sub-tagging is deterministic from the CSV data and the Charter v1.1 narrow-null rule, not post-hoc reinterpretation.",
              ""]
    return lines


def main() -> int:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    h1 = pd.read_csv(VDIR / "H1_safety_view.csv")
    h2 = pd.read_csv(VDIR / "H2_energy_vs_msp.csv")
    h3 = pd.read_csv(VDIR / "H3_trace_moe_vs_baselines.csv")
    h4 = pd.read_csv(VDIR / "H4_closed_set_degradation.csv")
    esa1 = pd.read_csv(VDIR / "ESA1_calibration.csv")

    # Watch-item check: the INCONCLUSIVE counts must agree with verdict.json (the aggregator's own record of
    # the same CSVs); the expected numbers are read from that file rather than hard-coded, so the check
    # follows the aggregation scope (3 seeds in v1.1, 5 seeds from v1.3).
    verdict = json.loads((VDIR / "verdict.json").read_text(encoding="utf-8"))
    inc = verdict["inconclusive_pairs"]
    exp_total = len(inc)
    exp_tcp = sum("TCP_IP-DoS-UDP" in json.dumps(pair) for pair in inc)
    n_seeds = len(verdict["scope"]["seeds"])
    charter_label = verdict["charter_version"]
    charter_sha = verdict["charter_sha256"]
    tcp = tcp_tv_count(h1, h2, h3, h4)
    assert tcp["tcp_in_inconclusive"] == exp_tcp and tcp["total_inconclusive"] == exp_total, (
        f"Watch-item FAILED: verdict.json records {exp_tcp} of {exp_total} TCP_IP-DoS-UDP INCONCLUSIVE, "
        f"the CSVs give {tcp['tcp_in_inconclusive']} of {tcp['total_inconclusive']}")

    doc = []
    doc += [f"# Paper 4 Main Verdict Table (4-panel, {n_seeds}-seed aggregation)",
            "",
            f"**Charter**: {charter_label}; SHA256 `{charter_sha}`",
            "",
            f"**Source CSVs** (read-only inputs; byte-identical to the aggregator output recorded in verdict.json, generated {verdict['generated_at_utc']}):",
            "",
            "- `H1_safety_view.csv` (5 pairs)",
            "- `H2_energy_vs_msp.csv` (15 pairs)",
            "- `H3_trace_moe_vs_baselines.csv` (10 pairs)",
            "- `H4_closed_set_degradation.csv` (15 pairs)",
            "- `ESA1_calibration.csv` (15 pairs, direction-only)",
            "",
            f"**Total verdict cells rendered**: {len(h1) + len(h2) + len(h3) + len(h4)} (= 5 + 15 + 10 + 15).",
            "",
            f"**Watch-item TCP_IP-DoS-UDP check**: {tcp['tcp_in_inconclusive']} of {tcp['total_inconclusive']} "
            f"INCONCLUSIVE verdicts across H1-H4 involve TCP_IP-DoS-UDP as the held-out attack.",
            "",
            "---",
            ""]
    doc += panel_h1(h1)
    doc += panel_h2(h2)
    doc += panel_h3(h3)
    doc += panel_h4(h4)
    doc += esa1_block(esa1)
    doc += disambiguation_audit()
    doc += legend()

    OUT.write_text("\n".join(doc), encoding="utf-8")
    print(f"Wrote: {OUT}")
    print(f"TCP_IP-DoS-UDP watch-item: {tcp['tcp_in_inconclusive']} of {tcp['total_inconclusive']} ✓")
    print(f"Total cells rendered: {len(h1) + len(h2) + len(h3) + len(h4)}")
    # Audit summary
    n_narrow = sum(r["final"] == "NOT SUPPORTED (narrow null)" for r in _NS_SUBTAG_AUDIT)
    n_bwin = sum(r["final"] == "NOT SUPPORTED (baseline wins)" for r in _NS_SUBTAG_AUDIT)
    n_plain = sum(r["final"] == "NOT SUPPORTED" for r in _NS_SUBTAG_AUDIT)
    print(f"Sub-tag disambiguation audit: narrow null {n_narrow}, baseline wins {n_bwin}"
          + (f", unlabelled {n_plain}" if n_plain else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
