"""
scripts/paper4_h8_verdict.py

Paper 4 H1-H4 verdict + ESA1 aggregation.

Pre-registration: documents_v2/paper4/PAPER4_CHARTER.md (locked 2026-10-14 edit pass 1)

Reads:
  - results_v3_paper3/zero_day/<model>_loao_T16_<class>_seed{42,43,44}/metrics.json
    (Phase 3(c), 45 LOAO runs)
  - results_v2/main/<model>_official_T16_seed{42,43,44}/metrics.json
    (full-model baseline for H4; same 3 seeds)

Writes:
  - results_v3_paper3/h8_verdict/per_seed_tables.csv
  - results_v3_paper3/h8_verdict/aggregate_tables.csv
  - results_v3_paper3/h8_verdict/H1_safety_view.{csv,md}
  - results_v3_paper3/h8_verdict/H2_energy_vs_msp.{csv,md}
  - results_v3_paper3/h8_verdict/H3_trace_moe_vs_baselines.{csv,md}
  - results_v3_paper3/h8_verdict/H4_closed_set_degradation.{csv,md}
  - results_v3_paper3/h8_verdict/ESA1_calibration.{csv,md}
  - results_v3_paper3/h8_verdict/VERDICT_SUMMARY.md
  - results_v3_paper3/h8_verdict/verdict.json
  - results_v3_paper3/h8_verdict/ESA2_DEFERRED.md

Does NOT modify any other code path.
Does NOT touch the paper draft.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PHASE3C_DIR = ROOT / "results_v3_paper3" / "zero_day"
REF_DIR = ROOT / "results_v2" / "main"
VERDICT_DIR = ROOT / "results_v3_paper3" / "h8_verdict"
VERDICT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]  # 5-seed scope per Charter v1.2 seed extension (locked ba703e25...)
MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data",
           "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
BOOT_N = 10_000
RNG_SEED = 42  # fixed for reproducibility of bootstrap resampling

OOD_METRICS = [
    "energy_auroc", "energy_fpr_at_95tpr",
    "msp_auroc", "msp_fpr_at_95tpr",
    "held_out_detected_as_attack",
    "held_out_flagged_unknown",
    "held_out_caught_any",
    "known_false_alarm_rate",
    "known_classes_macro_f1",
]


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

def load_phase3c_metrics() -> pd.DataFrame:
    """Load the 45 Phase 3(c) ood blocks + known_classes_macro_f1 into a long table."""
    rows = []
    for model in MODELS:
        for cls in CLASSES:
            for seed in SEEDS:
                run_dir = PHASE3C_DIR / f"{model}_loao_T16_{cls}_seed{seed}"
                mj = run_dir / "metrics.json"
                if not mj.exists():
                    raise FileNotFoundError(f"Missing metrics.json: {mj}")
                m = json.loads(mj.read_text())
                ood = m["ood"]
                rows.append({
                    "model": model, "class": cls, "seed": seed,
                    "energy_auroc":                 float(ood["energy"]["auroc"]),
                    "energy_fpr_at_95tpr":          float(ood["energy"]["fpr_at_95tpr"]),
                    "msp_auroc":                    float(ood["msp"]["auroc"]),
                    "msp_fpr_at_95tpr":             float(ood["msp"]["fpr_at_95tpr"]),
                    "held_out_detected_as_attack":  float(ood["held_out_detected_as_attack"]),
                    "held_out_flagged_unknown":     float(ood["held_out_flagged_unknown"]),
                    "held_out_caught_any":          float(ood["held_out_caught_any"]),
                    "known_false_alarm_rate":       float(ood["known_false_alarm_rate"]),
                    "known_classes_macro_f1":       float(ood["known_classes_macro_f1"]),
                })
    df = pd.DataFrame(rows)
    expected_rows = len(MODELS) * len(CLASSES) * len(SEEDS)  # 3 x 5 x 5 = 75 at 5-seed scope
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    return df


def load_reference_baseline_18class() -> pd.DataFrame:
    """For each (model, held-out class, seed) compute the full model's label19 macro-F1
    RESTRICTED to the 18 non-held-out classes. This is the fair baseline for H4.

    The full model is trained on all 19 classes, so per_class F1 for the held-out class
    is non-trivial; excluding it gives the apples-to-apples closed-set baseline.

    Bug fix 2026-11-13: prefer `__v2` suffix over the bare name when both exist,
    matching memory [[seed-42-v2-canonical]]. This mirrors the pattern already
    used in `scripts/analyze_h5_ablation.py::read_phase3a`. Currently inert for
    Paper 4 because the three MODELS have no `__v2` folders in results_v2/main/,
    but the latent inconsistency is now closed.
    """
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            ref_file = None
            for suffix in ["__v2", ""]:
                cand = REF_DIR / f"{model}_official_T16_seed{seed}{suffix}" / "metrics.json"
                if cand.exists():
                    ref_file = cand
                    break
            if ref_file is None:
                raise FileNotFoundError(
                    f"Missing full-model baseline for {model} seed {seed}: "
                    f"neither {REF_DIR / (model + '_official_T16_seed' + str(seed) + '__v2') / 'metrics.json'} "
                    f"nor {REF_DIR / (model + '_official_T16_seed' + str(seed)) / 'metrics.json'} exists"
                )
            m = json.loads(ref_file.read_text())
            per_class = m["test"]["label19"]["per_class"]
            # per_class: {class_name: {f1, precision, recall, support}}
            for cls in CLASSES:
                if cls not in per_class:
                    raise ValueError(f"Class {cls!r} not in full-model per_class keys: {list(per_class.keys())[:3]}...")
                # macro-F1 over the 18 non-held-out classes with non-zero support
                f1s = [v["f1"] for k, v in per_class.items() if k != cls and v["support"] > 0]
                baseline_18 = float(np.mean(f1s))
                rows.append({
                    "model": model, "class": cls, "seed": seed,
                    "ref_macro_f1_18class": baseline_18,
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def bootstrap_ci_mean(values: np.ndarray, n: int = BOOT_N, rng=None) -> tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) for the mean of `values` via nonparametric bootstrap."""
    rng = rng or np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = values[idx].mean(axis=1)
    mean = float(values.mean())
    lower, upper = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return mean, lower, upper


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n: int = BOOT_N, rng=None,
                          ) -> tuple[float, float, float]:
    """Paired bootstrap for mean(a - b)."""
    rng = rng or np.random.default_rng(RNG_SEED + 1)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert len(a) == len(b), f"paired bootstrap needs equal len; got {len(a)} vs {len(b)}"
    diff = a - b
    return bootstrap_ci_mean(diff, n=n, rng=rng)


# -----------------------------------------------------------------------------
# Verdict helpers
# -----------------------------------------------------------------------------

def h1_verdict(mean: float, lo: float, hi: float, strong_mean=0.95, strong_lo=0.90,
               ) -> str:
    """H1 four-tier verdict per Charter."""
    if mean >= strong_mean and lo >= strong_lo:
        return "SUPPORTED (strong)"
    if mean >= strong_mean and lo < strong_lo:
        return "SUPPORTED (preliminary)"
    if mean < strong_mean and hi >= strong_mean:  # CI straddles 0.95
        return "INCONCLUSIVE"
    if hi < strong_mean:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"  # defensive


def h2_pair_verdict_v1_1(mean_diff: float, lo: float, hi: float,
                          expected_effect: float = 0.05) -> str:
    """H2/H3 four-tier verdict per Charter v1.1 (edit pass 2).

    Expected effect size = 0.05 AUROC. CI width threshold = 2 × expected = 0.10.

    - SUPPORTED: CI excludes 0 AND mean > 0
    - SUPPORTED (preliminary): mean > 0 BUT CI width > 2 × expected
    - INCONCLUSIVE: CI straddles 0 AND CI width > 2 × expected
    - NOT SUPPORTED: CI excludes 0 AND mean < 0 (baseline wins), OR
                     CI contains 0 AND width ≤ 2 × expected (narrow null)
    """
    ci_width = hi - lo
    width_threshold = 2 * expected_effect

    # SUPPORTED (CI excludes 0, mean > 0)
    if lo > 0 and mean_diff > 0:
        return "SUPPORTED"
    # NOT SUPPORTED (CI excludes 0, mean < 0 — baseline wins)
    if hi < 0 and mean_diff < 0:
        return "NOT SUPPORTED"
    # Now CI contains 0
    if ci_width <= width_threshold:
        # Narrow null — NOT SUPPORTED
        return "NOT SUPPORTED"
    # CI contains 0 AND is wide (> 2 × expected)
    if mean_diff > 0:
        # Point estimate favorable, CI wide → preliminary
        return "SUPPORTED (preliminary)"
    # Point estimate not favorable, CI wide → inconclusive
    return "INCONCLUSIVE"


def h2_h3_overall_verdict_v1_1(verdicts: list[str]) -> str:
    """H2/H3 overall verdict per model (or per baseline for H3), harmonized with H1.

    5 class pairs → aggregation rules:
    - SUPPORTED (strong): ≥ 3/5 SUPPORTED
    - SUPPORTED (preliminary): ≥ 3/5 at least preliminary AND ≥ 1 preliminary
    - MIXED: some NOT SUPPORTED + some SUPPORTED
    - NOT SUPPORTED: ≥ 3/5 NOT SUPPORTED
    - INCONCLUSIVE: ≥ 3/5 INCONCLUSIVE
    """
    sup = sum(v == "SUPPORTED" for v in verdicts)
    prelim = sum(v == "SUPPORTED (preliminary)" for v in verdicts)
    inc = sum(v == "INCONCLUSIVE" for v in verdicts)
    not_sup = sum(v == "NOT SUPPORTED" for v in verdicts)
    at_least_sup = sup + prelim

    if sup >= 3:
        return "SUPPORTED (strong)"
    if at_least_sup >= 3 and prelim >= 1:
        return "SUPPORTED (preliminary)"
    if not_sup >= 3:
        return "NOT SUPPORTED"
    if inc >= 3:
        return "INCONCLUSIVE"
    if sup > 0 and not_sup > 0:
        return "MIXED"
    return "MIXED"  # fallback


def h4_pair_verdict_v1_1(mean_degr: float, lo: float, hi: float,
                          margin: float = 0.10) -> str:
    """H4 verdict per (model × class) under v1.1 tightened 10-pt threshold.

    - SUPPORTED: CI upper ≤ 0.10 AND mean ≤ 0.10
    - NOT SUPPORTED: CI lower > 0.10 AND mean > 0.10
    - INCONCLUSIVE: CI straddles 0.10
    """
    if hi <= margin and mean_degr <= margin:
        return "SUPPORTED"
    if lo > margin and mean_degr > margin:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


# Back-compat aliases if any legacy code imports the old names
h2_pair_verdict = h2_pair_verdict_v1_1
h4_pair_verdict = h4_pair_verdict_v1_1


def _per_seed_cols(prefix: str, values) -> dict:
    """Map an aligned-by-SEEDS value vector to columns like {prefix}_s{seed}."""
    return {f"{prefix}_s{s}": float(values[i]) for i, s in enumerate(SEEDS)}


def _per_seed_named_cols(prefix: str, per_seed_dict: dict) -> dict:
    """Map a {seed: value} dict to columns like {prefix}{seed}."""
    return {f"{prefix}{s}": per_seed_dict.get(s, np.nan) for s in SEEDS}


# -----------------------------------------------------------------------------
# H1 — safety view
# -----------------------------------------------------------------------------

def compute_h1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """H1 for TRACE-MoE only: `held_out_caught_any` ≥ 0.95 per class."""
    tm = df[df["model"] == "trace_moe"].copy()
    rows = []
    for cls in CLASSES:
        sub = tm[tm["class"] == cls]
        per_seed = {int(r["seed"]): float(r["held_out_caught_any"])
                    for _, r in sub.iterrows()}
        values = np.array([per_seed[s] for s in SEEDS])
        mean, lo, hi = bootstrap_ci_mean(values)
        verdict = h1_verdict(mean, lo, hi)
        row = {"class": cls}
        row.update(_per_seed_named_cols("seed", per_seed))
        row.update({
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "ci_width": hi - lo,
            "verdict": verdict,
        })
        rows.append(row)
    h1_df = pd.DataFrame(rows)

    # Overall H1 verdict
    verdicts = h1_df["verdict"].tolist()
    inc_count = sum(v == "INCONCLUSIVE" for v in verdicts)
    not_sup_count = sum(v == "NOT SUPPORTED" for v in verdicts)
    strong_count = sum(v == "SUPPORTED (strong)" for v in verdicts)
    prelim_count = sum(v == "SUPPORTED (preliminary)" for v in verdicts)

    if not_sup_count > 0:
        overall = "NOT SUPPORTED (overall)"
    elif inc_count >= 2:
        overall = f"INCONCLUSIVE (overall; {len(SEEDS)}-seed scope insufficient)"
    elif strong_count == 5:
        overall = "SUPPORTED (strong, overall)"
    elif strong_count + prelim_count == 5:
        overall = "SUPPORTED (preliminary, overall)"
    else:
        overall = "MIXED"

    summary = {
        "overall_verdict": overall,
        "per_class_breakdown": {
            "SUPPORTED (strong)": strong_count,
            "SUPPORTED (preliminary)": prelim_count,
            "INCONCLUSIVE": inc_count,
            "NOT SUPPORTED": not_sup_count,
        },
    }
    return h1_df, summary


# -----------------------------------------------------------------------------
# H2 — energy AUROC > MSP AUROC per (model × class)
# -----------------------------------------------------------------------------

def compute_h2(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Paired bootstrap of (energy_auroc - msp_auroc) across 3 seeds per (model × class)."""
    rows = []
    for model in MODELS:
        for cls in CLASSES:
            sub = df[(df["model"] == model) & (df["class"] == cls)].sort_values("seed")
            energy = sub["energy_auroc"].to_numpy()
            msp = sub["msp_auroc"].to_numpy()
            mean_d, lo, hi = bootstrap_ci_mean_diff(energy, msp)
            verdict = h2_pair_verdict(mean_d, lo, hi)
            row = {"model": model, "class": cls}
            row.update(_per_seed_cols("energy", energy))
            row.update(_per_seed_cols("msp", msp))
            row.update(_per_seed_cols("diff", energy - msp))
            row.update({
                "mean_diff": mean_d, "ci_lo": lo, "ci_hi": hi,
                "ci_width":  hi - lo,
                "verdict": verdict,
            })
            rows.append(row)
    h2_df = pd.DataFrame(rows)

    # Overall per model (v1.1: harmonized 4-tier via h2_h3_overall_verdict_v1_1)
    overall_per_model = {}
    for model in MODELS:
        sub = h2_df[h2_df["model"] == model]
        verdicts = sub["verdict"].tolist()
        overall = h2_h3_overall_verdict_v1_1(verdicts)
        overall_per_model[model] = {
            "verdict": overall,
            "SUPPORTED": int(sum(v == "SUPPORTED" for v in verdicts)),
            "SUPPORTED (preliminary)": int(sum(v == "SUPPORTED (preliminary)" for v in verdicts)),
            "INCONCLUSIVE": int(sum(v == "INCONCLUSIVE" for v in verdicts)),
            "NOT SUPPORTED": int(sum(v == "NOT SUPPORTED" for v in verdicts)),
        }

    summary = {
        "per_model_breakdown": overall_per_model,
        "primary_question": "Does energy abstention outperform MSP for TRACE-MoE?",
        "primary_answer": overall_per_model["trace_moe"]["verdict"],
    }
    return h2_df, summary


# -----------------------------------------------------------------------------
# H3 — TRACE-MoE energy AUROC > each baseline per class
# -----------------------------------------------------------------------------

def compute_h3(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Paired bootstrap of (trace_moe.energy_auroc - baseline.energy_auroc) per class."""
    tm = df[df["model"] == "trace_moe"].copy()
    rows = []
    for baseline in ["transformer", "bilstm_dnn"]:
        bl = df[df["model"] == baseline].copy()
        for cls in CLASSES:
            tm_vals = tm[tm["class"] == cls].sort_values("seed")["energy_auroc"].to_numpy()
            bl_vals = bl[bl["class"] == cls].sort_values("seed")["energy_auroc"].to_numpy()
            mean_d, lo, hi = bootstrap_ci_mean_diff(tm_vals, bl_vals)
            verdict = h2_pair_verdict(mean_d, lo, hi)
            row = {"baseline": baseline, "class": cls}
            row.update(_per_seed_cols("tm", tm_vals))
            row.update(_per_seed_cols("bl", bl_vals))
            row.update(_per_seed_cols("diff", tm_vals - bl_vals))
            row.update({
                "mean_diff": mean_d, "ci_lo": lo, "ci_hi": hi,
                "ci_width": hi - lo,
                "verdict": verdict,
            })
            rows.append(row)
    h3_df = pd.DataFrame(rows)

    # Overall per baseline (v1.1: harmonized 4-tier)
    per_baseline = {}
    for baseline in ["transformer", "bilstm_dnn"]:
        sub = h3_df[h3_df["baseline"] == baseline]
        verdicts = sub["verdict"].tolist()
        overall = h2_h3_overall_verdict_v1_1(verdicts)
        per_baseline[baseline] = {
            "verdict": overall,
            "SUPPORTED": int(sum(v == "SUPPORTED" for v in verdicts)),
            "SUPPORTED (preliminary)": int(sum(v == "SUPPORTED (preliminary)" for v in verdicts)),
            "INCONCLUSIVE": int(sum(v == "INCONCLUSIVE" for v in verdicts)),
            "NOT SUPPORTED": int(sum(v == "NOT SUPPORTED" for v in verdicts)),
        }

    summary = {"per_baseline_breakdown": per_baseline}
    return h3_df, summary


# -----------------------------------------------------------------------------
# H4 — closed-set degradation ≤ 15 pt per (model × class) vs the full-model baseline
# -----------------------------------------------------------------------------

def compute_h4(df: pd.DataFrame, baseline_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For each (model × class): degradation = Reference_18class_macroF1 - Phase3c known_classes_macro_f1."""
    merged = df.merge(baseline_df, on=["model", "class", "seed"], how="left")
    merged["degradation"] = merged["ref_macro_f1_18class"] - merged["known_classes_macro_f1"]
    rows = []
    for model in MODELS:
        for cls in CLASSES:
            sub = merged[(merged["model"] == model) & (merged["class"] == cls)].sort_values("seed")
            vals = sub["degradation"].to_numpy()
            baseline_vals = sub["ref_macro_f1_18class"].to_numpy()
            phase3c_vals = sub["known_classes_macro_f1"].to_numpy()
            mean_d, lo, hi = bootstrap_ci_mean(vals)
            verdict = h4_pair_verdict_v1_1(mean_d, lo, hi, margin=0.10)
            row = {"model": model, "class": cls}
            row.update(_per_seed_cols("p1", baseline_vals))
            row.update(_per_seed_cols("p3c", phase3c_vals))
            row.update(_per_seed_cols("degr", vals))
            row.update({
                "mean_degr": mean_d, "ci_lo": lo, "ci_hi": hi,
                "ci_width": hi - lo,
                "verdict": verdict,
            })
            rows.append(row)
    h4_df = pd.DataFrame(rows)
    sup = (h4_df["verdict"] == "SUPPORTED").sum()
    inc = (h4_df["verdict"] == "INCONCLUSIVE").sum()
    not_sup = (h4_df["verdict"] == "NOT SUPPORTED").sum()
    if not_sup >= 2:
        overall = "NOT SUPPORTED (overall; ≥2 pairs exceed 10 pt)"
    elif inc >= 3:
        overall = f"INCONCLUSIVE (overall; ≥3 pairs inconclusive at {len(SEEDS)}-seed scope)"
    elif sup == 15:
        overall = "SUPPORTED (overall; all 15 pairs within 10 pt)"
    elif sup >= 10:
        overall = "MIXED (majority supported)"
    else:
        overall = "MIXED"
    summary = {
        "overall_verdict": overall,
        "breakdown": {"SUPPORTED": int(sup), "INCONCLUSIVE": int(inc), "NOT SUPPORTED": int(not_sup)},
        "margin_pt": 10,  # v1.1 tightened from 15
    }
    return h4_df, summary


# -----------------------------------------------------------------------------
# ESA1 — reject-threshold calibration
# -----------------------------------------------------------------------------

def compute_esa1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """known_false_alarm_rate vs target 0.05 (95% coverage)."""
    rows = []
    for model in MODELS:
        for cls in CLASSES:
            sub = df[(df["model"] == model) & (df["class"] == cls)].sort_values("seed")
            far = sub["known_false_alarm_rate"].to_numpy()
            deviation = far - 0.05  # positive = over-flagging
            mean_d, lo, hi = bootstrap_ci_mean(deviation)
            row = {"model": model, "class": cls}
            row.update(_per_seed_cols("far", far))
            row.update({
                "mean_far": float(far.mean()),
                "dev_from_target_0.05": mean_d,
                "ci_lo": lo, "ci_hi": hi,
            })
            rows.append(row)
    esa1_df = pd.DataFrame(rows)
    summary = {
        "target_coverage": 0.95,
        "target_far": 0.05,
        "overall_mean_far": float(esa1_df["mean_far"].mean()),
        "report_only": "Direction-only reporting per Charter; no formal verdict",
    }
    return esa1_df, summary


# -----------------------------------------------------------------------------
# ESA2 — DEFERRED
# -----------------------------------------------------------------------------

ESA2_DEFERRED_NOTE = """# ESA2 — Spearman ρ between energy and (1 − MSP): DEFERRED

Per Charter §Exploratory Supplementary Analyses, ESA2 requires per-sample energy
scores for the held-out (unknown) attack subset of each test window.

**Blocker**: `results_v3_paper3/zero_day/<run>/predictions.npz` does NOT contain
per-sample energy values. Keys present (verified 2026-10-14):

  y_true_label2, y_pred_label2, proba_label2,
  y_true_label6, y_pred_label6, proba_label6,
  y_true_label19, y_pred_label19, proba_label19,
  val_y_true_label2, val_y_pred_label2, val_proba_label2,
  val_y_true_label6, val_y_pred_label6, val_proba_label6,
  val_y_true_label19, val_y_pred_label19, val_proba_label19,
  route

Energy = -logsumexp(logits) cannot be recovered from softmax probabilities alone
(the normalisation constant is lost in the softmax). MSP can be recovered as
`proba_label19.max(axis=1)` but energy requires either the raw logits or a
re-inference pass that saves energy.

**Resolution options** (none executed; await user decision):

- **Option A** — Re-run Script 07 for the 45 LOAO runs with an added energy-export
  flag. Compute cost: ~2-3 h on your i7-12700H / RTX 3050 since inference-only
  (no retraining); the model.pt checkpoints are already on disk. This produces
  per-sample energy arrays and enables ESA2 proper.

- **Option B** — Build a post-hoc `scripts/paper4_recompute_energy.py` that loads
  each saved model.pt, performs inference on the test windows, extracts logits,
  computes energy, and saves per-run `energy_test.npy`. Then re-run this
  verdict script with ESA2 enabled. Compute cost: similar to Option A, ~2-3 h.

- **Option C** — Replace ESA2 with a weaker proxy using `-log(max_proba)` as a
  "softmax-energy" surrogate. This is TRIVIALLY correlated with (1 − MSP) by
  construction (ρ ≈ 1.0) and therefore DOES NOT answer ESA2's question. Not
  recommended; would be a vacuous ESA2 report.

- **Option D** — Mark ESA2 as "deferred to future work" in the Paper 4
  manuscript. Charter already permits this at §ESA2.

**Recommended**: Option B (post-hoc per-run energy recompute). It produces real
ESA2 numbers from the existing checkpoints without re-training.

Decision pending user approval.
"""


# -----------------------------------------------------------------------------
# Write artefacts
# -----------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}" if isinstance(x, float) else str(x)


def write_h1(df: pd.DataFrame, summary: dict):
    df.to_csv(VERDICT_DIR / "H1_safety_view.csv", index=False)
    md = ["# H1 — Safety-view detection rate (TRACE-MoE caught_any ≥ 0.95)",
          "",
          "**Charter rule**: four-tier verdict.",
          "- SUPPORTED (strong): mean ≥ 0.95 AND CI lower bound ≥ 0.90",
          "- SUPPORTED (preliminary): mean ≥ 0.95, CI lower bound < 0.90",
          "- INCONCLUSIVE: mean < 0.95, CI straddles 0.95",
          "- NOT SUPPORTED: CI upper bound < 0.95",
          "",
          f"**Overall verdict**: `{summary['overall_verdict']}`",
          "",
          "## Per class (TRACE-MoE caught_any)",
          "",
          "| Class | " + " | ".join(f"seed{s}" for s in SEEDS) + " | mean | CI 95% | CI width | Verdict |",
          "|---|" + "---|" * len(SEEDS) + "---|---|---|---|"]
    for _, r in df.iterrows():
        seed_cells = " | ".join(f"{r[f'seed{s}']:.4f}" for s in SEEDS)
        md.append(f"| {r['class']} | {seed_cells} | "
                  f"{r['mean']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | "
                  f"{r['ci_width']:.4f} | {r['verdict']} |")
    (VERDICT_DIR / "H1_safety_view.md").write_text("\n".join(md), encoding="utf-8")


def write_h2(df: pd.DataFrame, summary: dict):
    df.to_csv(VERDICT_DIR / "H2_energy_vs_msp.csv", index=False)
    md = ["# H2 — Energy abstention vs MSP AUROC",
          "",
          "**Charter rule**: paired bootstrap 95% CI of (energy_auroc − msp_auroc).",
          "- SUPPORTED: CI excludes 0 AND mean > 0",
          "- NOT SUPPORTED: CI excludes 0 AND mean < 0",
          f"- INCONCLUSIVE: CI includes 0 (CI-wider-than-margin at {len(SEEDS)}-seed scope)",
          "",
          "**Per-model overall**:"]
    for model, s in summary["per_model_breakdown"].items():
        md.append(f"- `{model}`: **{s['verdict']}** "
                  f"(SUPPORTED {s['SUPPORTED']}, INC {s['INCONCLUSIVE']}, NOT {s['NOT SUPPORTED']})")
    md += ["",
           "**Primary H2 answer (for TRACE-MoE)**: " + summary["primary_answer"],
           "",
           "## Per (model × class) pairs",
           "",
           "| Model | Class | " + " | ".join(f"diff_s{s}" for s in SEEDS) + " | mean_diff | CI 95% | Verdict |",
           "|---|---|" + "---|" * len(SEEDS) + "---|---|---|"]
    for _, r in df.iterrows():
        diff_cells = " | ".join(f"{r[f'diff_s{s}']:+.4f}" for s in SEEDS)
        md.append(f"| {r['model']} | {r['class']} | {diff_cells} | "
                  f"{r['mean_diff']:+.4f} | "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['verdict']} |")
    (VERDICT_DIR / "H2_energy_vs_msp.md").write_text("\n".join(md), encoding="utf-8")


def write_h3(df: pd.DataFrame, summary: dict):
    df.to_csv(VERDICT_DIR / "H3_trace_moe_vs_baselines.csv", index=False)
    md = ["# H3 — TRACE-MoE energy AUROC vs baselines",
          "",
          "**Charter rule**: paired bootstrap of (trace_moe.energy_auroc − baseline.energy_auroc).",
          "- SUPPORTED (per-baseline overall): ≥ 3 of 5 class pairs SUPPORTED",
          "- NOT SUPPORTED (per-baseline overall): ≥ 3 of 5 class pairs NOT SUPPORTED",
          "- INCONCLUSIVE (per-baseline overall): ≥ 3 of 5 class pairs INCONCLUSIVE",
          "- MIXED (per-baseline overall): otherwise",
          "",
          "**Per-baseline overall**:"]
    for b, s in summary["per_baseline_breakdown"].items():
        md.append(f"- vs `{b}`: **{s['verdict']}** "
                  f"(SUPPORTED {s['SUPPORTED']}, INC {s['INCONCLUSIVE']}, NOT {s['NOT SUPPORTED']})")
    md += ["",
           "## Per (baseline × class) pairs",
           "",
           "| Baseline | Class | " + " | ".join(f"diff_s{s}" for s in SEEDS) + " | mean_diff | CI 95% | Verdict |",
           "|---|---|" + "---|" * len(SEEDS) + "---|---|---|"]
    for _, r in df.iterrows():
        diff_cells = " | ".join(f"{r[f'diff_s{s}']:+.4f}" for s in SEEDS)
        md.append(f"| {r['baseline']} | {r['class']} | {diff_cells} | "
                  f"{r['mean_diff']:+.4f} | "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['verdict']} |")
    (VERDICT_DIR / "H3_trace_moe_vs_baselines.md").write_text("\n".join(md), encoding="utf-8")


def write_h4(df: pd.DataFrame, summary: dict):
    df.to_csv(VERDICT_DIR / "H4_closed_set_degradation.csv", index=False)
    md = ["# H4 — Closed-set macro-F1 degradation ≤ 10 pt (v1.1 tightened from 15 pt)",
          "",
          "**Charter rule**: full-model baseline (recomputed over the SAME 18 "
          "non-held-out classes per pair) minus Phase 3(c) `known_classes_macro_f1`. "
          "Positive degradation = Phase 3(c) worse than the full-model baseline.",
          "- SUPPORTED (per-pair): CI upper bound ≤ 0.10 AND mean ≤ 0.10",
          "- NOT SUPPORTED (per-pair): CI lower bound > 0.10 AND mean > 0.10",
          "- INCONCLUSIVE (per-pair): CI straddles 0.10",
          "",
          "**Rationale for 10 pt threshold** (v1.1): 10 pt is the common open-set",
          "robustness convention and a deployment-acceptable macro-F1 degradation bar",
          "for IoMT NIDS.",
          "",
          f"**Overall verdict**: `{summary['overall_verdict']}`",
          f"- Breakdown: SUPPORTED {summary['breakdown']['SUPPORTED']}, "
          f"INC {summary['breakdown']['INCONCLUSIVE']}, "
          f"NOT {summary['breakdown']['NOT SUPPORTED']}",
          "",
          "## Per (model × class) pairs",
          "",
          "| Model | Class | " + " | ".join(f"p1_s{s}" for s in SEEDS)
          + " | " + " | ".join(f"p3c_s{s}" for s in SEEDS)
          + " | mean_degr | CI 95% | Verdict |",
          "|---|---|" + "---|" * (2 * len(SEEDS)) + "---|---|---|"]
    for _, r in df.iterrows():
        p1_cells = " | ".join(f"{r[f'p1_s{s}']:.4f}" for s in SEEDS)
        p3c_cells = " | ".join(f"{r[f'p3c_s{s}']:.4f}" for s in SEEDS)
        md.append(f"| {r['model']} | {r['class']} | "
                  f"{p1_cells} | {p3c_cells} | "
                  f"{r['mean_degr']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['verdict']} |")
    (VERDICT_DIR / "H4_closed_set_degradation.md").write_text("\n".join(md), encoding="utf-8")


def write_esa1(df: pd.DataFrame, summary: dict):
    df.to_csv(VERDICT_DIR / "ESA1_calibration.csv", index=False)
    md = ["# ESA1 — Reject-threshold calibration quality",
          "",
          f"**Target coverage**: {summary['target_coverage']:.2f} → target "
          f"`known_false_alarm_rate` = {summary['target_far']:.2f}.",
          "",
          "**Report** (direction-only; no formal verdict per Charter):",
          f"- Overall mean known_false_alarm_rate across "
          f"{len(MODELS) * len(CLASSES) * len(SEEDS)} runs: "
          f"{summary['overall_mean_far']:.4f}",
          "",
          "## Per (model × class)",
          "",
          "| Model | Class | " + " | ".join(f"far_s{s}" for s in SEEDS)
          + " | mean | dev vs 0.05 | CI 95% |",
          "|---|---|" + "---|" * len(SEEDS) + "---|---|---|"]
    for _, r in df.iterrows():
        far_cells = " | ".join(f"{r[f'far_s{s}']:.4f}" for s in SEEDS)
        md.append(f"| {r['model']} | {r['class']} | "
                  f"{far_cells} | "
                  f"{r['mean_far']:.4f} | {r['dev_from_target_0.05']:+.4f} | "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")
    (VERDICT_DIR / "ESA1_calibration.md").write_text("\n".join(md), encoding="utf-8")


def write_summary(h1s, h2s, h3s, h4s, esa1s):
    lines = [
        "# Paper 4 H1-H4 Verdict Summary",
        "",
        "Generated by `scripts/paper4_h8_verdict.py`.",
        "Charter: `documents_v2/paper4/PAPER4_CHARTER.md` (LOCKED 2026-10-14 edit pass 1).",
        "",
        "## H1 — Safety view (TRACE-MoE caught_any ≥ 0.95)",
        "",
        f"**Overall**: `{h1s['overall_verdict']}`",
        f"- Per-class breakdown: {h1s['per_class_breakdown']}",
        "",
        "## H2 — Energy abstention vs MSP AUROC",
        "",
        f"**Primary question (TRACE-MoE)**: `{h2s['primary_answer']}`",
        "",
        "Per-model overall:",
    ]
    for model, s in h2s["per_model_breakdown"].items():
        lines.append(f"- `{model}`: **{s['verdict']}** "
                     f"(SUP {s['SUPPORTED']}, INC {s['INCONCLUSIVE']}, NOT {s['NOT SUPPORTED']})")
    lines += ["",
              "## H3 — TRACE-MoE vs baselines (energy AUROC)",
              "",
              "Per-baseline overall:"]
    for b, s in h3s["per_baseline_breakdown"].items():
        lines.append(f"- vs `{b}`: **{s['verdict']}** "
                     f"(SUP {s['SUPPORTED']}, INC {s['INCONCLUSIVE']}, NOT {s['NOT SUPPORTED']})")
    lines += ["",
              "## H4 — Closed-set degradation ≤ 15 pt",
              "",
              f"**Overall**: `{h4s['overall_verdict']}`",
              f"- Breakdown: {h4s['breakdown']}",
              "",
              "## ESA1 — Calibration quality",
              "",
              f"- Target: 0.05 false-alarm rate (95% coverage)",
              f"- Observed overall mean FAR: {esa1s['overall_mean_far']:.4f}",
              "",
              "## ESA2 — DEFERRED",
              "",
              "See `ESA2_DEFERRED.md`. Resolution requires Option A (re-run with energy export) ",
              "or Option B (post-hoc energy recompute from model.pt). User decision pending.",
              "",
              "## INCONCLUSIVE flags (any pair where CI straddles the verdict margin)",
              ""]

    inc_flags = []
    for _, r in h1s.get("_df", pd.DataFrame()).iterrows() if False else []:
        pass
    # Flag inconclusive pairs across H1-H4
    # (we'll inject these at run time below)
    return "\n".join(lines)


def collect_inconclusive(h1_df, h2_df, h3_df, h4_df):
    """Return list of (hypothesis, pair label, reason) for every inconclusive pair."""
    out = []
    for _, r in h1_df[h1_df["verdict"] == "INCONCLUSIVE"].iterrows():
        out.append(("H1", f"TRACE-MoE × {r['class']}",
                    f"mean={r['mean']:.4f}, CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] straddles 0.95"))
    for _, r in h2_df[h2_df["verdict"] == "INCONCLUSIVE"].iterrows():
        out.append(("H2", f"{r['model']} × {r['class']}",
                    f"mean_diff={r['mean_diff']:+.4f}, CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] includes 0"))
    for _, r in h3_df[h3_df["verdict"] == "INCONCLUSIVE"].iterrows():
        out.append(("H3", f"TRACE-MoE vs {r['baseline']} × {r['class']}",
                    f"mean_diff={r['mean_diff']:+.4f}, CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] includes 0"))
    for _, r in h4_df[h4_df["verdict"] == "INCONCLUSIVE"].iterrows():
        out.append(("H4", f"{r['model']} × {r['class']}",
                    f"mean_degr={r['mean_degr']:+.4f}, CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] straddles 0.10 (v1.1 threshold)"))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

CHARTER_PATH = ROOT / "documents_v2" / "paper4_DT" / "PAPER4_CHARTER.md"
CHARTER_V1_1_SHA256 = "e1d29fc2b3278515579599bd5895f62e1706b173169db46b5978d80c43f6caec"  # recipe lock
CHARTER_V1_4_SHA256 = "9c476f63c8e996f585c689f53b8f2467638225a025ab3550375571bb85bd74f4"  # latest Charter lock (v1.4, locked 2026-11-18; v1.3 was 130767067f36...a42)
V1_0_ARCHIVE = ROOT / "results_v3_paper3" / "h8_verdict_v1_0_superseded"
V1_1_3SEED_ARCHIVE = ROOT / "results_v3_paper3" / "h8_verdict_v1_1_3seed_superseded"


def verify_charter_sha(expected: str) -> dict:
    """Recompute Charter SHA256 under the placeholder-protocol and verify.

    Reads the Charter as raw bytes (CRLF preserved) to match the exact byte
    sequence that was hashed at lock time. text-mode read with newline
    translation would alter byte counts on Windows and break verification.
    """
    import hashlib
    raw = CHARTER_PATH.read_bytes()
    placeholder = b"SHA256: PENDING_INSERTION"
    real_line = f"SHA256: {expected}".encode("utf-8")
    if real_line not in raw:
        return {"match": False, "reason": "expected lock-receipt line not found"}
    content_placeholder = raw.replace(real_line, placeholder, 1)
    actual = hashlib.sha256(content_placeholder).hexdigest()
    return {"match": actual == expected, "actual": actual, "expected": expected}


def load_v1_0_verdicts() -> dict:
    """Load v1.0 verdicts from the archive to detect tier flips."""
    return _load_archive_verdicts(V1_0_ARCHIVE, tag="v1.0")


def load_v1_1_3seed_verdicts() -> dict:
    """Load v1.1 3-seed verdicts from the archive to detect seed-extension tier flips."""
    return _load_archive_verdicts(V1_1_3SEED_ARCHIVE, tag="v1.1 3-seed")


def _load_archive_verdicts(archive_dir: Path, tag: str) -> dict:
    out = {}
    try:
        for h_name, fname in [
            ("H1", "H1_safety_view.csv"),
            ("H2", "H2_energy_vs_msp.csv"),
            ("H3", "H3_trace_moe_vs_baselines.csv"),
            ("H4", "H4_closed_set_degradation.csv"),
        ]:
            path = archive_dir / fname
            if path.exists():
                out[h_name] = pd.read_csv(path)
    except Exception as e:
        print(f"  WARNING: {tag} archive load failed: {e}")
    return out


def detect_surprises(v1_0: dict, v1_1_h1, v1_1_h2, v1_1_h3, v1_1_h4) -> list[dict]:
    """Compare v1.0 vs v1.1 per-pair verdicts; return list of flips."""
    surprises = []

    def compare(h_name, old_df, new_df, key_cols: list[str], verdict_col: str = "verdict"):
        for _, new_row in new_df.iterrows():
            match = old_df
            for k in key_cols:
                match = match[match[k] == new_row[k]]
            if len(match) == 1:
                old_v = match.iloc[0][verdict_col]
                new_v = new_row[verdict_col]
                if old_v != new_v:
                    surprises.append({
                        "hypothesis": h_name,
                        "pair": " × ".join(str(new_row[k]) for k in key_cols),
                        "v1_0": old_v,
                        "v1_1": new_v,
                    })

    if "H1" in v1_0:
        compare("H1", v1_0["H1"], v1_1_h1, ["class"])
    if "H2" in v1_0:
        compare("H2", v1_0["H2"], v1_1_h2, ["model", "class"])
    if "H3" in v1_0:
        compare("H3", v1_0["H3"], v1_1_h3, ["baseline", "class"])
    if "H4" in v1_0:
        compare("H4", v1_0["H4"], v1_1_h4, ["model", "class"])

    return surprises


def main() -> int:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Charter SHA verification — latest Charter lock (v1.4) is what pre-registers
    # both the 5-seed scope (via v1.2) and the H5 twin pipeline (v1.3).
    print("Verifying Charter v1.4 SHA256 (latest lock; recipe unchanged from v1.1)...")
    sha = verify_charter_sha(CHARTER_V1_4_SHA256)
    if not sha["match"]:
        print(f"  CHARTER SHA MISMATCH — expected {sha.get('expected')}, got {sha.get('actual')}")
        print(f"  Reason: {sha.get('reason', 'hash recompute does not match recorded')}")
        print("  REFUSING to run aggregation under a tampered Charter.")
        return 2
    print(f"  Charter v1.4 SHA256 VERIFIED: {CHARTER_V1_4_SHA256}")
    print()

    expected_rows = len(MODELS) * len(CLASSES) * len(SEEDS)
    print(f"Loading Phase 3(c) metrics from {PHASE3C_DIR}")
    df = load_phase3c_metrics()
    print(f"  Loaded {len(df)} rows (expected {expected_rows})")

    print(f"Loading full-model baseline from {REF_DIR}  (expected rows: {expected_rows})")
    baseline_df = load_reference_baseline_18class()
    print(f"  Loaded {len(baseline_df)} rows (expected {expected_rows})")

    # Save the merged long table
    (VERDICT_DIR / "per_seed_tables.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"  Wrote per_seed_tables.csv")

    # ---- Compute hypotheses ----
    print("Computing H1 (safety view)...")
    h1_df, h1_summary = compute_h1(df)
    write_h1(h1_df, h1_summary)
    print(f"  H1 overall: {h1_summary['overall_verdict']}")

    print("Computing H2 (energy vs MSP)...")
    h2_df, h2_summary = compute_h2(df)
    write_h2(h2_df, h2_summary)
    print(f"  H2 primary (trace_moe): {h2_summary['primary_answer']}")

    print("Computing H3 (TRACE-MoE vs baselines)...")
    h3_df, h3_summary = compute_h3(df)
    write_h3(h3_df, h3_summary)
    for b, s in h3_summary["per_baseline_breakdown"].items():
        print(f"  H3 vs {b}: {s['verdict']}")

    print("Computing H4 (closed-set degradation)...")
    h4_df, h4_summary = compute_h4(df, baseline_df)
    write_h4(h4_df, h4_summary)
    print(f"  H4 overall: {h4_summary['overall_verdict']}")

    print("Computing ESA1 (calibration)...")
    esa1_df, esa1_summary = compute_esa1(df)
    write_esa1(esa1_df, esa1_summary)
    print(f"  ESA1 overall mean FAR: {esa1_summary['overall_mean_far']:.4f}")

    # ESA2 deferred note
    (VERDICT_DIR / "ESA2_DEFERRED.md").write_text(ESA2_DEFERRED_NOTE, encoding="utf-8")
    print("  ESA2: DEFERRED (per-sample energy not saved; see ESA2_DEFERRED.md)")

    # ---- Inconclusive flags ----
    inconclusive = collect_inconclusive(h1_df, h2_df, h3_df, h4_df)

    # ---- Load v1.0 verdicts for surprise detection ----
    print("Loading v1.0 verdicts for surprise comparison...")
    v1_0 = load_v1_0_verdicts()
    surprises = detect_surprises(v1_0, h1_df, h2_df, h3_df, h4_df)
    print(f"  Detected {len(surprises)} tier flips vs v1.0")

    # ---- SURPRISES_v1_0_to_v1_1.md ----
    surprise_lines = [
        "# Surprises — v1.0 → v1.1 verdict tier flips",
        "",
        f"Charter v1.1 SHA256: `{CHARTER_V1_1_SHA256}`",
        "",
        "This file lists per-pair verdicts that CHANGED between v1.0 (3-tier H2/H3, "
        "15-pt H4 threshold) and v1.1 (4-tier H2/H3 with 0.10 CI-width margin, "
        "10-pt H4 threshold). The underlying data (per_seed_tables.csv) is identical "
        "between v1.0 and v1.1; only the verdict RULES changed.",
        "",
    ]
    if not surprises:
        surprise_lines.append("**No flips detected.** All per-pair verdicts are identical between v1.0 and v1.1.")
    else:
        surprise_lines.append(f"**{len(surprises)} per-pair tier flips** detected:")
        surprise_lines.append("")
        surprise_lines.append("| Hypothesis | Pair | v1.0 verdict | v1.1 verdict |")
        surprise_lines.append("|---|---|---|---|")
        for s in surprises:
            surprise_lines.append(f"| {s['hypothesis']} | {s['pair']} | {s['v1_0']} | **{s['v1_1']}** |")
    (VERDICT_DIR / "SURPRISES_v1_0_to_v1_1.md").write_text(
        "\n".join(surprise_lines), encoding="utf-8")
    print(f"Wrote {VERDICT_DIR / 'SURPRISES_v1_0_to_v1_1.md'}")

    # ---- SURPRISES_v1_1_3seed_to_v1_3_5seed.md ----
    # Compare 3-seed v1.1 verdicts vs 5-seed v1.3 verdicts (same statistical
    # recipe, extended seed scope per Charter v1.2).
    print("Loading v1.1 3-seed verdicts for seed-extension surprise comparison...")
    v1_1_3seed = load_v1_1_3seed_verdicts()
    seed_surprises = detect_surprises(v1_1_3seed, h1_df, h2_df, h3_df, h4_df)
    print(f"  Detected {len(seed_surprises)} tier flips vs v1.1 3-seed")
    seed_surprise_lines = [
        "# Surprises - v1.1 (3 seeds) -> v1.3 (5 seeds) verdict tier flips",
        "",
        f"Charter v1.4 SHA256: `{CHARTER_V1_4_SHA256}`",
        f"Charter v1.2 (seed-extension pre-reg) SHA256: `ba703e2524e4da907ee04ca66dc457114030f39d3503f12f2868a207430444a2`",
        "",
        "This file lists per-pair verdicts that CHANGED between v1.1 (3 seeds: 42, 43, 44) "
        "and v1.3 (5 seeds: 42-46). The statistical verdict RULES are identical (v1.1 recipe); "
        "only the seed scope was extended per Charter v1.2 pre-registration.",
        "",
    ]
    if not seed_surprises:
        seed_surprise_lines.append("**No flips detected.** All per-pair verdicts survived the seed extension unchanged.")
    else:
        seed_surprise_lines.append(f"**{len(seed_surprises)} per-pair tier flips** detected:")
        seed_surprise_lines.append("")
        seed_surprise_lines.append("| Hypothesis | Pair | v1.1 (3-seed) | v1.3 (5-seed) |")
        seed_surprise_lines.append("|---|---|---|---|")
        for s in seed_surprises:
            seed_surprise_lines.append(f"| {s['hypothesis']} | {s['pair']} | {s['v1_0']} | **{s['v1_1']}** |")
    (VERDICT_DIR / "SURPRISES_v1_1_3seed_to_v1_3_5seed.md").write_text(
        "\n".join(seed_surprise_lines), encoding="utf-8")
    print(f"Wrote {VERDICT_DIR / 'SURPRISES_v1_1_3seed_to_v1_3_5seed.md'}")

    # ---- VERDICT_SUMMARY_v1_3_5seed.md ----
    summary_lines = [
        "# Paper 4 H1-H4 Verdict Summary — v1.4 (5-seed scope, 50-epoch protocol)",
        "",
        f"**Latest Charter SHA256 (v1.4)**: `{CHARTER_V1_4_SHA256}`",
        f"**Statistical recipe** (unchanged since v1.1): SHA256 `{CHARTER_V1_1_SHA256}`",
        f"**Pre-registration**: `documents_v2/paper 4_DT/PAPER4_CHARTER.md` "
        f"(LOCKED 2026-11-18 as v1.4; v1.3 added the H5 twin pre-reg, v1.2 pre-registered the 5-seed extension).",
        f"**Scope**: {len(SEEDS)} seeds {SEEDS} x {len(MODELS)} models x {len(CLASSES)} classes "
        f"= {len(MODELS) * len(CLASSES) * len(SEEDS)} LOAO runs.",
        "**v1.0 output**: archived at `results_v3_paper3/h8_verdict_v1_0_superseded/` - do NOT cite.",
        "**v1.1 3-seed output**: archived at `results_v3_paper3/h8_verdict_v1_1_3seed_superseded/` - frozen reference.",
        f"**Surprises vs v1.0**: {len(surprises)} per-pair tier flips (see `SURPRISES_v1_0_to_v1_1.md`).",
        f"**Surprises v1.1 3-seed vs v1.3 5-seed**: {len(seed_surprises)} per-pair tier flips "
        f"(see `SURPRISES_v1_1_3seed_to_v1_3_5seed.md`).",
        "",
        "---",
        "",
        "## H1 — Safety view (TRACE-MoE `caught_any` ≥ 0.95)",
        "",
        f"**Overall**: **`{h1_summary['overall_verdict']}`**",
        f"- Per-class: {h1_summary['per_class_breakdown']}",
        "",
        "## H2 — Energy AUROC > MSP AUROC",
        "",
        f"**Primary (TRACE-MoE)**: **`{h2_summary['primary_answer']}`**",
        "",
        "Per-model breakdown:",
    ]
    for model, s in h2_summary["per_model_breakdown"].items():
        summary_lines.append(f"- `{model}`: **{s['verdict']}** "
                             f"(SUPPORTED {s['SUPPORTED']}, INCONCLUSIVE {s['INCONCLUSIVE']}, "
                             f"NOT {s['NOT SUPPORTED']})")
    summary_lines += ["",
                      "## H3 — TRACE-MoE energy AUROC > each baseline",
                      "",
                      "Per-baseline breakdown:"]
    for b, s in h3_summary["per_baseline_breakdown"].items():
        summary_lines.append(f"- vs `{b}`: **{s['verdict']}** "
                             f"(SUPPORTED {s['SUPPORTED']}, INCONCLUSIVE {s['INCONCLUSIVE']}, "
                             f"NOT {s['NOT SUPPORTED']})")
    summary_lines += ["",
                      "## H4 — Closed-set degradation ≤ 10 pt (v1.1 tightened from 15 pt)",
                      "",
                      f"**Overall**: **`{h4_summary['overall_verdict']}`**",
                      f"- Breakdown: {h4_summary['breakdown']}",
                      "",
                      "## ESA1 — Calibration quality",
                      "",
                      f"- Target false-alarm rate (95% coverage): 0.0500",
                      f"- Observed overall mean FAR across 45 runs: "
                      f"**{esa1_summary['overall_mean_far']:.4f}**",
                      "",
                      "## ESA2 — DEFERRED",
                      "",
                      "Per-sample energy not saved in `predictions.npz`. "
                      "See `ESA2_DEFERRED.md` for 4 resolution options "
                      "(Option B recommended: post-hoc energy recompute from `model.pt`).",
                      "",
                      "---",
                      "",
                      "## INCONCLUSIVE pairs (any pair where CI straddles the verdict margin)",
                      ""]
    if not inconclusive:
        summary_lines.append("- **None** — all pairs have clear verdicts at 3-seed scope.")
    else:
        summary_lines.append(f"- Total: {len(inconclusive)} pairs")
        summary_lines.append("")
        summary_lines.append("| Hypothesis | Pair | Reason |")
        summary_lines.append("|---|---|---|")
        for h, pair, reason in inconclusive:
            summary_lines.append(f"| {h} | {pair} | {reason} |")
    summary_lines += ["",
                      "---",
                      "",
                      "## Files written",
                      "",
                      f"- `per_seed_tables.csv` — raw {len(MODELS) * len(CLASSES) * len(SEEDS)}-row long table ({len(SEEDS)} seeds × {len(MODELS)} models × {len(CLASSES)} classes)",
                      "- `H1_safety_view.{csv,md}`",
                      "- `H2_energy_vs_msp.{csv,md}`",
                      "- `H3_trace_moe_vs_baselines.{csv,md}`",
                      "- `H4_closed_set_degradation.{csv,md}`",
                      "- `ESA1_calibration.{csv,md}`",
                      "- `ESA2_DEFERRED.md`",
                      "- `SURPRISES_v1_0_to_v1_1.md` - tier flips introduced by Charter v1.1",
                      "- `SURPRISES_v1_1_3seed_to_v1_3_5seed.md` - tier flips from seed extension per Charter v1.2",
                      "- `VERDICT_SUMMARY_v1_3_5seed.md` (this file)",
                      "- `verdict.json` (machine-readable summary)",
                      ""]
    (VERDICT_DIR / "VERDICT_SUMMARY_v1_3_5seed.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {VERDICT_DIR / 'VERDICT_SUMMARY_v1_3_5seed.md'}")

    # ---- verdict.json ----
    import datetime
    verdict_json = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "charter_version": "v1.4 (SHA-verified); statistical recipe v1.1; seed scope extended per v1.2",
        "charter_sha256": CHARTER_V1_4_SHA256,
        "charter_locked_at": "2026-11-18 (v1.4 lock)",
        "v1_0_archived_at": str(V1_0_ARCHIVE.relative_to(ROOT).as_posix()),
        "v1_1_3seed_archived_at": str(V1_1_3SEED_ARCHIVE.relative_to(ROOT).as_posix()),
        "bootstrap_resamples": BOOT_N,
        "bootstrap_rng_seed": RNG_SEED,
        "scope": {"seeds": SEEDS, "models": MODELS, "classes": CLASSES,
                  "total_runs": len(MODELS) * len(CLASSES) * len(SEEDS)},
        "H1": h1_summary,
        "H2": h2_summary,
        "H3": h3_summary,
        "H4": h4_summary,
        "ESA1": esa1_summary,
        "ESA2": {"status": "DEFERRED",
                 "blocker": "per-sample energy not in predictions.npz",
                 "recommended_resolution": "Option B — post-hoc energy recompute from model.pt"},
        "inconclusive_pairs": [
            {"hypothesis": h, "pair": p, "reason": r} for h, p, r in inconclusive
        ],
        "surprises_v1_0_to_v1_1": surprises,
        "surprises_v1_1_3seed_to_v1_3_5seed": seed_surprises,
    }
    (VERDICT_DIR / "verdict.json").write_text(
        json.dumps(verdict_json, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {VERDICT_DIR / 'verdict.json'}")

    print()
    print("=" * 70)
    print("DONE — artefacts at", VERDICT_DIR)
    print("=" * 70)
    print(f"H1 overall: {h1_summary['overall_verdict']}")
    print(f"H2 primary (trace_moe): {h2_summary['primary_answer']}")
    for b, s in h3_summary["per_baseline_breakdown"].items():
        print(f"H3 vs {b}: {s['verdict']}")
    print(f"H4 overall: {h4_summary['overall_verdict']}")
    print(f"ESA1 mean FAR: {esa1_summary['overall_mean_far']:.4f}")
    print(f"INCONCLUSIVE pairs flagged: {len(inconclusive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
