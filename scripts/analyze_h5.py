"""H5 verdict and 2D per-variant ablation table (Paper 2).

Applies the pre-registered rule from the paper charter:
  SUPPORTED if dense_expert closes >=50% of the (TRACE-MoE minus Transformer)
  F1@epsilon=0.25 gap AND both seeds agree on direction.

Reads per-variant metrics.json from results/ablation/ and the
full TRACE-MoE and Transformer baselines from results/robustness/.
Writes results to stdout; redirect to capture.
"""
from __future__ import annotations
import json
import pathlib
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent   # repository root
RV   = ROOT / "results" / "ablation"
RB   = ROOT / "results" / "robustness"

VARIANTS = ['dense_expert', 'protocol_routing', 'no_experts', 'no_balance_loss',
            'no_hierarchy', 'plain_ce_loss', 'no_protocol_token']
SEEDS    = [43, 44]
EPS      = [0.0, 0.05, 0.1, 0.25, 0.5]
EPS_STR  = ['0.0', '0.05', '0.1', '0.25', '0.5']


def read_variant(variant, seed):
    best = None; best_n = -1
    for p in RV.glob(f"{variant}_seed{seed}*"):
        mj = p / "metrics.json"
        if mj.exists():
            o = json.load(open(mj, encoding='utf-8'))
            n = o.get('n_attack_windows', 0)
            if n > best_n:
                best_n = n; best = o
    return best


def read_baseline(model, seed):
    p = RB / f"{model}_seed{seed}" / "metrics.json"
    if p.exists():
        return json.load(open(p, encoding='utf-8'))
    return None


def auc_robustness(curve_dict):
    f1s = [curve_dict[e]['label19']['macro_f1'] for e in EPS_STR]
    return float(np.trapezoid(f1s, EPS)) / (EPS[-1] - EPS[0])


def main():
    rows = []; missing = []
    for v in VARIANTS:
        for s in SEEDS:
            m = read_variant(v, s)
            if m is None:
                missing.append(f"{v}_seed{s}"); continue
            cc = m['curve_clean_model']; ca = m['curve_adv_finetuned']
            if sorted(cc.keys()) != EPS_STR or m.get('n_attack_windows', 0) < 1000:
                missing.append(f"{v}_seed{s} (not full depth)"); continue
            rows.append({'variant': v, 'seed': s,
                'n_attack_windows':  m['n_attack_windows'],
                'clean_f1_pre':      cc['0.0']['label19']['macro_f1'],
                'clean_f1_post':     ca['0.0']['label19']['macro_f1'],
                'adv_ft_cost_pt':   (cc['0.0']['label19']['macro_f1'] - ca['0.0']['label19']['macro_f1']) * 100,
                'f1_at_025_post':    ca['0.25']['label19']['macro_f1'],
                'f1_at_050_post':    ca['0.5']['label19']['macro_f1'],
                'auc_robust_post':   auc_robustness(ca),
                'auc_robust_pre':    auc_robustness(cc)})
    if missing:
        print("MISSING/INCOMPLETE RUNS:")
        for m in missing: print(f"  - {m}")
        print()

    tm_025 = {}; tx_025 = {}
    for s in SEEDS:
        tm = read_baseline('trace_moe', s);  tx = read_baseline('transformer', s)
        if tm: tm_025[s] = tm['curve_adv_finetuned']['0.25']['label19']['macro_f1']
        if tx: tx_025[s] = tx['curve_adv_finetuned']['0.25']['label19']['macro_f1']

    print("=== Baselines at eps=0.25 (adv-FT) ===")
    for s in SEEDS:
        gap = tm_025.get(s, float('nan')) - tx_025.get(s, float('nan'))
        print(f"  seed {s}: TRACE-MoE={tm_025.get(s,float('nan')):.4f}  "
              f"Transformer={tx_025.get(s,float('nan')):.4f}  gap={gap:+.4f}")
    print()

    de_results = {r['seed']: r['f1_at_025_post'] for r in rows if r['variant'] == 'dense_expert'}
    per_seed = {}
    for s in SEEDS:
        if s not in de_results or s not in tm_025 or s not in tx_025:
            per_seed[s] = {'ok': False}; continue
        de = de_results[s]; full_moe = tm_025[s]; tx = tx_025[s]
        gap_orig = full_moe - tx; gap_de = de - tx
        closure = abs(gap_orig) - abs(gap_de)
        pct = (closure / abs(gap_orig)) * 100 if abs(gap_orig) > 1e-4 else float('inf')
        per_seed[s] = {'ok': True, 'dense_expert_f1_025': de, 'full_moe_f1_025': full_moe,
            'transformer_f1_025': tx, 'gap_moe_tx': gap_orig, 'gap_de_tx': gap_de,
            'pct_closed': pct, 'closed_50': closure > 0 and pct >= 50,
            'direction': 'closed' if closure > 0 else 'widened'}
        print(f"  seed {s}: dense_expert={de:.4f}  TRACE-MoE={full_moe:.4f}  Transformer={tx:.4f}")
        print(f"           gap(MoE-Tx)={gap_orig:+.4f}  pct_closed={pct:+.1f}%  direction={per_seed[s]['direction']}")
        print()

    both_ok = all(per_seed[s]['ok'] for s in SEEDS)
    if both_ok:
        dirs = [per_seed[s]['direction'] for s in SEEDS]
        agree = len(set(dirs)) == 1
        all50 = all(per_seed[s]['closed_50'] for s in SEEDS)
        if not agree:   verdict = 'INCONCLUSIVE'; reason = f"seeds disagree: {dirs}"
        elif all50:     verdict = 'SUPPORTED';    reason = f"both closed >=50% ({dirs[0]})"
        else:           verdict = 'NOT SUPPORTED'; reason = (
            f"agree direction ({dirs[0]}) but <50%: "
            f"seed43={per_seed[43]['pct_closed']:.1f}%  seed44={per_seed[44]['pct_closed']:.1f}%")
    else:
        verdict = 'ERROR'; reason = 'missing data'

    print(f"=== H5 VERDICT: {verdict} ===")
    print(f"    reason: {reason}"); print()

    print("=== 2D ablation table ===")
    print(f"  {'variant':20s}  {'seed':>4s}  {'adv-FT cost':>12s}  {'AUC-robust':>11s}  {'worst-F1@0.5':>13s}")
    for r in rows:
        print(f"  {r['variant']:20s}  {r['seed']:>4d}  {r['adv_ft_cost_pt']:+8.3f} pt    "
              f"{r['auc_robust_post']:.4f}       {r['f1_at_050_post']:.4f}")


if __name__ == "__main__":
    main()
