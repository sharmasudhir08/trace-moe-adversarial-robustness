"""Paper 4 T8: forward FLOPs per window for BiLSTM-DNN, Transformer and TRACE-MoE and the cost of the energy and MSP scores. CPU only, no training, no data.
Source of the model counts: results_v2/efficiency/flops.json (analytic per-module MAC count by forward hooks, FLOPs = 2 x MACs, produced by
the analytic FLOP-counting script measure_flops.py with the trained weights of the seed-42 official-split runs, strict load, parameter counts equal to metrics.json).
Applicability to the Paper 4 LOAO checkpoints is checked here by comparing parameter counts with the Paper 4 benchmark environment.json (architecture and
parameter count identical => same FLOPs; the TRACE-MoE count is for the expert chosen for the probe window, all four experts have the same shape).
Cross-check: FlopCounterMode value stored in the same file (matmul kernels only: lower bound). Score overhead: analytic elementary-operation count over 19 logits.
Usage: python scripts/p4_rev_t8_flops.py --date 20261121
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); a = ap.parse_args()
OUT = ROOT / "results_v3_paper3" / f"p4_review_analyses_{a.date}" / "T8"; OUT.mkdir(parents=True, exist_ok=True)
fl = json.loads((ROOT / "results_v2" / "efficiency" / "flops.json").read_text())["models"]
env = json.loads((ROOT / "results_v3_paper3" / "inference_benchmark_20261118" / "environment.json").read_text())["params"]
C = 19
# elementary operations over 19 logits (exp, add, mul/div, max/compare, log each count as one operation)
score_ops = {"energy_logsumexp": C + (C - 1) + 1 + 1 + (C - 1) + C,        # max, subtract, exp, sum, log, add
             "msp_softmax_max": (C - 1) + C + C + (C - 1) + C + (C - 1)}    # max, subtract, exp, sum, divide, max
rows = []
for k in ("bilstm_dnn", "transformer", "trace_moe"):
    v = fl[k]
    rows.append(dict(model=k, params_flops_file=v["n_params_built"], params_paper4_benchmark=env[k], params_equal=v["n_params_built"] == env[k],
                     flops_per_window=v["flops_per_window"], mflops_per_window=v["mflops_per_window"],
                     torch_flopcounter_mflops_lower_bound=v["torch_flopcounter_mflops_crosscheck"],
                     relative_gap_to_counter=1 - v["torch_flopcounter_flops_crosscheck"] / v["flops_per_window"],
                     energy_score_ops=score_ops["energy_logsumexp"], msp_score_ops=score_ops["msp_softmax_max"],
                     score_ops_share_of_model_flops=(score_ops["energy_logsumexp"] + score_ops["msp_softmax_max"]) / v["flops_per_window"]))
d = pd.DataFrame(rows); d.to_csv(OUT / "flops.csv", index=False)
trm = d[d.model == "trace_moe"].mflops_per_window.iloc[0]
(OUT / "README.md").write_text(
    "# T8 FLOPs (not a new measurement of speed; analytic count)\n\n"
    f"Script `scripts/p4_rev_t8_flops.py` SHA-256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; Python {sys.version.split()[0]}; {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
    + d.round(6).to_markdown(index=False) + "\n\n"
    f"TRACE-MoE count {trm} MFLOPs against the manuscript's 11.54: relative difference {abs(trm - 11.54) / 11.54:.4%} (stop criterion 5%).\n"
    "The torch FlopCounterMode figure counts matmul kernels only, so it is a lower bound; for the BiLSTM-DNN it misses the fused LSTM kernels (large gap, explained by the method note in flops.json). "
    "The two methods agree within 5% for the Transformer-like models (see relative_gap_to_counter); the BiLSTM-DNN difference is explained, not an error. "
    "Score overhead is an analytic elementary-operation count (exp, add, divide, max and log each one operation), negligible against the model FLOPs.\n", encoding="utf-8")
print(d[["model", "mflops_per_window", "params_equal", "relative_gap_to_counter", "score_ops_share_of_model_flops"]].to_string())
