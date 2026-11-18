"""E1 — PGD random-restart validation (Strategy B: seed 42 only).

Runs 5-restart PGD-20 on seed 42 for all four deep models and compares
AUC-robustness and worst-case F1 against the single-start results already
in results_v3_paper2/robustness/.

Usage:
    python scripts/e1_pgd_restarts.py --config configs/paper2_robustness.yaml --regime official
"""
import json, sys, time, pathlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from _common import setup
from trace_iomt.attacks import pgd, constraint_tensors, attack_eval_starts
from trace_iomt.experiment import load_or_train_deep, load_prepared, latest_run_dir
from trace_iomt.metrics import evaluate_levels
from trace_iomt.train import iterate
from trace_iomt.utils import resolve_device

N_RESTARTS = 5
SEED       = 42
EPS_LIST   = [0.0, 0.05, 0.1, 0.25, 0.5]
EPS_STR    = ["0.0", "0.05", "0.1", "0.25", "0.5"]


def add_args(p):
    p.add_argument("--regime", default="official")


def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def pgd_multistart(model, x, proto, y19, eps, steps, step_frac, mask, lower, upper, n_restarts, momentum=0.0):
    """Run PGD from n_restarts random starts, return worst-case adversarial examples."""
    if eps <= 0:
        return x
    best_loss = torch.full((len(x),), -1e9, device=x.device)
    best_adv  = x.clone()
    for _ in range(n_restarts):
        adv = pgd(model, x, proto, y19, eps, steps, step_frac, mask, lower, upper,
                  rand_init=True, momentum=momentum)
        with torch.no_grad():
            logits, _ = model(adv, proto)
            loss = F.cross_entropy(logits["label19"].float(), y19, reduction="none")
        improved = loss > best_loss
        best_loss = torch.where(improved, loss, best_loss)
        best_adv[improved] = adv[improved]
    return best_adv


def robustness_curve_multistart(model, wd, cfg, device, starts, n_restarts):
    rc = cfg["robustness"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = {i for i, c in enumerate(wd.hierarchy.classes["label19"])
                if wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0}
    model.eval()
    out = {}
    for eps in EPS_LIST:
        probs_all = {k: [] for k in ("label2", "label6", "label19")}
        ys_all    = {k: [] for k in probs_all}
        for x, p, y in iterate(wd, starts, 512, device):
            is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
            xa = x
            if eps > 0 and is_attack.any():
                xa = x.clone()
                xa[is_attack] = pgd_multistart(
                    model, x[is_attack], p[is_attack], y["label19"][is_attack],
                    eps, rc["pgd_steps"], rc["pgd_step_frac"], mask, lower, upper, n_restarts,
                    momentum=rc.get("momentum", 0.0))
            with torch.no_grad():
                logits, _ = model(xa, p)
            for k in probs_all:
                probs_all[k].append(logits[k].softmax(-1).cpu().numpy())
                ys_all[k].append(y[k].cpu().numpy())
        probs = {k: np.concatenate(v) for k, v in probs_all.items()}
        ys    = {k: np.concatenate(v) for k, v in ys_all.items()}
        m = evaluate_levels(ys, probs, wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
        att = ys["label2"] == 1
        m["attack_detection_rate"] = float((probs["label2"][att].argmax(1) == 1).mean()) if att.any() else float("nan")
        out[str(eps)] = m
        f1 = m["label19"]["macro_f1"]
        print(f"  eps={eps:<5} macro-F1={f1:.4f}", flush=True)
    return out


def auc_rob(curve):
    f1s = [curve[e]["label19"]["macro_f1"] for e in EPS_STR]
    return float(np.trapezoid(f1s, EPS_LIST)) / (EPS_LIST[-1] - EPS_LIST[0])


def main():
    args, cfg, log = setup("E1 PGD restarts", "e1_pgd_restarts", add_args)
    df = load_prepared(cfg)
    device = resolve_device(cfg["train"]["device"])
    T = cfg["windows"]["length"]

    set_seed(SEED)
    models_deep = ["trace_moe", "bilstm_dnn", "transformer", "tfsage_lite"]
    results = {}

    # Step 1: build wd by loading one model, then free it
    wd = None
    for mname in models_deep:
        try:
            first_model, wd = load_or_train_deep(df, cfg, mname, args.regime, SEED, "main", log,
                                                  strict=True)
            first_model.cpu(); del first_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"[{mname}] could not load for wd init: {e}", flush=True)

    if wd is None:
        print("No model loaded; aborting.", flush=True); return

    starts = attack_eval_starts(wd, cfg["robustness"]["max_eval_windows"], SEED, 0)
    print(f"Attack windows: {len(starts):,}  restarts: {N_RESTARTS}  seed: {SEED}", flush=True)

    # Step 2: evaluate each model one at a time
    for mname in models_deep:
        try:
            model, _ = load_or_train_deep(df, cfg, mname, args.regime, SEED, "main", log,
                                          strict=True, wd=wd)
        except Exception as e:
            print(f"[{mname}] SKIP — could not load: {e}", flush=True)
            continue

        model.eval()
        print(f"\n[{mname}] running {N_RESTARTS}-restart PGD-20 ...", flush=True)
        t0 = time.time()
        curve = robustness_curve_multistart(model, wd, cfg, device, starts, N_RESTARTS)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        model.cpu(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Find original single-start result for comparison
        orig_dir = latest_run_dir(cfg, "robustness", f"{mname}_{args.regime}_T{T}_seed{SEED}")
        orig = None
        if orig_dir is not None and (orig_dir / "metrics.json").exists():
            orig = json.loads((orig_dir / "metrics.json").read_text(encoding="utf-8"))

        auc_ms = auc_rob(curve)
        auc_1s = auc_rob(orig["curve_clean_model"]) if orig else float("nan")
        wf_ms  = curve["0.5"]["label19"]["macro_f1"]
        wf_1s  = orig["curve_clean_model"]["0.5"]["label19"]["macro_f1"] if orig else float("nan")
        print(f"  AUC-rob:  single={auc_1s:.4f}  multi={auc_ms:.4f}  Δ={auc_ms-auc_1s:+.4f}", flush=True)
        print(f"  F1@0.5:   single={wf_1s:.4f}  multi={wf_ms:.4f}  Δ={wf_ms-wf_1s:+.4f}", flush=True)
        results[mname] = {
            "n_restarts": N_RESTARTS, "seed": SEED,
            "auc_rob_single": auc_1s, "auc_rob_multi": auc_ms,
            "f1_at_05_single": wf_1s, "f1_at_05_multi": wf_ms,
            "curve_multistart": curve
        }

    out_path = pathlib.Path(cfg["paths"]["results_dir"]) / "e1_pgd_restarts" / f"restarts{N_RESTARTS}_seed{SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)

    print("\n=== SUMMARY ===")
    for m, r in results.items():
        print(f"  {m:20s}  AUC-rob Δ={r['auc_rob_multi']-r['auc_rob_single']:+.4f}"
              f"  F1@0.5 Δ={r['f1_at_05_multi']-r['f1_at_05_single']:+.4f}")


if __name__ == "__main__":
    main()
