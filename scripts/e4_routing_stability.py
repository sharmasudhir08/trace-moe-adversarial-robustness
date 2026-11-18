"""E4 — Routing instability analysis (expert-switching under PGD attack).

Measures the fraction of windows where TRACE-MoE's expert selection changes
when the input is perturbed by PGD. Reports per-epsilon routing instability
rate and correlates with the epsilon=0.25 F1 anomaly.

Usage:
    python scripts/e4_routing_stability.py --config configs/paper2_robustness.yaml --regime official
"""
import json, sys, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from _common import setup
from trace_iomt.attacks import pgd, constraint_tensors, attack_eval_starts
from trace_iomt.experiment import load_or_train_deep, load_prepared
from trace_iomt.train import iterate
from trace_iomt.utils import resolve_device

SEEDS    = [42, 43, 44, 45, 46]
EPS_LIST = [0.0, 0.05, 0.1, 0.25, 0.5]
MODEL    = "trace_moe"


def add_args(p):
    p.add_argument("--regime", default="official")


def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def routing_instability(model, wd, cfg, device, starts, eps_list):
    """Return per-eps routing instability rate (fraction of attack windows that
    switch expert) and expert-selection IoU between clean and adversarial."""
    rc = cfg["robustness"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = {i for i, c in enumerate(wd.hierarchy.classes["label19"])
                if wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0}
    model.eval()
    results = {}

    # --- clean routing (eps=0) ---
    clean_routes = []
    for x, p, y in iterate(wd, starts, 512, device):
        with torch.no_grad():
            _ = model(x, p)
        is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
        clean_routes.append(model.last_route[is_attack].cpu().numpy())
    clean_routes = np.concatenate(clean_routes)   # shape (n_attack_windows,)
    n_att = len(clean_routes)
    n_exp = model.n_experts
    print(f"  clean routing: {n_att:,} attack windows, {n_exp} experts", flush=True)

    # expert share distribution (clean)
    clean_dist = np.bincount(clean_routes, minlength=n_exp) / max(n_att, 1)
    results["clean_expert_dist"] = clean_dist.tolist()
    print(f"  clean expert shares: {[f'{v:.3f}' for v in clean_dist]}", flush=True)

    for eps in eps_list:
        if eps == 0.0:
            results["0.0"] = {"instability_rate": 0.0, "jaccard": 1.0,
                              "adv_expert_dist": clean_dist.tolist()}
            continue

        adv_routes = []
        for x, p, y in iterate(wd, starts, 512, device):
            is_attack = torch.tensor([int(v) not in benign19 for v in y["label19"].tolist()], device=device)
            if not is_attack.any():
                continue
            adv = pgd(model, x[is_attack], p[is_attack], y["label19"][is_attack],
                      eps, rc["pgd_steps"], rc["pgd_step_frac"], mask, lower, upper,
                      rand_init=True, momentum=rc.get("momentum", 0.0))
            with torch.no_grad():
                _ = model(adv, p[is_attack])
            adv_routes.append(model.last_route.cpu().numpy())

        adv_routes = np.concatenate(adv_routes)
        min_len = min(len(clean_routes), len(adv_routes))
        c, a = clean_routes[:min_len], adv_routes[:min_len]

        switched = (c != a).mean()
        agree = (c == a).sum()
        jaccard = float(agree) / max(min_len, 1)

        adv_dist = np.bincount(adv_routes, minlength=n_exp) / max(len(adv_routes), 1)
        results[str(eps)] = {
            "instability_rate": float(switched),
            "jaccard": float(jaccard),
            "adv_expert_dist": adv_dist.tolist()
        }
        print(f"  eps={eps:<5}  instability={switched:.4f}  jaccard={jaccard:.4f}"
              f"  adv_dist={[f'{v:.3f}' for v in adv_dist]}", flush=True)
    return results


def main():
    args, cfg, log = setup("E4 routing instability", "e4_routing_stability", add_args)
    df = load_prepared(cfg)
    device = resolve_device(cfg["train"]["device"])
    all_results = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        set_seed(seed)
        try:
            model, wd = load_or_train_deep(df, cfg, MODEL, args.regime, seed, "main", log, strict=True)
        except Exception as e:
            print(f"  SKIP: {e}", flush=True)
            continue
        model.eval()
        starts = attack_eval_starts(wd, cfg["robustness"]["max_eval_windows"], seed, 0)
        res = routing_instability(model, wd, cfg, device, starts, EPS_LIST)
        all_results[str(seed)] = res

    out_path = pathlib.Path(cfg["paths"]["results_dir"]) / "e4_routing_stability" / "routing_instability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)

    # Summary table
    print("\n=== SUMMARY: instability rate per epsilon ===")
    print(f"  {'eps':>6}  " + "  ".join(f"seed{s}" for s in SEEDS))
    for eps in EPS_LIST:
        row = []
        for s in SEEDS:
            v = all_results.get(str(s), {}).get(str(float(eps)), {}).get("instability_rate", float("nan"))
            row.append(f"{v:.4f}" if not np.isnan(v) else "  nan ")
        print(f"  {str(eps):>6}  " + "  ".join(row))


if __name__ == "__main__":
    main()
