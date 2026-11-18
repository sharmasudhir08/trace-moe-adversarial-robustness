"""E6 — Transfer attack matrix at multiple epsilon values.

Extends Table 9 (transfer matrix) from a single epsilon=0.25 to
epsilon in {0.05, 0.1, 0.25, 0.5} across all 5 seeds.

Usage:
    python scripts/e6_transfer_multi_eps.py --config configs/paper2_robustness.yaml --regime official
"""
import json, sys, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from _common import setup
from trace_iomt.attacks import pgd, constraint_tensors, attack_eval_starts
from trace_iomt.experiment import load_or_train_deep, load_prepared, load_or_fit_classical, classical_proba
from trace_iomt.train import iterate
from trace_iomt.utils import resolve_device

SEEDS       = [42, 43, 44, 45, 46]
EPS_LIST    = [0.05, 0.1, 0.25, 0.5]
DEEP_MODELS = ["trace_moe", "bilstm_dnn", "transformer", "tfsage_lite"]


def add_args(p):
    p.add_argument("--regime", default="official")


def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _predict_batch(model, x, p):
    with torch.no_grad():
        logits, _ = model(x, p)
    return {k: torch.softmax(v.float(), -1) for k, v in logits.items()}


def transfer_matrix_multi_eps(models, trees, wd, cfg, device, starts, eps_list, logger):
    """Like trace_iomt.attacks.transfer_matrix but runs at multiple eps values."""
    from trace_iomt.experiment import fine_to_coarse
    rc = cfg["robustness"]
    mask, lower, upper = constraint_tensors(wd, rc["mutable_features"], device)
    benign19 = torch.tensor(
        [wd.hierarchy.parent6[wd.hierarchy.parent19[i]] == 0
         for i in range(wd.hierarchy.n("label19"))], device=device)

    results_by_eps = {}
    for eps in eps_list:
        rows = []
        for sname, smodel in models.items():
            smodel.eval()
            flagged = {t: 0 for t in list(models) + list(trees)}
            n_att = 0
            for x, p, y in iterate(wd, starts, 512, device):
                att = ~benign19[y["label19"]]
                if not att.any():
                    continue
                xa = pgd(smodel, x[att], p[att], y["label19"][att], eps,
                         rc["pgd_steps"], rc["pgd_step_frac"], mask, lower, upper,
                         rand_init=True, momentum=rc.get("momentum", 0.0))
                pa = p[att]
                n_att += int(att.sum())
                for tname, tmodel in models.items():
                    tmodel.eval()
                    pr = _predict_batch(tmodel, xa, pa)
                    flagged[tname] += int((pr["label2"].argmax(1) == 1).sum())
                if trees:
                    last = xa[:, -1].cpu().numpy()
                    for tname, fn in trees.items():
                        p2 = fine_to_coarse(fn(last), wd.hierarchy)["label2"]
                        flagged[tname] += int((p2.argmax(1) == 1).sum())
            for tname, k in flagged.items():
                asr = 1 - k / max(n_att, 1)
                rows.append({"eps": eps, "surrogate": sname, "target": tname,
                             "n_attack_windows": n_att, "asr": asr,
                             "white_box": tname == sname})
            logger.info(f"eps={eps} surrogate={sname}: " +
                        ", ".join(f"{r['target']} ASR {r['asr']:.3f}"
                                  for r in rows if r["surrogate"] == sname and r["eps"] == eps))
        results_by_eps[str(eps)] = rows
    return results_by_eps


def main():
    args, cfg, log = setup("E6 transfer multi-eps", "e6_transfer_multi_eps", add_args)
    df = load_prepared(cfg)
    device = resolve_device(cfg["train"]["device"])
    all_results = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        set_seed(seed)

        models, wd = {}, None
        for mname in DEEP_MODELS:
            try:
                m, wd = load_or_train_deep(df, cfg, mname, args.regime, seed, "main", log,
                                           strict=True, wd=wd)
                m.eval()
                models[mname] = m
            except Exception as e:
                print(f"  SKIP {mname}: {e}", flush=True)

        if wd is None:
            print(f"  No models loaded for seed {seed}, skipping.", flush=True)
            continue

        trees = {}
        try:
            clf, _, _ = load_or_fit_classical("rf", wd, cfg, seed, args.regime, "main", log)
            trees["rf"] = lambda X, c=clf: classical_proba(c, X, wd.hierarchy.n("label19"))
        except Exception:
            pass

        starts = attack_eval_starts(wd, cfg["robustness"]["max_eval_windows"], seed, 0)
        result = transfer_matrix_multi_eps(models, trees, wd, cfg, device, starts, EPS_LIST, log)
        all_results[str(seed)] = result

    out_path = pathlib.Path(cfg["paths"]["results_dir"]) / "e6_transfer_multi_eps" / "transfer_multi_eps.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)

    # Summary: mean ASR across seeds for each surrogate-target pair at each eps
    print("\n=== MEAN ASR across seeds (black-box transfer) ===")
    surrogates = DEEP_MODELS
    targets    = DEEP_MODELS
    for eps in EPS_LIST:
        print(f"\n  eps={eps}")
        print(f"  {'Surrogate \\ Target':22s}  " + "  ".join(f"{t:15s}" for t in targets))
        for s in surrogates:
            row = []
            for t in targets:
                vals = []
                for seed in SEEDS:
                    for r in all_results.get(str(seed), {}).get(str(eps), []):
                        if r["surrogate"] == s and r["target"] == t:
                            vals.append(r["asr"])
                row.append(f"{np.mean(vals):.3f}" if vals else "  nan")
            print(f"  {s:22s}  " + "  ".join(f"{v:15s}" for v in row))


if __name__ == "__main__":
    main()
