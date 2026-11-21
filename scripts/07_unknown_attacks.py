"""Step 7. Unknown attacks and the abstain layer (contribution C3, hypothesis H8).

Leave one attack class out of training, then ask whether its test windows are
  (a) separated from known traffic by the abstain score (energy) and by plain softmax confidence (AUROC, FPR@95),
  (b) flagged "unknown" by the reject threshold fixed on validation (95% of known windows accepted),
  (c) still caught as "Attack" by the binary head.

Results: results/zero_day/<model>_loao_T<T>_<class>_seed<k>/metrics.json with an "ood" block.
"""
import gc
import os

import numpy as np

from _common import setup
from trace_iomt.experiment import build_run_windows, load_prepared, run_deep
from trace_iomt.metrics import classification_metrics, ood_metrics
from trace_iomt.train import predict
from trace_iomt.utils import load_json, resolve_device, save_json


def add(p):
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)


def main():
    args, cfg, log = setup("Zero-day (leave one attack out)", "07_unknown_attacks", add)
    zc = cfg["zero_day"]
    models = args.models or zc["models"]
    classes = args.classes or zc["classes"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    df = load_prepared(cfg)
    known = set(df["label19"].astype(str).unique())
    device = resolve_device(cfg["train"]["device"])
    # Opt-in memory saver (P4_RELEASE_TABLE=1): a process that runs exactly one (class, seed) builds its windows,
    # then frees the full table (about 1.7 GB) before training. Results are unchanged.
    release_table = (os.environ.get("P4_RELEASE_TABLE") == "1" and len(seeds) == 1
                     and len([h for h in classes if h in known]) == 1)
    for held in classes:
        if held not in known:
            log.warning(f"{held} is not a class in the data; skipping")
            continue
        c = {**cfg, "split": {**cfg["split"], "loao_class": held}}
        for seed in seeds:
            wd = None
            if release_table:
                holder = [df]
                df = None            # the holder is now the only reference; build_windowed frees it mid-build
                wd = build_run_windows(holder, c, "loao", seed, None, log)
                gc.collect()
                log.info("Released the full table after building the windows (P4_RELEASE_TABLE=1)")
            for mname in models:
                d, meta, model, wd = run_deep(df, c, mname, "loao", seed, "zero_day", log, tag=held, wd=wd)
                st = wd.starts["test"]
                y19 = wd.labels("test", "label19")
                hid = wd.hierarchy.index["label19"][held]
                pr = predict(model, wd, st, device, amp=cfg["train"]["amp"])
                unk = y19 == hid
                cov = cfg["abstain"]["target_coverage"]
                if len(wd.starts["val"]):
                    e_val = predict(model, wd, wd.starts["val"], device, amp=cfg["train"]["amp"])["energy"]
                    thr = float(np.quantile(e_val, 1 - cov))
                else:
                    thr = float(np.quantile(pr["energy"][~unk], 1 - cov))
                flagged = pr["energy"] < thr
                msp = pr["proba"]["label19"].max(1)
                ood = {"held_out_class": held,
                       "energy": ood_metrics(pr["energy"][~unk], pr["energy"][unk]),
                       "msp": ood_metrics(msp[~unk], msp[unk]),
                       "held_out_detected_as_attack": float((pr["proba"]["label2"][unk].argmax(1) == 1).mean())
                       if unk.any() else float("nan"),
                       "reject_threshold_energy": thr, "target_coverage": cov,
                       "held_out_flagged_unknown": float(flagged[unk].mean()) if unk.any() else float("nan"),
                       "known_false_alarm_rate": float(flagged[~unk].mean()) if (~unk).any() else float("nan"),
                       # caught = flagged unknown OR called "Attack": the safety view for a hospital network
                       "held_out_caught_any": float((flagged[unk] | (pr["proba"]["label2"][unk].argmax(1) == 1)).mean())
                       if unk.any() else float("nan")}
                # closed-set metrics on the known classes only
                names = wd.hierarchy.classes["label19"]
                kn = classification_metrics(y19[~unk], pr["proba"]["label19"][~unk].argmax(1), None, names)
                ood["known_classes_macro_f1"] = float(np.mean([v["f1"] for k, v in kn["per_class"].items()
                                                               if v["support"] > 0]))
                m = load_json(d / "metrics.json")
                m["ood"] = ood
                save_json(m, d / "metrics.json")
                log.info(f"[{mname} without {held}] energy AUROC {ood['energy']['auroc']:.4f} | "
                         f"FPR@95 {ood['energy']['fpr_at_95tpr']:.4f} | caught as attack "
                         f"{ood['held_out_detected_as_attack']:.4f} | flagged unknown {ood['held_out_flagged_unknown']:.4f}")


if __name__ == "__main__":
    main()
