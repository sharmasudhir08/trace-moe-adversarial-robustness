"""Paper 4 saved-artifact export (reviewer items, exploratory). Inference only: no training, no result file modified.

For each of the 75 canonical LOAO runs (3 models x 5 held-out classes x seeds 42-46), loads the saved model.pt, rebuilds
the run's windows with the project's own pipeline, and writes per-window scores to
  results_v3_paper3/score_export/<run>.npz
containing: y19 (true fine label), energy (float32, T=1), maxlogit (float32), msp (float32), p_attack (float32, binary head),
logits19 (float16), val_energy (float32), hid (held-out class index), thr (the run's calibrated energy threshold, recomputed
exactly as in scripts/07_unknown_attacks.py lines 62-64), and a consistency record against the run's metrics.json.

Resumable: a run with an existing, complete npz is skipped. Run with the laptop idle (no training lane, no benchmark).
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_iomt.config import load_config  # noqa: E402
from trace_iomt.experiment import load_prepared  # noqa: E402
from trace_iomt.data.windows import build_windowed  # noqa: E402
from trace_iomt.models.factory import build_model  # noqa: E402
from trace_iomt.models.heads import energy_score  # noqa: E402
from trace_iomt.train import _to_tensors  # noqa: E402

MODELS = ["trace_moe", "transformer", "bilstm_dnn"]
CLASSES = ["ARP_Spoofing", "Recon-VulScan", "MQTT-Malformed_Data", "TCP_IP-DoS-UDP", "MQTT-DDoS-Publish_Flood"]
SEEDS = [42, 43, 44, 45, 46]
ZD = ROOT / "results_v3_paper3" / "zero_day"
OUT = ROOT / "results_v3_paper3" / "score_export"


@torch.no_grad()
def scores(model, wd, starts, device, amp, bs=1024):
    model.eval()
    use_amp = amp and device.type == "cuda"
    en, ml, msp, pa, lg = [], [], [], [], []
    for i in range(0, len(starts), bs):
        x, p, _ = _to_tensors(wd, starts[i:i + bs], device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _h = model(x, p)
        pr = model.heads.probabilities(logits)
        l19 = logits["label19"].float()
        en.append(energy_score(logits["label19"]).cpu().numpy())
        ml.append(l19.max(-1).values.cpu().numpy())
        msp.append(pr["label19"].max(-1).values.float().cpu().numpy())
        pa.append(pr["label2"][:, 1].float().cpu().numpy())
        lg.append(l19.cpu().numpy().astype(np.float16))
    cat = lambda v, dt: np.concatenate(v).astype(dt) if v else np.zeros(0, dt)
    return dict(energy=cat(en, np.float32), maxlogit=cat(ml, np.float32), msp=cat(msp, np.float32),
                p_attack=cat(pa, np.float32), logits19=np.concatenate(lg) if lg else np.zeros((0, 19), np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "project rule: GPU only"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    device = torch.device("cuda")
    amp = bool(cfg["train"]["amp"])
    T = int(cfg["windows"]["length"])
    cov = cfg["abstain"]["target_coverage"]
    df = load_prepared(cfg)
    log = open(ROOT / "logs_v3_paper3" / "score_export_20261118.log", "a", encoding="utf-8")
    def say(s):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + s
        print(line, flush=True); log.write(line + "\n"); log.flush()
    for held in a.classes:
        safe = held.replace("/", "_")
        for seed in a.seeds:
            todo = [m for m in a.models if not (OUT / f"{m}_loao_T16_{safe}_seed{seed}.npz").exists()]
            if not todo:
                continue
            c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held}}
            wd = build_windowed(df, c2, regime="loao", T=T, seed=seed)
            hid = wd.hierarchy.index["label19"][held]
            y19 = wd.labels("test", "label19")
            for m in todo:
                run = f"{m}_loao_T16_{safe}_seed{seed}"
                rd = ZD / run
                mj = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
                model = build_model(m, len(wd.features), wd.hierarchy, cfg["model"]).to(device)
                model.load_state_dict(torch.load(rd / "model.pt", map_location="cpu", weights_only=True))
                t0 = time.time()
                te = scores(model, wd, wd.starts["test"], device, amp)
                va = scores(model, wd, wd.starts["val"], device, amp) if len(wd.starts["val"]) else None
                e_val = va["energy"] if va is not None else te["energy"][y19 != hid]
                thr = float(np.quantile(e_val, 1 - cov))
                unk = y19 == hid
                auc = float(roc_auc_score(unk.astype(int), -te["energy"]))
                cons = dict(auroc_export=auc, auroc_metrics=mj["ood"]["energy"]["auroc"], thr_export=thr,
                            thr_metrics=mj["ood"]["reject_threshold_energy"],
                            caught_export=float(((te["energy"] < thr)[unk] | (te["p_attack"][unk] > 0.5)).mean()),
                            caught_metrics=mj["ood"]["held_out_caught_any"], code_sha=mj["code_sha256"][:8], epochs=mj["epochs_run"])
                np.savez_compressed(OUT / f"{run}.npz", y19=y19.astype(np.int8), hid=hid, thr=thr, val_energy=e_val.astype(np.float32),
                                    consistency=json.dumps(cons), **te)
                say(f"{run}: n_test={len(y19):,} auroc export {auc:.4f} vs metrics {cons['auroc_metrics']:.4f}; "
                    f"caught {cons['caught_export']:.4f} vs {cons['caught_metrics']:.4f}; {time.time() - t0:.0f}s")
                del model
                torch.cuda.empty_cache()
            del wd
            gc.collect()
    say("export finished")


if __name__ == "__main__":
    main()
