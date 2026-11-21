"""Paper 4, task T3 (post-hoc exploratory, approved by the author 2026-11-21): feature-space OOD scores on learned embeddings.

Inference only: no training, no model or result file is modified. For each of the 75 canonical LOAO runs
(3 models x 5 held-out classes x seeds 42-46) the saved model.pt is loaded and the window embedding that enters the
hierarchical heads (second return value of model.forward) is used.

Fixed settings (set before any test result was seen; do NOT change after looking at results):
  * fit data: known-class TRAINING windows only, class-stratified (proportional) subsample of at most 200,000, rng 42
  * Mahalanobis: class-conditional, Ledoit-Wolf covariance per known class, score = minimum distance over known classes
  * kNN: L2-normalised embeddings, k = 50, score = Euclidean distance to the 50th nearest training embedding
  * ReAct: embeddings clipped from above at the 90th percentile of all training activation values (one scalar),
           then the label19 linear head is applied exactly; score = -logsumexp(logits)
  * ViM: principal subspace dimension = d/2, origin u = -pinv(W) b, alpha = sum(max logit) / sum(residual norm) on training
         windows, score = alpha * ||residual|| - logsumexp(logits)
  * all scores: higher = more unknown. AUROC with the held-out class as positive; FPR@95 = FPR at the first TPR >= 0.95.
  * baseline energy recomputed in the same pass and compared with score_export energy (max abs difference recorded).

Test embeddings are not stored (about 30 GB for 75 runs); test scores are computed in the same pass and saved as float32.
Resumable: runs already present in per_run_feature_scores.csv are skipped.
GPU only, AC power only (aborts on battery or without CUDA). Run with no other project job and with at least 6 GB free RAM.
"""
from __future__ import annotations
import argparse, csv, ctypes, gc, hashlib, json, os, shutil, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

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
SE = ROOT / "results_v3_paper3" / "score_export"
MAX_TRAIN = 200_000
K_NN, REACT_PCT, RNG = 50, 90, 42
SCORES = ["energy", "mahalanobis_lw", "knn50", "react", "vim"]
TEMPS = [0.5, 1, 2, 5]   # T2 temperature grid, recomputed here from float32 logits (the saved float16 logits are not accurate enough)
SCORES_ALL = SCORES + [f"energy_T{t}" for t in TEMPS if t != 1] + [f"msp_T{t}" for t in TEMPS]


def on_ac():
    class S(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = S()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return s.ACLineStatus == 1


def strat_subsample(y, n_max, rng):
    n = len(y)
    if n <= n_max:
        return np.arange(n)
    idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        k = min(len(ic), int(np.ceil(n_max * len(ic) / n)))
        idx.append(rng.choice(ic, size=k, replace=False))
    return np.sort(np.concatenate(idx))


@torch.no_grad()
def embed(model, wd, starts, device, amp, bs=1024):
    model.eval()
    use_amp = amp and device.type == "cuda"
    for i in range(0, len(starts), bs):
        x, p, _ = _to_tensors(wd, starts[i:i + bs], device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, h = model(x, p)
        yield logits["label19"].float(), h.float()


def fpr_at_95(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[min(np.searchsorted(tpr, 0.95, side="left"), len(fpr) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "project rule: GPU only"
    assert on_ac(), "T3 needs AC power: plug in the charger and rerun"
    out_emb = ROOT / "results_v3_paper3" / f"embedding_export_{a.date}"
    out_t3 = ROOT / "results_v3_paper3" / f"p4_review_analyses_{a.date}" / "T3"
    out_emb.mkdir(parents=True, exist_ok=True); out_t3.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_emb).free / 1e9
    assert free_gb > 20, f"only {free_gb:.0f} GB free on the drive"
    csv_path = out_t3 / "per_run_feature_scores.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            done = {r["run"] for r in csv.DictReader(f)}
    cfg = load_config(str(ROOT / "configs" / "paper3_openset.yaml"))
    device = torch.device("cuda")
    amp, T = bool(cfg["train"]["amp"]), int(cfg["windows"]["length"])
    df = load_prepared(cfg)
    log = open(out_t3 / "run_log.txt", "a", encoding="utf-8")
    def say(s):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + s
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say(f"start; torch {torch.__version__}; script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    cols = ["run", "model", "held_out_class", "seed", "n_test", "n_unknown", "n_train_fit", "energy_maxabs_diff_vs_score_export",
            "label_match", "consistency_ok"] + [f"{s}_{m}" for s in SCORES for m in ("auroc", "fpr95")] + [f"{s}_auroc" for s in SCORES_ALL[len(SCORES):]]
    new = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=cols)
    if new:
        w.writeheader()
    for held in a.classes:
        safe = held.replace("/", "_")
        for seed in a.seeds:
            todo = [m for m in a.models if f"{m}_loao_T16_{safe}_seed{seed}" not in done]
            if not todo:
                continue
            c2 = {**cfg, "split": {**cfg["split"], "regime": "loao", "loao_class": held}}
            wd = build_windowed(df, c2, regime="loao", T=T, seed=seed)
            hid = wd.hierarchy.index["label19"][held]
            y_te = wd.labels("test", "label19")
            unk = (y_te == hid).astype(int)
            y_tr_all = wd.labels("train", "label19")
            assert not (y_tr_all == hid).any(), "held-out class present in training windows"
            sub = strat_subsample(y_tr_all, MAX_TRAIN, np.random.default_rng(RNG))
            st_fit, y_fit = wd.starts["train"][sub], y_tr_all[sub]
            for m in todo:
                run = f"{m}_loao_T16_{safe}_seed{seed}"
                t0 = time.time()
                model = build_model(m, len(wd.features), wd.hierarchy, cfg["model"]).to(device)
                model.load_state_dict(torch.load(ZD / run / "model.pt", map_location="cpu", weights_only=True))
                Wt = model.heads.heads["label19"].weight.detach().float()
                bt = model.heads.heads["label19"].bias.detach().float()
                # ---- fit on training embeddings
                H, LG = [], []
                for lg, h in embed(model, wd, st_fit, device, amp):
                    H.append(h); LG.append(lg)
                H, LG = torch.cat(H), torch.cat(LG)
                d = H.shape[1]
                Hn = H.cpu().numpy().astype(np.float64)
                cls = np.unique(y_fit)
                mus, precs = [], []
                for c in cls:
                    xc = Hn[y_fit == c]
                    if len(xc) < 2:
                        continue
                    lw = LedoitWolf().fit(xc)
                    mus.append(xc.mean(0)); precs.append(lw.precision_)
                mus_t = torch.tensor(np.stack(mus), dtype=torch.float32, device=device)
                precs_t = torch.tensor(np.stack(precs), dtype=torch.float32, device=device)
                react_c = float(np.percentile(Hn, REACT_PCT))
                Hnorm = torch.nn.functional.normalize(H, dim=1)
                u = -(torch.linalg.pinv(Wt) @ bt)
                Hc = H - u
                cov = (Hc - Hc.mean(0)).T @ (Hc - Hc.mean(0)) / (len(Hc) - 1)
                evals, evecs = torch.linalg.eigh(cov.double())
                P = evecs[:, -(d // 2):].float()                       # principal subspace (top d/2)
                res_tr = (Hc - (Hc @ P) @ P.T).norm(dim=1)
                alpha = float(LG.max(1).values.sum() / res_tr.sum())
                # record train subsample and val embeddings (small) for the export record
                va_starts = wd.starts["val"]
                Hv = torch.cat([h for _, h in embed(model, wd, va_starts, device, amp)]).cpu().numpy().astype(np.float16) if len(va_starts) else np.zeros((0, d), np.float16)
                np.savez_compressed(out_emb / f"{run}_fit_val.npz", train_emb=H.cpu().numpy().astype(np.float16), train_y19=y_fit.astype(np.int8),
                                    val_emb=Hv, W19=Wt.cpu().numpy(), b19=bt.cpu().numpy(), react_clip=react_c, vim_alpha=alpha)
                # ---- score test windows (streamed)
                S = {k: [] for k in SCORES_ALL}
                for lg, h in embed(model, wd, wd.starts["test"], device, amp):
                    S["energy"].append(-energy_score(lg))              # higher = more unknown
                    diff = h[:, None, :] - mus_t[None, :, :]           # (B, C, d)
                    md = torch.einsum("bcd,cde,bce->bc", diff, precs_t, diff).clamp_min(0).sqrt() / (d ** 0.5)
                    S["mahalanobis_lw"].append(md.min(1).values)
                    hn = torch.nn.functional.normalize(h, dim=1)
                    sim = hn @ Hnorm.T
                    kth = sim.topk(K_NN, dim=1).values[:, -1]
                    S["knn50"].append((2 - 2 * kth).clamp_min(0).sqrt())
                    lr = torch.minimum(h, torch.tensor(react_c, device=device)) @ Wt.T + bt
                    S["react"].append(-torch.logsumexp(lr, dim=-1))
                    hc = h - u
                    res = (hc - (hc @ P) @ P.T).norm(dim=1)
                    S["vim"].append(alpha * res - torch.logsumexp(lg, dim=-1))
                    for t in TEMPS:
                        if t != 1:
                            S[f"energy_T{t}"].append(-(t * torch.logsumexp(lg / t, dim=-1)))
                        S[f"msp_T{t}"].append(-torch.softmax(lg / t, dim=-1).max(-1).values)
                S = {k: torch.cat(v).cpu().numpy().astype(np.float32) for k, v in S.items()}
                np.savez_compressed(out_emb / f"{run}_test_scores.npz", y19=y_te.astype(np.int8), hid=hid, **S)
                # ---- checks against score_export and metrics
                se = np.load(SE / f"{run}.npz")
                label_match = bool(len(se["y19"]) == len(y_te) and (se["y19"] == y_te).all())
                diff_e = float(np.abs(-S["energy"] - se["energy"]).max()) if len(se["energy"]) == len(y_te) else float("nan")
                row = dict(run=run, model=m, held_out_class=held, seed=seed, n_test=len(y_te), n_unknown=int(unk.sum()), n_train_fit=len(sub),
                           energy_maxabs_diff_vs_score_export=diff_e, label_match=label_match,
                           consistency_ok=bool(label_match and diff_e <= 1e-3))
                for k in SCORES:
                    row[f"{k}_auroc"] = float(roc_auc_score(unk, S[k]))
                    row[f"{k}_fpr95"] = fpr_at_95(unk, S[k])
                for k in SCORES_ALL[len(SCORES):]:
                    row[f"{k}_auroc"] = float(roc_auc_score(unk, S[k]))
                w.writerow(row); fcsv.flush()
                say(f"{run}: n_test={len(y_te):,} consistency_ok={row['consistency_ok']} (max|dE|={diff_e:.2e}) "
                    + " ".join(f"{k}={row[k + '_auroc']:.4f}" for k in SCORES) + f"; {time.time() - t0:.0f}s")
                del model, H, LG, Hnorm, S
                torch.cuda.empty_cache(); gc.collect()
            del wd
            gc.collect()
    say("T3 scoring finished")


if __name__ == "__main__":
    main()
