"""Training and prediction loops for the neural models (mixed precision, early stopping, MoE balance loss)."""
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.windows import class_balanced_sample, class_counts
from .metrics import evaluate_levels
from .models.heads import LEVELS, HierarchicalLoss, energy_score


def _to_tensors(wd, starts, device):
    x, proto, y = wd.gather(starts)
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
    pt = torch.from_numpy(np.ascontiguousarray(proto)).to(device, non_blocking=True)
    yt = {lvl: torch.from_numpy(v).to(device) for lvl, v in y.items()}
    return xt, pt, yt


def iterate(wd, starts, batch_size, device):
    for i in range(0, len(starts), batch_size):
        yield _to_tensors(wd, starts[i:i + batch_size], device)


@torch.no_grad()
def predict(model, wd, starts, device, batch_size=1024, amp=True, return_embeddings=False):
    """Probabilities for every level, the energy score, the expert chosen per window (TRACE-MoE only)
    and (optionally) embeddings."""
    model.eval()
    probs = {lvl: [] for lvl in LEVELS}
    energies, embs, routes = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, p, _ in iterate(wd, starts, batch_size, device):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, h = model(x, p)
        pr = model.heads.probabilities(logits)
        for lvl in LEVELS:
            probs[lvl].append(pr[lvl].cpu().numpy())
        energies.append(energy_score(logits["label19"]).cpu().numpy())
        if getattr(model, "last_route", None) is not None:
            routes.append(model.last_route.cpu().numpy())
        if return_embeddings:
            embs.append(h.float().cpu().numpy())
    out = {"proba": {lvl: np.concatenate(v) if v else np.zeros((0, model.heads.n[lvl]))
                     for lvl, v in probs.items()},
           "energy": np.concatenate(energies) if energies else np.zeros(0)}
    if routes:
        out["route"] = np.concatenate(routes)
    if return_embeddings:
        out["embeddings"] = np.concatenate(embs)
    return out


def evaluate(model, wd, split, device, cfg, starts=None):
    starts = wd.starts[split] if starts is None else starts
    pred = predict(model, wd, starts, device, amp=cfg["train"]["amp"])
    last = starts + wd.T - 1
    y_true = {lvl: wd.y[lvl][last] for lvl in LEVELS}
    metrics = evaluate_levels(y_true, pred["proba"], wd.hierarchy, cfg["evaluation"]["recall_at_fpr"])
    return metrics, pred, y_true


def _priors(wd):
    lab = {lvl: wd.labels("train", lvl) for lvl in LEVELS}
    return {lvl: class_counts(lab[lvl], wd.hierarchy.n(lvl)) for lvl in LEVELS}


def _atomic_save(obj, path):
    """Write to a temporary file, then replace, so a crash never leaves a half-written checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _checkpoint_dir(cfg):
    """Opt-in checkpointing: active only when the environment variable P4_CHECKPOINT_DIR is set.
    One sub-folder per run, keyed by model, regime, held-out class, window length and seed."""
    root = os.environ.get("P4_CHECKPOINT_DIR")
    if not root:
        return None
    key = (f"{cfg['model']['name']}_{cfg['split']['regime']}_T{cfg['windows']['length']}"
           f"_{cfg['split'].get('loao_class') or 'none'}_seed{cfg['train']['seed']}")
    d = Path(root) / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def train_model(model, wd, cfg, device, logger):
    """Train with early stopping on validation macro-F1 of the finest level.
    Models with experts add model.aux_loss (load balancing) times model.moe.balance_weight.
    Returns (best_model, history DataFrame).

    Checkpointing (opt-in, no effect on the training maths): when P4_CHECKPOINT_DIR is set, after every epoch it
    saves epochs/epoch_NNN.pt (model weights and that epoch's metrics), history.csv, best.pt (best weights so far)
    and resume.pt (model, optimiser, scheduler, grad-scaler, early-stopping counters, RNG state; overwritten
    atomically), plus a full snapshot full_epoch_NNN.pt every P4_CHECKPOINT_KEEP_EVERY epochs (default 10).
    If resume.pt exists for the same run, training continues from the next epoch.
    """
    tc = cfg["train"]
    model.to(device)
    criterion = HierarchicalLoss(model.heads, _priors(wd), cfg["loss"]).to(device)
    params = list(model.parameters())
    balance_w = float((cfg["model"].get("moe") or {}).get("balance_weight", 0.01))
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tc["epochs"], 1))
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_starts = wd.starts["train"]
    train_lab = wd.labels("train", "label19")
    oversample = bool(cfg.get("data", {}).get("oversample_small_classes", False))
    best, best_score, bad, hist = None, -1.0, 0, []

    ckpt_dir = _checkpoint_dir(cfg)
    keep_every = int(os.environ.get("P4_CHECKPOINT_KEEP_EVERY", "10"))
    signature = {"epochs": tc["epochs"], "lr": tc["lr"], "batch_size": tc["batch_size"], "seed": tc["seed"],
                 "weight_decay": tc["weight_decay"], "patience": tc["patience"],
                 "n_params": sum(p.numel() for p in model.parameters())}
    start_epoch, finished = 1, False
    if ckpt_dir is not None:
        logger.info(f"Checkpointing to {ckpt_dir} (full snapshot every {keep_every} epochs)")
        resume_path = ckpt_dir / "resume.pt"
        if resume_path.exists():
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            if ck["signature"] != signature:
                raise RuntimeError(f"Checkpoint at {resume_path} was made with different settings "
                                   f"{ck['signature']} vs {signature}; refusing to resume. Move it aside.")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            best, best_score, bad, hist = ck["best"], ck["best_score"], ck["bad"], ck["hist"]
            start_epoch, finished = ck["epoch"] + 1, ck.get("finished", False)
            _set_rng_state(ck["rng"])
            logger.info(f"RESUMED from checkpoint: last completed epoch {ck['epoch']}, best val macro-F1 "
                        f"{best_score:.4f}, patience counter {bad}, finished={finished}")

    for epoch in range(start_epoch, (tc["epochs"] + 1) if not finished else start_epoch):
        model.train()
        t0 = time.time()
        idx = class_balanced_sample(train_lab, cfg["windows"]["max_windows_per_class_train"],
                                     tc["seed"] + epoch, oversample=oversample)
        starts = train_starts[idx]
        n_steps = math.ceil(len(starts) / tc["batch_size"])
        tot, parts_sum, aux_sum = 0.0, {}, 0.0
        for step, (x, p, y) in enumerate(iterate(wd, starts, tc["batch_size"], device)):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, h = model(x, p)
                loss, parts = criterion(logits, y)
                aux = getattr(model, "aux_loss", None)
                if aux is not None and balance_w > 0:
                    loss = loss + balance_w * aux
                    aux_sum += aux.detach().item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += loss.detach().item()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + (v.detach().item() if hasattr(v, "detach") else float(v))
        sched.step()

        val_metrics, _, _ = evaluate(model, wd, "val", device, cfg) if len(wd.starts.get("val", [])) else ({}, None, None)
        score = val_metrics.get("label19", {}).get("macro_f1", -tot)
        row = {"epoch": epoch, "train_loss": tot / max(n_steps, 1), "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1),
               **{f"loss_{k}": v / max(n_steps, 1) for k, v in parts_sum.items()},
               **{f"val_macro_f1_{lvl}": val_metrics.get(lvl, {}).get("macro_f1") for lvl in LEVELS}}
        if getattr(model, "aux_loss", None) is not None or aux_sum:
            row["balance_loss"] = aux_sum / max(n_steps, 1)
        hist.append(row)
        logger.info(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | val macro-F1 "
                    f"2:{row['val_macro_f1_label2'] or 0:.4f} 6:{row['val_macro_f1_label6'] or 0:.4f} "
                    f"19:{row['val_macro_f1_label19'] or 0:.4f} | {row['seconds']}s")
        improved = score > best_score + 1e-4
        stop = False
        if improved:
            best_score, best, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                logger.info(f"Early stopping at epoch {epoch} (best val macro-F1 {best_score:.4f})")
                stop = True
        if ckpt_dir is not None:
            (ckpt_dir / "epochs").mkdir(exist_ok=True)
            _atomic_save({"epoch": epoch, "model": model.state_dict(), "row": row}, ckpt_dir / "epochs" / f"epoch_{epoch:03d}.pt")
            pd.DataFrame(hist).to_csv(ckpt_dir / "history.csv", index=False)
            if improved:
                _atomic_save({"epoch": epoch, "model": best, "best_score": best_score}, ckpt_dir / "best.pt")
            state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "best": best, "best_score": best_score, "bad": bad, "hist": hist,
                     "rng": _rng_state(), "signature": signature, "finished": bool(stop or epoch >= tc["epochs"])}
            _atomic_save(state, ckpt_dir / "resume.pt")
            if keep_every > 0 and epoch % keep_every == 0:
                _atomic_save(state, ckpt_dir / f"full_epoch_{epoch:03d}.pt")
        if stop:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, pd.DataFrame(hist)
