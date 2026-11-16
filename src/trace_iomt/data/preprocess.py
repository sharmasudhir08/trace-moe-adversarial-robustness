"""Scaling, protocol tokens and duplicate-window removal. Everything is fitted on training rows only."""
import numpy as np
import pandas as pd


class Scaler:
    """Robust (median/IQR) or standard (mean/std) scaling with clipping. NaNs become 0 after scaling."""

    def __init__(self, kind="robust", clip=10.0):
        self.kind, self.clip = kind, clip
        self.center_ = self.scale_ = None

    def fit(self, X, max_rows=500_000, seed=0):
        if len(X) > max_rows:
            X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
        X = X.astype(np.float64)
        if self.kind == "robust":
            self.center_ = np.nanmedian(X, axis=0)
            q75, q25 = np.nanpercentile(X, 75, axis=0), np.nanpercentile(X, 25, axis=0)
            scale = q75 - q25
            std = np.nanstd(X, axis=0)
            scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
        else:
            self.center_ = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            scale = np.where(std > 1e-12, std, 1.0)
        self.center_ = np.nan_to_num(self.center_)
        self.scale_ = scale
        return self

    def transform(self, X, chunk=1_000_000):
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, len(X), chunk):
            z = (X[i:i + chunk] - self.center_) / self.scale_
            z = np.nan_to_num(z, nan=0.0, posinf=self.clip, neginf=-self.clip)
            out[i:i + chunk] = np.clip(z, -self.clip, self.clip)
        return out

    def state(self):
        return {"kind": self.kind, "clip": self.clip,
                "center": self.center_.tolist(), "scale": self.scale_.tolist()}


# IP protocol numbers -> token id. Derived from the "Protocol Type" feature only (never from file names).
_PROTO_MAP = {0: 0, 1: 1, 6: 2, 17: 3}
N_PROTO_TOKENS = 5


def protocol_tokens(raw_protocol):
    """Bucket the (window-averaged) 'Protocol Type' value into {other/0, ICMP, TCP, UDP, mixed}."""
    if raw_protocol is None:
        return None
    v = np.nan_to_num(np.asarray(raw_protocol, dtype=np.float64), nan=-1)
    r = np.rint(v).astype(np.int64)
    tok = np.full(len(v), 4, dtype=np.int64)        # 4 = mixed / other value
    exact = np.abs(v - r) < 1e-6
    for proto, t in _PROTO_MAP.items():
        tok[exact & (r == proto)] = t
    return tok


def window_hashes(X, starts, T, seed=0):
    """Hash each window X[s:s+T] into a pair of floats (random projections). Equal windows -> equal hashes."""
    rng = np.random.default_rng(seed)
    F = X.shape[1]
    keys = []
    for _ in range(2):
        r = rng.standard_normal(F)
        p = np.round(X.astype(np.float64) @ r, 6)            # one number per row
        if T == 1:
            keys.append(p[starts])
        else:
            c = rng.standard_normal(T)
            # weighted sum over the T rows of each window
            keys.append(sum(c[t] * p[starts + t] for t in range(T)))
    return np.round(np.stack(keys, axis=1), 5)


def dedup_windows(X, starts_by_split, T, logger=None):
    """Drop duplicate training windows, and val/test windows identical to a training window."""
    out, report = dict(starts_by_split), {}
    if "train" not in out or len(out["train"]) == 0:
        return out, report
    h_tr = window_hashes(X, out["train"], T)
    _, first = np.unique(h_tr, axis=0, return_index=True)
    keep = np.sort(first)
    report["train_duplicate_windows_removed"] = int(len(h_tr) - len(keep))
    out["train"] = out["train"][keep]
    train_keys = pd.MultiIndex.from_arrays([h_tr[keep, 0], h_tr[keep, 1]])
    for s in ("val", "test"):
        if s in out and len(out[s]):
            h = window_hashes(X, out[s], T)
            mask = ~pd.MultiIndex.from_arrays([h[:, 0], h[:, 1]]).isin(train_keys)
            report[f"{s}_windows_also_in_train_removed"] = int((~mask).sum())
            out[s] = out[s][mask]
    if logger:
        logger.info(f"Duplicate removal: {report}")
    return out, report
