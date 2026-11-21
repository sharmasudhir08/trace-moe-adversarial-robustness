"""Split regimes and windows of consecutive flows.

Rows are grouped into contiguous *segments* (a whole file, or the head/tail of a file). A window is
T consecutive rows inside one segment, so no window ever crosses a train/val/test boundary.
The label of a window is the label of its last row.
"""
import gc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .labels import LEVELS, Hierarchy
from .loaders import feature_cols
from .preprocess import Scaler, dedup_windows, protocol_tokens

PROTOCOL_COL = "Protocol Type"


@dataclass
class WindowedData:
    X: np.ndarray                      # (N, F) scaled features, rows in segment order
    y: dict                            # level -> (N,) int labels per row
    proto: np.ndarray                  # (N,) protocol token per row
    starts: dict                       # split -> window start positions into X
    T: int
    features: list
    hierarchy: Hierarchy
    scaler: Scaler
    row_index: np.ndarray              # position in X -> row index in the source table
    report: dict = field(default_factory=dict)

    def windows(self, split):
        return self.starts[split]

    def gather(self, starts):
        """Return (B, T, F) features, (B, T) protocol tokens, and labels of the last row per level."""
        idx = starts[:, None] + np.arange(self.T)[None, :]
        last = starts + self.T - 1
        return self.X[idx], self.proto[idx], {lvl: self.y[lvl][last] for lvl in self.y}

    def labels(self, split, level):
        return self.y[level][self.starts[split] + self.T - 1]


def _segments(df, regime, cfg):
    """Return a list of (split_name, row_indices_in_time_order)."""
    val_frac = cfg["data"]["val_tail_fraction"]
    segs = []
    order = df.sort_values(["source_file", "row_in_file"]).index.to_numpy()
    files = df.loc[order, "source_file"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, files[1:] != files[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        file_split = df.at[rows[0], "split"]
        n = len(rows)
        if regime in ("official", "loao"):
            if file_split == "train":
                cut = int(round(n * (1 - val_frac)))
                segs += [("train", rows[:cut]), ("val", rows[cut:])]
            else:
                segs.append(("test", rows))
        elif regime == "temporal":
            cut_test = int(round(n * (1 - cfg["split"]["temporal_test_fraction"])))
            cut_val = int(round(cut_test * (1 - val_frac)))
            segs += [("train", rows[:cut_val]), ("val", rows[cut_val:cut_test]), ("test", rows[cut_test:])]
        elif regime == "random":
            segs.append(("pool", rows))
        else:
            raise ValueError(f"Unknown split regime: {regime}")
    return [(s, r) for s, r in segs if len(r) > 0]


def _random_split(starts, labels, test_frac, val_frac, seed):
    """Stratified random split of window starts (this is the leaky protocol used by most papers)."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        s = starts[labels == c]
        s = s[rng.permutation(len(s))]
        n_test = int(round(len(s) * test_frac))
        n_val = int(round((len(s) - n_test) * val_frac))
        out["test"].append(s[:n_test])
        out["val"].append(s[n_test:n_test + n_val])
        out["train"].append(s[n_test + n_val:])
    return {k: np.sort(np.concatenate(v)) if v else np.array([], dtype=np.int64) for k, v in out.items()}


def build_windowed(df, cfg, regime=None, T=None, hierarchy=None, seed=None, logger=None, drop_features=None,
                   low_memory=False):
    """Turn a table into scaled, split, windowed arrays according to cfg."""
    # low_memory: a caller may pass the table inside a one-element list and drop its own reference; the table
    # is then freed as soon as the arrays have been extracted, before scaling allocates the second full-size array.
    holder = None
    if isinstance(df, list):
        holder, df = df, df[0]
    regime = regime or cfg["split"]["regime"]
    T = int(T or cfg["windows"]["length"])
    seed = cfg["train"]["seed"] if seed is None else seed
    hierarchy = hierarchy or Hierarchy.from_frame(df)
    drop = set(drop_features if drop_features is not None else cfg["data"].get("drop_features") or [])
    feats = [c for c in feature_cols(df) if c not in drop]

    segs = _segments(df, regime, cfg)
    row_index = np.concatenate([r for _, r in segs])
    raw = df.loc[row_index, feats].to_numpy(dtype=np.float32)
    proto = protocol_tokens(df.loc[row_index, PROTOCOL_COL].to_numpy()) if PROTOCOL_COL in df.columns \
        else np.zeros(len(row_index), dtype=np.int64)
    y = {}
    for lvl in LEVELS:
        mapping = hierarchy.index[lvl]
        y[lvl] = df.loc[row_index, lvl].astype(str).map(lambda v: mapping.get(v, -1)).to_numpy(np.int64)
    if holder is not None:
        holder.clear()
        del df
        gc.collect()

    # window starts per segment
    starts = {"train": [], "val": [], "test": [], "pool": []}
    stride_tr, stride_ev = cfg["windows"]["stride_train"], cfg["windows"]["stride_eval"]
    stride_val = cfg["windows"].get("stride_val", stride_ev)
    if T == 1:
        stride_tr = stride_ev = stride_val = 1
    off = 0
    for split, rows in segs:
        n = len(rows)
        stride = {"train": stride_tr, "pool": stride_tr, "val": stride_val}.get(split, stride_ev)
        if n >= T:
            starts[split].append(off + np.arange(0, n - T + 1, stride))
        off += n
    starts = {k: (np.concatenate(v) if v else np.array([], dtype=np.int64)) for k, v in starts.items()}

    if regime == "random":
        lab = y["label19"][starts["pool"] + T - 1]
        starts = _random_split(starts["pool"], lab, cfg["split"]["random_test_fraction"],
                               cfg["data"]["val_tail_fraction"], seed)
    else:
        starts.pop("pool")

    if regime == "loao":
        held = cfg["split"].get("loao_class")
        if not held:
            raise ValueError("split.loao_class must be set for the loao regime")
        c = hierarchy.index["label19"][held]
        for s in ("train", "val"):
            st = starts[s]
            starts[s] = st[y["label19"][st + T - 1] != c]

    # drop windows whose label is unknown to the hierarchy
    for s in starts:
        st = starts[s]
        starts[s] = st[y["label6"][st + T - 1] >= 0]

    # fit the scaler on rows covered by training windows only
    tr_rows = np.unique((starts["train"][:, None] + np.arange(T)[None, :]).ravel()) if len(starts["train"]) \
        else np.arange(len(raw))
    scaler = Scaler(cfg["data"]["scaler"], cfg["data"]["clip_value"]).fit(raw[tr_rows], seed=seed)
    X = scaler.transform(raw, chunk=250_000 if low_memory else 1_000_000)   # smaller chunks: lower transient memory, same numbers
    del raw

    report = {"regime": regime, "T": T, "n_features": len(feats), "dropped_features": sorted(drop)}
    if cfg["data"].get("drop_exact_duplicates", True):
        starts, dup = dedup_windows(X, starts, T, logger)
        report.update(dup)
    report.update({f"n_windows_{k}": int(len(v)) for k, v in starts.items()})
    if logger:
        logger.info(f"Windows ({regime}, T={T}): " +
                    ", ".join(f"{k}={len(v):,}" for k, v in starts.items()))
    return WindowedData(X=X, y=y, proto=proto, starts=starts, T=T, features=feats, hierarchy=hierarchy,
                        scaler=scaler, row_index=row_index, report=report)


def class_balanced_sample(labels, max_per_class, seed, oversample=False):
    """Indices of at most max_per_class items per class (used once per epoch for the training set).

    When oversample=True, classes with FEWER than max_per_class windows are up-sampled with
    replacement so every class contributes exactly max_per_class windows per epoch. This is
    essential for CICIoMT2024's rare classes under windowing: Recon-Ping_Sweep has ~45 windows
    at T=16, so without oversampling it gets ~45 gradient updates per epoch while DDoS giants
    get 60,000 — a 1000x gradient-signal imbalance that under-trains rare classes. With
    oversample=True, every class gets exactly max_per_class updates per epoch.

    Backward-compatible: oversample defaults to False so existing configs are unchanged.
    """
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if max_per_class:
            if len(idx) > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            elif oversample and len(idx) < max_per_class:
                idx = rng.choice(idx, max_per_class, replace=True)
        out.append(idx)
    return rng.permutation(np.concatenate(out))


def class_counts(labels, n_classes):
    return np.bincount(labels[labels >= 0], minlength=n_classes)


def summary_frame(wd, split):
    lab = wd.labels(split, "label19")
    names = wd.hierarchy.classes["label19"]
    return pd.Series(np.bincount(lab, minlength=len(names)), index=names, name=split)
