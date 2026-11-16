"""Edge-IIoTset NATIVE preprocessing helpers.

Project → encode strings → dedup → split → fit scaler.

Imports (NOT modifies) Paper 1's Scaler class from trace_iomt.data.preprocess
for engineering parity (zero-variance fallback, NaN safety, clipping).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import (not modify) Paper 1's Scaler class.
from trace_iomt.data.preprocess import Scaler

from .labels import LABEL_MAPPING_NATIVE
from .schema import (
    CATEGORICAL_COLUMNS, DROPPED_COLUMNS, NATIVE_COLUMNS, NUMERIC_COLUMNS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary building for categorical columns
# ---------------------------------------------------------------------------

def build_vocabularies(csv_path: Path, chunk_size: int = 500_000,
                      ) -> dict[str, dict[str, int]]:
    """Scan the full CSV once to collect every unique value per categorical column.

    Returns a dict: {col_name: {value_string: int_code}}.
    "" and "0" and "0.0" are all mapped to code 0 (the "absent" sentinel).
    """
    log.info("Building vocabularies for %d categorical columns ...", len(CATEGORICAL_COLUMNS))
    uniques: dict[str, set[str]] = {c: set() for c in CATEGORICAL_COLUMNS}
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             usecols=CATEGORICAL_COLUMNS, low_memory=False,
                             dtype=str):
        for c in CATEGORICAL_COLUMNS:
            uniques[c].update(chunk[c].dropna().astype(str).unique())

    vocab: dict[str, dict[str, int]] = {}
    for c in CATEGORICAL_COLUMNS:
        # Code 0 reserved for "absent/null" sentinel values.
        absent = {"", "0", "0.0", "nan", "NaN", "<NA>"}
        values = sorted(v for v in uniques[c] if v not in absent)
        mapping = {v: i + 1 for i, v in enumerate(values)}
        for a in absent:
            mapping[a] = 0
        vocab[c] = mapping
        log.info("  %-28s %6d unique values (+ absent sentinel = code 0)",
                 c, len(values))
    return vocab


def encode_categorical(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    """Label-encode a string series using the given vocab mapping. Unknown → 0."""
    arr = series.astype(str).map(mapping).fillna(0).to_numpy().astype(np.int32)
    return arr


# ---------------------------------------------------------------------------
# Projection: raw DNN CSV chunk → NATIVE 46-col float32 table + labels
# ---------------------------------------------------------------------------

def project_chunk_native(chunk_native: pd.DataFrame,
                          vocab: dict[str, dict[str, int]],
                          ) -> pd.DataFrame:
    """Project a raw Edge-IIoTset chunk to the NATIVE 46-col schema.

    - 15 ID/payload cols are dropped per paper Section VI.
    - Numeric cols coerced to float32 (non-numeric → 0.0).
    - Categorical cols label-encoded via supplied vocab (unknown → 0).
    - label_native (Attack_type verbatim) and label_coarse (native 6-class) attached.
    """
    # Drop 15 ID/payload columns.
    drop_present = [c for c in DROPPED_COLUMNS if c in chunk_native.columns]
    if drop_present:
        chunk_native = chunk_native.drop(columns=drop_present)

    n = len(chunk_native)
    out = {}

    for c in NUMERIC_COLUMNS:
        if c in chunk_native.columns:
            out[c] = pd.to_numeric(chunk_native[c], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        else:
            out[c] = np.zeros(n, dtype=np.float32)

    for c in CATEGORICAL_COLUMNS:
        if c in chunk_native.columns:
            out[c] = encode_categorical(chunk_native[c], vocab[c]).astype(np.float32)
        else:
            out[c] = np.zeros(n, dtype=np.float32)

    projected = pd.DataFrame(out, columns=NATIVE_COLUMNS, copy=False)

    native_attack = chunk_native["Attack_type"].astype(str)
    projected["label_native"] = native_attack.to_numpy()
    projected["label_coarse"] = native_attack.map(LABEL_MAPPING_NATIVE).fillna("UNKNOWN").to_numpy()

    return projected


# ---------------------------------------------------------------------------
# Row-level dedup
# ---------------------------------------------------------------------------

def row_level_dedup_native(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level drop_duplicates on NATIVE_COLUMNS + label_native."""
    before = len(df)
    key_cols = NATIVE_COLUMNS + ["label_native"]
    df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    log.info("Dedup (row-level): %d -> %d (removed %d, %.2f%%)",
             before, len(df), before - len(df),
             100.0 * (before - len(df)) / max(before, 1))
    return df


# ---------------------------------------------------------------------------
# Stratified 70/15/15 split (fixed split_seed=42)
# ---------------------------------------------------------------------------

def stratified_split_70_15_15_native(df: pd.DataFrame, split_seed: int = 42,
                                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        df, test_size=0.30,
        stratify=df["label_coarse"], random_state=split_seed)
    val, test = train_test_split(
        temp, test_size=0.50,
        stratify=temp["label_coarse"], random_state=split_seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Scaler fit (uses Paper 1's Scaler class imported, not modified)
# ---------------------------------------------------------------------------

def fit_scaler_native(train: pd.DataFrame, seed: int) -> Scaler:
    """Fit Paper 1's Scaler(kind='standard', clip=10) on train[NATIVE_COLUMNS].

    Paper 1's Scaler has built-in zero-variance fallback (critical for columns
    whose train slice happens to be constant) and NaN/inf safe clipping.
    """
    scaler = Scaler(kind="standard", clip=10.0)
    X = train[NATIVE_COLUMNS].to_numpy().astype(np.float64)
    scaler.fit(X, max_rows=500_000, seed=seed)
    return scaler


# ---------------------------------------------------------------------------
# File hashing (for MANIFEST)
# ---------------------------------------------------------------------------

def sha256_file(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()
