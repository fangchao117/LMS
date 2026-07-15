"""预测分 → 交易信号。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def score_to_position(score: pd.Series, threshold: float | None = None) -> pd.Series:
    th = threshold if threshold is not None else config.SIGNAL_THRESHOLD
    out = np.zeros(len(score))
    s = score.to_numpy()
    out[s > th] = 1.0
    out[s < -th] = -1.0
    return pd.Series(out, index=score.index, name="position")


def smooth_position(pos: pd.Series, bars: int = 2) -> pd.Series:
    if bars <= 1:
        return pos
    out = pos.copy()
    prev = 0.0
    for i in range(len(out)):
        cur = float(out.iloc[i])
        if cur == 0:
            out.iloc[i] = prev
        else:
            prev = cur
    return out
