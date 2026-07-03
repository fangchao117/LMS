"""
信号后处理 —— 将原始回归预测转为可交易的因果 z-score 信号（无未来函数）。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from vnpy.alpha import AlphaDataset, Segment


def causal_rolling_zscore(
    pred: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """仅用截至当日的历史预测做滚动 z-score。"""
    out = np.full(len(pred), np.nan, dtype=float)
    for i in range(len(pred)):
        start = max(0, i - window + 1)
        chunk = pred[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < min_periods:
            continue
        std = float(chunk.std())
        if std < 1e-12:
            continue
        out[i] = (pred[i] - float(chunk.mean())) / std
    return out


def build_processed_predictions(
    dataset: AlphaDataset,
    model,
    window: int,
    min_periods: int = 20,
) -> dict[Segment, np.ndarray]:
    """按时间顺序拼接各段预测，做因果 z-score 后再切回各段。"""
    segments = [Segment.TRAIN, Segment.VALID, Segment.TEST]
    parts: list[np.ndarray] = []
    for seg in segments:
        infer = dataset.fetch_infer(seg).sort(["datetime", "vt_symbol"])
        raw = model.predict(dataset, seg)
        parts.append(raw)
    raw_all = np.concatenate(parts)
    z_all = causal_rolling_zscore(raw_all, window=window, min_periods=min_periods)

    out: dict[Segment, np.ndarray] = {}
    offset = 0
    for seg, raw in zip(segments, parts):
        out[seg] = z_all[offset : offset + len(raw)]
        offset += len(raw)
    return out


def segment_signal_frame(
    dataset: AlphaDataset,
    segment: Segment,
    signal: np.ndarray,
    signal_col: str = "signal",
) -> pl.DataFrame:
    infer = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
    return infer.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series(signal_col, signal)
    )
