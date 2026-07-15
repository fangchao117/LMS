"""
第七梯队 —— 因子训练产出（factor_train.py 自动生成，勿手改）
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ops import delay, delta, ts_corr, ts_mean, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


def compute_panel(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    names = names or list(_REGISTRY.keys())
    return pd.DataFrame({n: _REGISTRY[n](df) for n in names if n in _REGISTRY})


def _oi(df: pd.DataFrame) -> pd.Series:
    if "open_interest" in df.columns:
        return df["open_interest"].astype(float)
    return df["volume"].astype(float) * 0.0

@register("disc_mom_vol_5")
def disc_mom_vol_5(df: pd.DataFrame) -> pd.Series:
    c, v = df['close'].astype(float), df['volume'].astype(float)
    return (c / delay(c, 5) - 1.0) * ts_rank(v, 5)

@register("disc_don_pos_20")
def disc_don_pos_20(df: pd.DataFrame) -> pd.Series:
    c = df['close'].astype(float)
    hi = df['high'].rolling(20, min_periods=1).max()
    lo = df['low'].rolling(20, min_periods=1).min()
    pos = (c - lo) / (hi - lo + 1e-12) - 0.5
    return pos * (hi - lo) / (ts_mean(hi - lo, 20) + 1e-12)

@register("disc_accel_30")
def disc_accel_30(df: pd.DataFrame) -> pd.Series:
    c = df['close'].astype(float)
    mom5 = c / delay(c, 5) - 1.0
    return delta(mom5, 6)

@register("disc_accel_40")
def disc_accel_40(df: pd.DataFrame) -> pd.Series:
    c = df['close'].astype(float)
    mom5 = c / delay(c, 5) - 1.0
    return delta(mom5, 8)

@register("disc_don_pos_30")
def disc_don_pos_30(df: pd.DataFrame) -> pd.Series:
    c = df['close'].astype(float)
    hi = df['high'].rolling(30, min_periods=1).max()
    lo = df['low'].rolling(30, min_periods=1).min()
    pos = (c - lo) / (hi - lo + 1e-12) - 0.5
    return pos * (hi - lo) / (ts_mean(hi - lo, 30) + 1e-12)

@register("disc_accel_20")
def disc_accel_20(df: pd.DataFrame) -> pd.Series:
    c = df['close'].astype(float)
    mom5 = c / delay(c, 5) - 1.0
    return delta(mom5, 4)


@register("disc_lgb_score")
def disc_lgb_score(df: pd.DataFrame) -> pd.Series:
    """ML 预测分；完整重算请运行 factor_train.py。"""
    import json
    from pathlib import Path
    import config
    p = config.ARTIFACT_PATH / "disc_lgb_score.parquet"
    if not p.exists():
        return pd.Series(0.0, index=df.index)
    ser = pd.read_parquet(p)["disc_lgb_score"]
    ser.index = df.index[: len(ser)]
    return ser.reindex(df.index).fillna(0.0)
