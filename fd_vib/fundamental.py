"""
第一梯队 —— 商品基本面 / 持仓量量价因子

数据：30m open_interest + volume；FG00 日线补充长历史 OI 水平。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ops import ts_corr, ts_mean, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}

_FG00_PATH = Path(__file__).resolve().parent.parent / "data" / "FG00.CZCE.parquet"
_FG00_CACHE: pd.DataFrame | None = None


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


def _load_fg00() -> pd.DataFrame:
    global _FG00_CACHE
    if _FG00_CACHE is None and _FG00_PATH.exists():
        d = pd.read_parquet(_FG00_PATH).sort_values("datetime").reset_index(drop=True)
        d["date"] = d["datetime"].dt.date
        _FG00_CACHE = d
    return _FG00_CACHE if _FG00_CACHE is not None else pd.DataFrame()


def _merge_fg00_daily(df: pd.DataFrame) -> pd.DataFrame:
    fg = _load_fg00()
    if fg.empty or "open_interest" not in fg.columns:
        return df
    cols = ["date", "open_interest", "volume", "close"]
    daily = fg[cols].copy()
    daily = daily.rename(columns={
        "open_interest": "oi_d",
        "volume": "vol_d",
        "close": "close_d",
    })
    m = df.copy()
    m["date"] = m["datetime"].dt.date
    return m.merge(daily, on="date", how="left")


def _oi_series(df: pd.DataFrame) -> pd.Series:
    if "open_interest" in df.columns and df["open_interest"].notna().any():
        return df["open_interest"].astype(float)
    merged = _merge_fg00_daily(df)
    if "oi_d" in merged.columns:
        return merged["oi_d"].astype(float)
    return pd.Series(np.nan, index=df.index)


@register("fund_oi_quad")
def fund_oi_quad(df: pd.DataFrame) -> pd.Series:
    """价涨增仓 +1，价跌减仓 +1（趋势延续象限）"""
    c = df["close"].astype(float)
    oi = _oi_series(df)
    pc, oc = c.diff(), oi.diff()
    out = np.zeros(len(df))
    out[(pc > 0) & (oc > 0)] = 1.0
    out[(pc <= 0) & (oc <= 0)] = 1.0
    out[(pc > 0) & (oc <= 0)] = -0.5
    out[(pc <= 0) & (oc > 0)] = -0.5
    return pd.Series(out, index=df.index)


@register("fund_oi_z20")
def fund_oi_z20(df: pd.DataFrame) -> pd.Series:
    oi = _oi_series(df)
    mu = ts_mean(oi, 20)
    sd = ts_std(oi, 20)
    return (oi - mu) / (sd + 1e-12)


@register("fund_oi_mom5")
def fund_oi_mom5(df: pd.DataFrame) -> pd.Series:
    oi = _oi_series(df)
    return oi.pct_change(5)


@register("fund_vol_ratio")
def fund_vol_ratio(df: pd.DataFrame) -> pd.Series:
    v = df["volume"].astype(float)
    return v / (ts_mean(v, 20) + 1e-12) - 1.0


@register("fund_price_oi_div")
def fund_price_oi_div(df: pd.DataFrame) -> pd.Series:
    """20 根价格趋势 vs 持仓趋势背离"""
    c = df["close"].astype(float)
    oi = _oi_series(df)
    pr = c / c.shift(20) - 1.0
    or_ = oi / oi.shift(20) - 1.0
    return -np.sign(pr) * np.sign(or_) * (pr.abs() + or_.abs())


@register("fund_vp_regime")
def fund_vp_regime(df: pd.DataFrame) -> pd.Series:
    """价量配合：同向为正"""
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    ret = c.pct_change()
    vr = v.pct_change()
    return np.sign(ret) * np.sign(vr)


@register("fund_oi_vol_corr")
def fund_oi_vol_corr(df: pd.DataFrame) -> pd.Series:
    oi = _oi_series(df)
    v = df["volume"].astype(float)
    return ts_corr(oi.pct_change(), v.pct_change(), 20)


@register("fund_oi_level_rank")
def fund_oi_level_rank(df: pd.DataFrame) -> pd.Series:
    oi = _oi_series(df)
    return ts_rank(oi, 60) - 0.5


@register("fund_inventory_proxy")
def fund_inventory_proxy(df: pd.DataFrame) -> pd.Series:
    """无库存数据时用 OI 日变化 + FG00 长历史 OI 分位作代理"""
    merged = _merge_fg00_daily(df)
    oi = _oi_series(df)
    oi_chg = oi.pct_change()
    if "oi_d" in merged.columns:
        oi_d = merged["oi_d"].astype(float)
        level = ts_rank(oi_d, 120) - 0.5
        return -level * 0.5 - oi_chg * 0.5
    return -oi_chg


def compute_panel(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    reg = all_factors()
    keys = names or list(reg.keys())
    out = {}
    for k in keys:
        if k not in reg:
            continue
        try:
            out[k] = reg[k](df).astype(float)
        except Exception:
            out[k] = np.nan
    return pd.DataFrame(out, index=df.index)
