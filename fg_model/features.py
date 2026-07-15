"""特征工程 + 标签。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _oi(df: pd.DataFrame) -> pd.Series:
    if "open_interest" in df.columns:
        return df["open_interest"].astype(float)
    return pd.Series(0.0, index=df.index)


def _delay(s: pd.Series, n: int) -> pd.Series:
    return s.shift(n)


def _ts_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(1, w // 2)).mean()


def _ts_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(1, w // 2)).std()


def _ts_rank(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def _ts_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    return a.rolling(w, min_periods=max(3, w // 2)).corr(b)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)
    v = df["volume"].astype(float).replace(0, np.nan)
    oi = _oi(df)
    ret = c.pct_change()
    hl = (df["high"] - df["low"]) / (c + 1e-12)
    oi_ret = oi.pct_change()

    x = pd.DataFrame(index=df.index)
    for w in (5, 10, 20, 30, 40):
        x[f"ret_{w}"] = c / _delay(c, w) - 1.0
        x[f"vol_r_{w}"] = v / (_ts_mean(v, w) + 1e-12) - 1.0
        x[f"oi_r_{w}"] = oi / (_ts_mean(oi, w) + 1e-12) - 1.0
        x[f"std_{w}"] = _ts_std(ret, w)
        x[f"rank_ret_{w}"] = _ts_rank(ret, w)
        x[f"rank_vol_{w}"] = _ts_rank(v, w)
        x[f"corr_ro_{w}"] = _ts_corr(ret, oi_ret, w)
        x[f"mom_vol_{w}"] = x[f"ret_{w}"] * x[f"rank_vol_{w}"]
    x["hl_ratio"] = hl / (_ts_mean(hl, 20) + 1e-12) - 1.0
    x["body"] = (c - df["open"].astype(float)) / (c + 1e-12)
    for n in (20, 30, 38):
        hi = df["high"].rolling(n, min_periods=1).max()
        lo = df["low"].rolling(n, min_periods=1).min()
        pos = (c - lo) / (hi - lo + 1e-12) - 0.5
        x[f"don_pos_{n}"] = pos * (hi - lo) / (_ts_mean(hi - lo, n) + 1e-12)
    return x.replace([np.inf, -np.inf], np.nan)


def build_feature_matrix(
    df: pd.DataFrame,
    use_fd_vib: bool = False,
    top_n: int | None = None,
) -> pd.DataFrame:
    """基础特征 + 可选 fd_vib 因子（仅 fg_model 使用）。"""
    base = build_features(df)
    if not use_fd_vib:
        return base
    from fd_vib_factors import load_fd_vib_features

    fv = load_fd_vib_features(df, top_n=top_n)
    if fv.empty:
        return base
    return pd.concat([base, fv], axis=1)


def build_label(df: pd.DataFrame, horizon: int | None = None) -> pd.Series:
    h = horizon or config.LABEL_HORIZON
    c = df["close"].astype(float)
    return c.pct_change(h).shift(-h)


def spearman_ic(pred: pd.Series, label: pd.Series) -> float:
    valid = pred.notna() & label.notna() & np.isfinite(pred) & np.isfinite(label)
    if valid.sum() < 50:
        return 0.0
    return float(pred[valid].corr(label[valid], method="spearman"))
