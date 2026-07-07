"""
信号生成 —— LMS 自适应滤波 + 多因子融合（严格无未来函数）

模式：
  · "return" —— 预测下一根收益
  · "trend"  —— NLMS 自适应价格趋势
  · "multi"  —— LMS + 动量/均线/量能/持仓/RSI 加权融合（推荐，收益最大化）

T 日收盘出信号 -> T+1 成交。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import config
import data
import factors
from lms_filter import LmsEnsemble


def generate_return(
    returns: np.ndarray,
    orders: tuple[int, ...] = config.FILTER_ORDERS,
    mu: float = config.MU,
    eps: float = config.EPS,
    leak: float = config.LEAK,
    vol_window: int = config.VOL_WINDOW,
) -> np.ndarray:
    n = len(returns)
    yhat = np.full(n, np.nan)

    s = pd.Series(returns)
    vol = s.rolling(vol_window).std().shift(1).to_numpy()
    rnorm = returns / (vol + eps)
    rnorm = np.nan_to_num(rnorm, nan=0.0, posinf=0.0, neginf=0.0)

    ens = LmsEnsemble(orders=orders, mu=mu, eps=eps, leak=leak)
    start = max(max(orders), vol_window) + 1

    for t in range(start, n - 1):
        recent = rnorm[t - ens.max_order + 1: t + 1]
        yhat[t] = ens.predict(recent)
        ens.adapt(recent, rnorm[t + 1])
    return yhat


def generate_trend(
    close: np.ndarray,
    orders: tuple[int, ...] = config.TREND_ORDERS,
    mu: float = config.TREND_MU,
    eps: float = config.EPS,
    leak: float = config.TREND_LEAK,
) -> np.ndarray:
    n = len(close)
    yhat = np.full(n, np.nan)

    s = pd.Series(close)
    ma = s.rolling(max(orders), min_periods=1).mean().to_numpy()
    pnorm = close / (ma + eps)

    ens = LmsEnsemble(orders=orders, mu=mu, eps=eps, leak=leak)
    start = max(orders) + 1

    for t in range(start, n - 1):
        recent = pnorm[t - ens.max_order + 1: t + 1]
        pred_norm = ens.predict(recent)
        pred_price = pred_norm * ma[t]
        yhat[t] = (pred_price - close[t]) / close[t]
        ens.adapt(recent, pnorm[t + 1])
    return yhat


def generate_lms_raw(returns: np.ndarray, close: np.ndarray, mode: str) -> np.ndarray:
    if mode == "trend":
        return generate_trend(close)
    return generate_return(returns)


def _apply_factor_agree(composite: np.ndarray, raw_factors: dict[str, np.ndarray]) -> np.ndarray:
    """子因子方向一致性过滤：不一致时置 NaN（策略层视为无信号）。"""
    min_agree = config.MIN_FACTOR_AGREE
    if min_agree <= 0:
        return composite

    dirs = factors.factor_directions(raw_factors)
    out = composite.copy()
    sign_comp = np.sign(out)
    n = len(out)
    names = [k for k in config.FACTOR_WEIGHTS if config.FACTOR_WEIGHTS[k] > 0 and k in dirs]

    for i in range(n):
        if np.isnan(out[i]) or sign_comp[i] == 0:
            continue
        agree = sum(1 for name in names if dirs[name][i] == sign_comp[i])
        if agree < min_agree:
            out[i] = np.nan
    return out


def generate_multi(returns: np.ndarray, close: np.ndarray, pdf: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lms_raw = generate_trend(close)
    raw = factors.compute_raw_factors(pdf, lms_raw)
    composite, factor_z = factors.fuse_factors(raw)
    composite = _apply_factor_agree(composite, raw)
    return composite, factor_z


def generate(
    returns: np.ndarray,
    close: np.ndarray | None = None,
    pdf: pd.DataFrame | None = None,
    mode: str = config.SIGNAL_MODE,
) -> np.ndarray:
    if mode == "multi":
        if close is None or pdf is None:
            raise ValueError("multi mode requires close and ohlcv dataframe")
        composite, _ = generate_multi(returns, close, pdf)
        return composite
    if mode == "trend":
        if close is None:
            raise ValueError("trend mode requires close array")
        return generate_trend(close)
    return generate_return(returns)


def build_signal_df() -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """产出信号表；multi 模式附带各因子 z-score 列。"""
    pdf = data.load_dataframe()
    dt = pdf["datetime"].to_numpy()
    close = pdf["close"].to_numpy(dtype=float)

    if config.RETURN_MODE == "log":
        returns = np.diff(np.log(close), prepend=np.log(close[0]))
    else:
        returns = np.diff(close, prepend=close[0]) / np.concatenate([[close[0]], close[:-1]])
    returns[0] = 0.0

    mode = config.SIGNAL_MODE
    if mode == "multi" or (config.USE_MULTI_FACTOR and mode != "return"):
        zsig, factor_z = generate_multi(returns, close, pdf)
        extra = {f"z_{k}": v for k, v in factor_z.items()}
    else:
        yhat = generate(returns, close, pdf, mode=mode)
        zsig = factors.zscore_causal(
            yhat, min_periods=config.FACTOR_ZSCORE_MIN, smooth=config.SIGNAL_SMOOTH,
        )
        extra = {}

    mask = ~np.isnan(zsig)
    payload = {
        "datetime": pl.Series(dt[mask]).cast(pl.Datetime("us")),
        "vt_symbol": [config.VT_SYMBOL] * int(mask.sum()),
        "signal": zsig[mask].astype(float),
    }
    for col, arr in extra.items():
        payload[col] = arr[mask].astype(float)

    signal_df = pl.DataFrame(payload).sort("datetime")
    return signal_df, dt, close, returns


if __name__ == "__main__":
    sig, dt, close, returns = build_signal_df()
    idx = np.where(~np.isnan(sig["signal"].to_numpy()))[0]
    print(f"模式: {config.SIGNAL_MODE} | 有效信号 {len(idx)} 根")
    print(sig.tail(3))
