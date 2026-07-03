"""
信号生成 —— 走窗在线预测（严格无未来函数）

两种模式（由 config.SIGNAL_MODE 切换）：
  · "return" —— 用归一化收益预测下一根收益（自相关）；日线弱，不推荐。
  · "trend"  —— 自适应线性预测价格，signal = (预测价 - 现价)/现价，低滞后趋势跟随。

对每一根 K 线 t：
  1. 用截至 t 的数据预测 t+1 的目标（收益/价格）-> yhat[t]；
  2. 待 t+1 真实值已知后，再在线更新滤波器权重。
因此 yhat[t] 只依赖 t 及之前，对应"T 收盘出信号 -> T+1 成交"。

输出 polars 信号表 [datetime, vt_symbol, signal]。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import config
import data
from lms_filter import LmsEnsemble


def generate_return(
    returns: np.ndarray,
    orders: tuple[int, ...] = config.FILTER_ORDERS,
    mu: float = config.MU,
    eps: float = config.EPS,
    leak: float = config.LEAK,
    vol_window: int = config.VOL_WINDOW,
) -> np.ndarray:
    """
    模式 "return"：用归一化收益预测下一根收益。
    返回 yhat[t] = 对 returns[t+1] 的预测（归一化）。
    """
    n = len(returns)
    yhat = np.full(n, np.nan)

    # 因果波动率归一化
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
    """
    模式 "trend"：自适应预测价格，signal[t] = (预测价[t+1] - close[t]) / close[t]。
    输入向量为过去 order 根的价格（非差分），目标为 close[t+1]。
    NLMS 天然应对尺度（归一到输入能量），直接用价格可以，但更稳健的是用归一化价格。
    """
    n = len(close)
    yhat = np.full(n, np.nan)

    # 对价格做滚动归一（除以自身均值），尺度不变但保留趋势
    s = pd.Series(close)
    ma = s.rolling(max(orders), min_periods=1).mean().to_numpy()
    pnorm = close / (ma + eps)

    ens = LmsEnsemble(orders=orders, mu=mu, eps=eps, leak=leak)
    start = max(orders) + 1

    for t in range(start, n - 1):
        recent = pnorm[t - ens.max_order + 1: t + 1]
        pred_norm = ens.predict(recent)              # 预测 pnorm[t+1]
        pred_price = pred_norm * ma[t]               # 反归一到价格
        yhat[t] = (pred_price - close[t]) / close[t] # 相对变化
        ens.adapt(recent, pnorm[t + 1])
    return yhat


def generate(
    returns: np.ndarray,
    close: np.ndarray | None = None,
    mode: str = config.SIGNAL_MODE,
) -> np.ndarray:
    """统一入口，按 mode 选分支。"""
    if mode == "trend":
        if close is None:
            raise ValueError("trend mode requires close array")
        return generate_trend(close)
    else:
        return generate_return(returns)


def _postprocess(yhat: np.ndarray) -> np.ndarray:
    """
    信号后处理（严格因果）：
      1. 用扩展窗（只含历史）的均值/标准差做 z-score，统一尺度；
      2. EMA 平滑（span=config.SIGNAL_SMOOTH）去噪、降换手。
    返回 z-score 化并平滑后的信号（NaN 位置保持 NaN）。
    """
    s = pd.Series(yhat)
    mask = s.notna()

    z = pd.Series(np.nan, index=s.index)
    valid = s[mask]
    mu = valid.expanding(min_periods=20).mean()
    sd = valid.expanding(min_periods=20).std()
    z_valid = (valid - mu) / (sd + 1e-12)
    z_valid = z_valid.ewm(span=config.SIGNAL_SMOOTH, adjust=False).mean()
    z.loc[mask] = z_valid
    return z.to_numpy()


def build_signal_df() -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """产出信号表及原始数组，供回测/可视化复用。signal 为 z-score 化并平滑的预测。"""
    dt, close, returns = data.load_arrays()
    yhat = generate(returns, close, mode=config.SIGNAL_MODE)
    zsig = _postprocess(yhat)

    mask = ~np.isnan(zsig)
    signal_df = pl.DataFrame({
        "datetime": pl.Series(dt[mask]).cast(pl.Datetime("us")),
        "vt_symbol": [config.VT_SYMBOL] * int(mask.sum()),
        "signal": zsig[mask].astype(float),
    }).sort("datetime")
    return signal_df, dt, close, returns


if __name__ == "__main__":
    sig, dt, close, returns = build_signal_df()
    y = generate(returns, close, mode=config.SIGNAL_MODE)
    idx = np.where(~np.isnan(y))[0]
    idx = idx[idx < len(returns) - 1]

    if config.SIGNAL_MODE == "return":
        pred_dir = np.sign(y[idx])
        real_dir = np.sign(returns[idx + 1])
        hit = np.mean(pred_dir == real_dir)
        ic = np.corrcoef(y[idx], returns[idx + 1])[0, 1]
        print(f"模式: return | 样本 {len(idx)} | 方向命中率 {hit:.3f} | IC {ic:+.4f}")
    else:
        pred_dir = np.sign(y[idx])
        real_ret = (close[idx + 1] - close[idx]) / close[idx]
        hit = np.mean(pred_dir == np.sign(real_ret))
        ic = np.corrcoef(y[idx], real_ret)[0, 1]
        print(f"模式: trend | 样本 {len(idx)} | 方向命中率 {hit:.3f} | IC {ic:+.4f}")
    print(sig.tail(3))
