"""因子组合信号"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import gtja
import glass
import registry
import config
from bridge import fg_generate_signal, load_daily
from ops import zscore_causal


def _load_ic_data() -> tuple[list[str], dict[str, float], dict[str, list[str]]]:
    import json
    import config

    p = config.ARTIFACT_PATH / "factor_ic.json"
    if not p.exists():
        return [], {}, {}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("all", [])
    by_tier = data.get("by_tier") or registry.top_by_tier(rows, n=3)
    factors = registry.integrated_factors(by_tier=by_tier, n_per_tier=3)
    weights = registry.integrated_weights(rows, factors)
    return factors, weights, by_tier


def _weights_for_factors(names: list[str], weights: dict[str, float] | None = None) -> dict[str, float]:
    if weights:
        return weights
    import json
    import config

    p = config.ARTIFACT_PATH / "factor_ic.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        ic_map = {r["name"]: r for r in data.get("all", [])}
        raw = {}
        for n in names:
            r = ic_map.get(n, {})
            raw[n] = float(r.get("abs_ic", 0.02)) * float(r.get("sign", 1))
        if sum(abs(v) for v in raw.values()) > 1e-9:
            s = sum(abs(v) for v in raw.values())
            return {k: v / s for k, v in raw.items()}
    return {n: 1.0 / len(names) for n in names}


def _load_ic_weights() -> dict[str, float]:
    factors, weights, _ = _load_ic_data()
    if weights:
        return weights
    return _weights_for_factors(factors or list(gtja.all_factors().keys())[:8])


def _factor_panel(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    if not names:
        return registry.compute_panel(df)
    return registry.compute_panel(df, names)


def _align_daily_filter(df: pd.DataFrame, sig: pd.Series) -> pd.Series:
    daily = load_daily(
        str(df["datetime"].min().date()),
        str(df["datetime"].max().date()),
    )
    c = daily["close"]
    e1 = c.ewm(span=26, adjust=False).mean()
    e2 = c.ewm(span=46, adjust=False).mean()
    ddir = np.where(e1 > e2, 1, np.where(e1 < e2, -1, 0))
    tbl = pd.DataFrame({"date": daily["datetime"].dt.date, "ddir": ddir})
    m = df.copy()
    m["date"] = m["datetime"].dt.date
    aligned = m.merge(tbl, on="date", how="left")["ddir"].ffill().fillna(0).astype(int)
    out = sig.copy()
    for i in range(len(out)):
        d = int(aligned.iloc[i])
        s = float(out.iloc[i])
        if d == 0 or (d > 0 and s < 0) or (d < 0 and s > 0):
            out.iloc[i] = 0.0
    return out


def _score_from_panel(
    panel: pd.DataFrame,
    wmap: dict[str, float],
    smooth: int,
) -> np.ndarray:
    score = np.zeros(len(panel))
    for col in panel.columns:
        wt = wmap.get(col, 0.0)
        if wt == 0:
            continue
        z = zscore_causal(panel[col].astype(float))
        if smooth > 0:
            z = z.rolling(smooth, min_periods=1).mean()
        score += wt * z.to_numpy()
    return score


def generate(
    df: pd.DataFrame,
    mode: str = "gtja_combo",
    factors: list[str] | None = None,
    weights: dict[str, float] | None = None,
    threshold: float = 0.35,
    smooth: int = 3,
    daily_filter: bool = True,
    donchian: int = 38,
    tier_per: int = 3,
    **kwargs: Any,
) -> pd.Series:
    if mode == "donchian_baseline":
        sig = fg_generate_signal(df, "breakout", donchian=donchian)
        return sig.replace(0, np.nan).ffill().fillna(0.0)

    if mode == "don_enhanced":
        """Don 主信号 + 全因子 IC 加权确认 + 震荡/高波过滤（降回撤）"""
        don = generate(df, "donchian_baseline", donchian=donchian)
        names = factors
        if not names:
            import json
            p = config.ARTIFACT_PATH / "factor_ic.json"
            if p.exists():
                rows = json.loads(p.read_text(encoding="utf-8")).get("all", [])
                names = [r["name"] for r in rows[:12] if r.get("abs_ic", 0) >= 0.02]
            if not names:
                names = registry.integrated_factors(n_per_tier=2)
        panel = _factor_panel(df, names)
        wmap = _weights_for_factors(list(panel.columns), weights)
        score = _score_from_panel(panel, wmap, smooth)
        chop = glass.glass_chop_filter(df).astype(float).to_numpy()
        vol_r = glass.glass_vol_regime(df).astype(float).to_numpy()
        chop_max = float(kwargs.get("chop_max", 0.50))
        vol_min = float(kwargs.get("vol_min", -1.2))
        strict = kwargs.get("strict", True)
        out = np.zeros(len(df))
        for i in range(len(df)):
            d = float(don.iloc[i])
            s = float(score[i])
            if chop[i] > chop_max or vol_r[i] < vol_min:
                out[i] = 0.0
                continue
            if d > 0 and s > threshold * 0.45:
                out[i] = 1.0
            elif d < 0 and s < -threshold * 0.45:
                out[i] = -1.0
            elif not strict and d != 0 and abs(s) < threshold * 0.12:
                out[i] = d
        sig = pd.Series(out, index=df.index)
        return _align_daily_filter(df, sig) if daily_filter else sig

    if mode in ("four_tier_combo", "four_tier_don", "four_tier_don_filter"):
        int_factors, int_weights, _ = _load_ic_data()
        names = factors or int_factors
        wmap = weights or int_weights
        if not names:
            names = list(registry.integrated_factors(n_per_tier=tier_per))
            wmap = registry.integrated_weights(
                __import__("json").loads(
                    (__import__("config").ARTIFACT_PATH / "factor_ic.json").read_text(encoding="utf-8")
                ).get("all", []),
                names,
            ) if names else {}
        panel = _factor_panel(df, names)
        if not wmap:
            wmap = {c: 1.0 / max(len(panel.columns), 1) for c in panel.columns}
        score = _score_from_panel(panel, wmap, smooth)

        if mode == "four_tier_combo":
            sig = pd.Series(
                np.where(score > threshold, 1.0, np.where(score < -threshold, -1.0, 0.0)),
                index=df.index,
            )
            return _align_daily_filter(df, sig) if daily_filter else sig

        don = generate(df, "donchian_baseline", donchian=donchian)
        out = np.zeros(len(df))
        for i in range(len(df)):
            d = float(don.iloc[i])
            s = float(score[i])
            if mode == "four_tier_don":
                if d > 0 and s > 0:
                    out[i] = 1.0
                elif d < 0 and s < 0:
                    out[i] = -1.0
            else:
                if d > 0 and s > threshold * 0.5:
                    out[i] = 1.0
                elif d < 0 and s < -threshold * 0.5:
                    out[i] = -1.0
                elif d != 0 and abs(s) < threshold * 0.15:
                    out[i] = d
                else:
                    out[i] = 0.0
        sig = pd.Series(out, index=df.index)
        return _align_daily_filter(df, sig) if daily_filter else sig

    if mode == "gtja_don":
        base = generate(df, "gtja_combo", factors=factors, weights=weights,
                        threshold=threshold, smooth=smooth, daily_filter=False)
        don = generate(df, "donchian_baseline", donchian=donchian)
        out = np.zeros(len(df))
        for i in range(len(df)):
            a, b = base.iloc[i], don.iloc[i]
            if a > 0 and b > 0:
                out[i] = 1.0
            elif a < 0 and b < 0:
                out[i] = -1.0
        sig = pd.Series(out, index=df.index)
        return _align_daily_filter(df, sig) if daily_filter else sig

    if mode == "don_factor_filter":
        """唐奇安主信号 + 因子方向确认（推荐）"""
        don = generate(df, "donchian_baseline", donchian=donchian)
        names = factors
        if not names:
            w = weights or _load_ic_weights()
            names = list(w.keys()) if w else list(gtja.all_factors().keys())[:8]
        panel = _factor_panel(df, names)
        wmap = _weights_for_factors(list(panel.columns), weights)
        score = _score_from_panel(panel, wmap, smooth)
        out = np.zeros(len(df))
        for i in range(len(df)):
            d = float(don.iloc[i])
            s = float(score[i])
            if d > 0 and s > threshold * 0.5:
                out[i] = 1.0
            elif d < 0 and s < -threshold * 0.5:
                out[i] = -1.0
            elif d != 0 and abs(s) < threshold * 0.15:
                out[i] = d  # 因子中性时不拦唐奇安
            else:
                out[i] = 0.0
        sig = pd.Series(out, index=df.index)
        return _align_daily_filter(df, sig) if daily_filter else sig

    names = factors
    if not names:
        w = weights or _load_ic_weights()
        names = list(w.keys()) if w else list(gtja.all_factors().keys())[:8]

    panel = _factor_panel(df, names)
    wmap = _weights_for_factors(list(panel.columns), weights)

    score = _score_from_panel(panel, wmap, smooth)

    sig = pd.Series(
        np.where(score > threshold, 1.0, np.where(score < -threshold, -1.0, 0.0)),
        index=df.index,
    )
    if daily_filter:
        sig = _align_daily_filter(df, sig)
    return sig
