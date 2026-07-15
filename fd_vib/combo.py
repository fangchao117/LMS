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


def _daily_direction_series(df: pd.DataFrame) -> np.ndarray:
    daily = load_daily(
        str(df["datetime"].min().date()),
        str(df["datetime"].max().date()),
    )
    c = daily["close"].astype(float)
    e1 = c.ewm(span=26, adjust=False).mean()
    e2 = c.ewm(span=46, adjust=False).mean()
    ddir = np.where(e1 > e2, 1, np.where(e1 < e2, -1, 0))
    tbl = pd.DataFrame({"date": daily["datetime"].dt.date, "ddir": ddir})
    m = df.copy()
    m["date"] = m["datetime"].dt.date
    return m.merge(tbl, on="date", how="left")["ddir"].ffill().fillna(0).astype(int).to_numpy()


def _atr_series(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _adx_series(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).rolling(n, min_periods=1).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=h.index).rolling(n, min_periods=1).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.rolling(n, min_periods=1).mean()


def _bar_momentum_bundle(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """近端 30m K 线 + 量价 + 持仓 → 动能分量（因果，无未来）"""
    import commodity
    import fundamental as fund

    c = df["close"].astype(float)
    h, l = df["high"].astype(float), df["low"].astype(float)
    ret5 = c.pct_change(5).fillna(0).to_numpy()
    ret10 = c.pct_change(10).fillna(0).to_numpy()
    ret20 = c.pct_change(20).fillna(0).to_numpy()
    adx = _adx_series(h, l, c).fillna(0).to_numpy()
    atr = _atr_series(h, l, c).fillna(0)
    atr_exp = (atr / (atr.rolling(20, min_periods=1).mean() + 1e-12) - 1.0).fillna(0).to_numpy()
    tsmom = commodity.comm_tsmom_20d(df).fillna(0).to_numpy()
    vol_r = fund.fund_vol_ratio(df).fillna(0).to_numpy()
    oi_mom = fund.fund_oi_mom5(df).fillna(0).to_numpy()
    vp = fund.fund_vp_regime(df).fillna(0).to_numpy()
    return {
        "ret5": ret5,
        "ret10": ret10,
        "ret20": ret20,
        "adx": adx,
        "atr_exp": atr_exp,
        "tsmom": tsmom,
        "vol_r": vol_r,
        "oi_mom": oi_mom,
        "vp": vp,
    }


def _momentum_bundle(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """兼容旧名"""
    return _bar_momentum_bundle(df)


def _momentum_aligned_score(
    mom: dict[str, np.ndarray],
    chop: np.ndarray,
    direction: float,
    i: int,
) -> float:
    """给定 Don 方向，综合近端 K 线/量价/持仓动能，越高越支持开仓"""
    if direction == 0:
        return 0.0
    sgn = 1.0 if direction > 0 else -1.0
    price = 0.35 * np.tanh(mom["ret5"][i] * sgn * 40) + 0.15 * np.tanh(mom["ret10"][i] * sgn * 25)
    vol_part = 0.12 * np.clip(mom["vol_r"][i], -0.4, 1.2) + 0.10 * float(mom["vp"][i]) * sgn
    oi_part = 0.15 * np.tanh(mom["oi_mom"][i] * sgn * 25)
    trend = 0.08 * np.clip(mom["adx"][i] / 30.0, 0.0, 1.0)
    chop_ok = 0.05 * np.clip(1.0 - chop[i] / 0.55, 0.0, 1.0)
    return float(price + vol_part + oi_part + trend + chop_ok)


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


def _pick_don_on_warmup(
    warm_df: pd.DataFrame,
    candidates: tuple[int, ...] = (25, 30, 35, 38, 45),
    default: int = 38,
    fallback_if_neg: bool = True,
) -> int:
    """主力段换月后：用前 warm 根 30m 小窗回测，选最优 Don（因果，不含未来）"""
    if len(warm_df) < 80:
        return default
    from risk_bt import run as run_risk

    best_don, best_ret = default, -1e18
    for don in candidates:
        sig = generate(warm_df, mode="donchian_baseline", donchian=don)
        st, _ = run_risk(
            warm_df, sig, max_lots=2, atr_stop_mult=3.5,
            trail_atr_mult=4.0, max_loss_pct=6.0, handle_roll=False,
        )
        ret = float(st["total_return_pct"])
        if ret > best_ret:
            best_ret, best_don = ret, don
    # 震荡段 warm 窗常全负（如 FG605）：自校准 Don 易过拟合 → 回退 Don38 纯突破
    if fallback_if_neg and best_ret < 0:
        return default
    return best_don


def _iter_symbol_blocks(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "symbol" not in df.columns:
        return [("", df)]
    out: list[tuple[str, pd.DataFrame]] = []
    sym = df["symbol"].astype(str)
    change = sym.ne(sym.shift()).cumsum()
    for _, block in df.groupby(change, sort=False):
        s = str(block["symbol"].iloc[0])
        out.append((s, block.reset_index(drop=True)))
    return out


def _generate_don_auto(df: pd.DataFrame, warm_bars: int = 320, **kwargs: Any) -> pd.Series:
    """
    每主力段自动选 Don + 纯突破（震荡/趋势通吃，优于固定 lms）。
    warm_bars: 换月后用于校准的前 N 根 30m（默认约 20 天）。
    """
    candidates = tuple(kwargs.get("don_candidates", (25, 30, 35, 38, 45)))
    default_don = int(kwargs.get("donchian", 38))
    warm = int(kwargs.get("warm_bars", warm_bars))
    parts: list[pd.Series] = []
    for sym, block in _iter_symbol_blocks(df):
        if len(block) <= warm + 40:
            don = default_don
        else:
            don = _pick_don_on_warmup(block.iloc[:warm], candidates=candidates, default=default_don)
        sig = generate(block, mode="donchian_baseline", donchian=don)
        sig = sig.copy()
        sig.attrs["auto_don"] = don
        sig.attrs["symbol"] = sym
        parts.append(sig)
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)


def _pick_auto_or_regime(
    warm_df: pd.DataFrame,
    auto_kw: dict[str, Any],
    regime_kw: dict[str, Any],
) -> str:
    """换月 warm 窗：auto vs regime；双负选 regime（震荡段如 FG605）。"""
    from risk_bt import run as run_risk

    if len(warm_df) < 80:
        return "auto"
    scores: dict[str, float] = {}
    for name, kw in (("auto", auto_kw), ("regime", regime_kw)):
        st, _ = run_risk(
            warm_df,
            generate(warm_df, **kw),
            max_lots=2,
            atr_stop_mult=3.5,
            trail_atr_mult=4.0,
            max_loss_pct=6.0,
            handle_roll=False,
        )
        scores[name] = float(st["total_return_pct"])
    if scores["auto"] <= 0 and scores["regime"] <= 0:
        return "regime"
    if scores["regime"] > scores["auto"]:
        return "regime"
    return "auto"


def _generate_don_smart(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    """
    每主力段最大化：warm 窗在 auto（趋势自校准 Don）与 regime（震荡自适应）间自动切换。
    """
    warm = int(kwargs.get("warm_bars", 320))
    auto_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in ("use_risk", "max_lots", "atr_stop_mult", "trail_atr_mult", "max_loss_pct", "mode")
    }
    auto_kw["mode"] = "don_auto"
    regime_kw = {**auto_kw, "mode": "don_regime", "chop_hi": float(kwargs.get("chop_hi", 0.38))}

    parts: list[pd.Series] = []
    picks: list[str] = []
    for sym, block in _iter_symbol_blocks(df):
        if len(block) <= warm + 40:
            pick = "auto"
            sig = generate(block, **auto_kw)
        else:
            pick = _pick_auto_or_regime(block.iloc[:warm], auto_kw, regime_kw)
            kw = regime_kw if pick == "regime" else auto_kw
            sig = generate(block, **kw)
        sig = sig.copy()
        sig.attrs["segment_pick"] = pick
        sig.attrs["symbol"] = sym
        picks.append(pick)
        parts.append(sig)
    out = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    out.attrs["segment_picks"] = picks
    return out


def _generate_don_regime(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    """
    震荡/趋势自适应：chop 高 → 仅日线方向+Don（少假突破）；
    趋势段 → 每主力段 don_auto 自校准 Don 突破。
    """
    chop_hi = float(kwargs.get("chop_hi", 0.38))
    don = int(kwargs.get("donchian", 38))
    chop = glass.glass_chop_filter(df).astype(float).to_numpy()
    auto = _generate_don_auto(df, **kwargs)
    dir_sig = generate(df, mode="don_lms", donchian=don, require_daily=True, mom_mode="off")
    out = np.where(chop > chop_hi, dir_sig.to_numpy(), auto.to_numpy())
    # 与 don 突破一致：0 表示“无新信号”时保持仓位；震荡段日线过滤的 0 为真平仓
    sig = pd.Series(out, index=df.index)
    is_dir_flat = (chop > chop_hi) & (dir_sig.to_numpy() == 0)
    sig = sig.where(~is_dir_flat, 0.0)
    return sig.replace(0, np.nan).ffill().fillna(0.0)


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

    if mode == "don_auto":
        return _generate_don_auto(df, **kwargs)

    if mode == "don_regime":
        return _generate_don_regime(df, **kwargs)

    if mode == "don_smart":
        return _generate_don_smart(df, **kwargs)

    if mode == "don_lms":
        """
        L.M.S：Don=空间；日线 EMA=方向；近端 K 线+量价+持仓=动能（与方向同时生效）。
        mom_mode: off=仅方向 | score=方向+综合动能 | soft/strict=旧版硬阈值
        """
        don = generate(df, "donchian_baseline", donchian=donchian)
        if kwargs.get("require_daily", True):
            don = _align_daily_filter(df, don)
        mom_mode = str(kwargs.get("mom_mode", "score"))
        if mom_mode == "off":
            return don

        mom = _bar_momentum_bundle(df)
        chop = glass.glass_chop_filter(df).astype(float).to_numpy()
        adx_min = float(kwargs.get("adx_min", 15.0))
        chop_max = float(kwargs.get("chop_max", 0.52))
        vol_min = float(kwargs.get("vol_min", -0.45))
        mom_min = float(kwargs.get("mom_min", 0.12))
        out = np.zeros(len(df))
        prev = 0.0
        for i in range(len(df)):
            d = float(don.iloc[i])
            if d != 0 and mom_mode == "score":
                sc = _momentum_aligned_score(mom, chop, d, i)
                if d != prev and sc < mom_min:
                    d = prev if prev != 0 else 0.0
            elif d != 0 and mom_mode in ("soft", "strict"):
                weak = (
                    mom["adx"][i] < adx_min
                    or chop[i] > chop_max
                    or mom["vol_r"][i] < vol_min
                )
                if mom_mode == "strict":
                    tsm = mom["tsmom"][i]
                    tsm_ok = (d > 0 and tsm > 0) or (d < 0 and tsm < 0) or d == 0
                    oi_ok = (d > 0 and mom["oi_mom"][i] >= 0) or (d < 0 and mom["oi_mom"][i] <= 0) or d == 0
                    weak = weak or not tsm_ok or not oi_ok
                if weak and d != 0 and d != prev:
                    d = prev
                elif weak and d != 0 and prev == 0:
                    d = 0.0
            out[i] = d
            prev = d if d != 0 else prev
        return pd.Series(out, index=df.index)

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
