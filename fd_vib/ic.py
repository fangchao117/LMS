"""因子 IC 筛选 —— 时序 Spearman IC"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

import config
import registry
from bridge import load_30m
from dominant import load_dominant_30m


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.pct_change(horizon).shift(-horizon)


def spearman_ic(factor: pd.Series, fwd: pd.Series) -> float:
    valid = factor.notna() & fwd.notna() & np.isfinite(factor) & np.isfinite(fwd)
    if valid.sum() < 50:
        return 0.0
    return float(factor[valid].corr(fwd[valid], method="spearman"))


def screen_factors(
    df: pd.DataFrame | None = None,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    df = df if df is not None else load_dominant_30m()
    h = horizon or config.IC_HORIZON
    fwd = forward_return(df["close"].astype(float), h)

    panel = registry.compute_panel(df)
    rows: list[dict[str, Any]] = []
    for col in panel.columns:
        ic = spearman_ic(panel[col], fwd)
        tier = registry.tier_of(col)
        rows.append({
            "name": col,
            "tier": tier,
            "tier_label": registry.TIERS.get(tier, {}).get("label", tier),
            "ic": ic,
            "abs_ic": abs(ic),
            "sign": 1 if ic >= 0 else -1,
        })

    rows.sort(key=lambda x: x["abs_ic"], reverse=True)
    return rows


def top_factor_names(
    rows: list[dict[str, Any]] | None = None,
    n: int | None = None,
    min_abs: float | None = None,
) -> list[str]:
    n = n or config.TOP_FACTORS
    min_abs = min_abs if min_abs is not None else config.IC_MIN_ABS
    if rows is None:
        rows = screen_factors()
    out = []
    for r in rows:
        if not np.isfinite(r["abs_ic"]) or r["abs_ic"] < min_abs:
            continue
        out.append(r["name"])
        if len(out) >= n:
            break
    return out


def save_screen(path=None) -> dict:
    config.ensure_dirs()
    rows = screen_factors()
    top = top_factor_names(rows)
    by_tier = registry.top_by_tier(rows, n=6)
    out = {
        "horizon": config.IC_HORIZON,
        "period": list(config.BACKTEST_PERIOD),
        "n_factors": len(rows),
        "top": top,
        "by_tier": by_tier,
        "all": rows,
    }
    path = path or config.ARTIFACT_PATH / "factor_ic.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
