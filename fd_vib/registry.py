"""统一因子注册表 —— 四梯队"""
from __future__ import annotations

import pandas as pd
import numpy as np

import academic
import commodity
import fundamental
import glass
import gtja
import research
import speculator
import vibe

TIERS: dict[str, dict] = {
    "tier1": {"label": "商品基本面", "module": fundamental, "prefixes": ("fund_",)},
    "tier2": {"label": "商品学术", "module": commodity, "prefixes": ("comm_",)},
    "tier3": {
        "label": "GitHub因子库",
        "modules": [gtja, academic, vibe],
        "prefixes": ("gtja_", "acad_", "alpha_", "vibe_"),
    },
    "tier4": {"label": "扩展研究", "module": research, "prefixes": ("res_",)},
    "tier5": {"label": "投机专用", "module": speculator, "prefixes": ("spec_",)},
    "tier6": {"label": "玻璃专项", "module": glass, "prefixes": ("glass_",)},
}


def _all_modules():
    seen = set()
    for info in TIERS.values():
        if "module" in info:
            yield info["module"]
            seen.add(id(info["module"]))
        for mod in info.get("modules", []):
            if id(mod) not in seen:
                yield mod
                seen.add(id(mod))


def all_factor_names() -> list[str]:
    names: list[str] = []
    for mod in _all_modules():
        names.extend(mod.all_factors().keys())
    return names


def tier_of(name: str) -> str:
    for tid, info in TIERS.items():
        for p in info["prefixes"]:
            if name.startswith(p):
                return tid
    return "other"


def compute_panel(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    names = names or all_factor_names()
    by_mod: dict = {}
    for mod in _all_modules():
        reg = mod.all_factors()
        picked = [n for n in names if n in reg]
        if picked:
            by_mod[mod] = picked
    parts = []
    for mod, picked in by_mod.items():
        parts.append(mod.compute_panel(df, picked))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def top_by_tier(rows: list[dict], n: int = 4) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {tid: [] for tid in TIERS}
    for r in rows:
        tid = tier_of(r["name"])
        if tid not in out:
            continue
        if not np.isfinite(r.get("abs_ic", float("nan"))):
            continue
        if len(out[tid]) >= n:
            continue
        out[tid].append(r["name"])
    return out


def integrated_factors(
    by_tier: dict[str, list[str]] | None = None,
    n_per_tier: int = 3,
    ic_rows: list[dict] | None = None,
) -> list[str]:
    """每梯队取 top-N，合并为四梯队整合因子列表"""
    if by_tier is None:
        if ic_rows is None:
            return []
        by_tier = top_by_tier(ic_rows, n=n_per_tier)
    names: list[str] = []
    for tid in TIERS:
        for n in by_tier.get(tid, [])[:n_per_tier]:
            if n not in names:
                names.append(n)
    return names


def integrated_weights(
    ic_rows: list[dict],
    factors: list[str] | None = None,
    tier_balance: float = 0.25,
) -> dict[str, float]:
    """梯队等权 × 梯队内 IC 加权"""
    import numpy as np

    ic_map = {r["name"]: r for r in ic_rows}
    by_tier = top_by_tier(ic_rows, n=6)
    picks = factors or integrated_factors(by_tier=by_tier)
    tier_of_pick: dict[str, str] = {n: tier_of(n) for n in picks}

    tier_names: dict[str, list[str]] = {tid: [] for tid in TIERS}
    for n in picks:
        tid = tier_of_pick.get(n, "")
        if tid in tier_names:
            tier_names[tid].append(n)

    weights: dict[str, float] = {}
    for tid, names in tier_names.items():
        if not names:
            continue
        raw = {}
        for n in names:
            r = ic_map.get(n, {})
            raw[n] = float(r.get("abs_ic", 0.02)) * float(r.get("sign", 1))
        s = sum(abs(v) for v in raw.values()) or 1.0
        for n, v in raw.items():
            weights[n] = tier_balance * v / s

    total = sum(abs(v) for v in weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}
