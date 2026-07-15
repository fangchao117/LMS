"""只读接入 fd_vib 因子库（不修改 fd_vib 任何文件）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from features import build_label, spearman_ic

_FD_VIB = config.DATA_ROOT / "fd_vib"
_IC_PATH = _FD_VIB / "artifacts" / "factor_ic.json"


def _zscore_causal(s: pd.Series, min_periods: int = 40) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - mu) / (sd + 1e-12)


def _import_registry():
    """隔离导入 fd_vib.registry，避免污染 fg_model 模块名。"""
    fd = str(_FD_VIB)
    if fd not in sys.path:
        sys.path.insert(0, fd)
    import registry as fd_registry  # noqa: WPS433

    return fd_registry


def load_ic_top_names(top_n: int | None = None) -> list[str]:
    top_n = top_n or config.FD_VIB_TOP_N
    if not _IC_PATH.exists():
        return []
    data = json.loads(_IC_PATH.read_text(encoding="utf-8"))
    names: list[str] = []
    for n in data.get("top", []):
        if n not in names:
            names.append(n)
    for tier_list in (data.get("by_tier") or {}).values():
        for n in tier_list:
            if n not in names:
                names.append(n)
    return names[:top_n]


def screen_factors_local(df: pd.DataFrame, top_n: int) -> list[str]:
    """无 factor_ic.json 时在内存里筛选，不写 fd_vib。"""
    reg = _import_registry()
    panel = reg.compute_panel(df)
    label = build_label(df)
    rows: list[tuple[str, float]] = []
    for col in panel.columns:
        ic = abs(spearman_ic(panel[col].astype(float), label))
        if ic > 0:
            rows.append((col, ic))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in rows[:top_n]]


def resolve_factor_names(df: pd.DataFrame, top_n: int | None = None) -> list[str]:
    top_n = top_n or config.FD_VIB_TOP_N
    names = load_ic_top_names(top_n)
    if len(names) < min(8, top_n):
        names = screen_factors_local(df, top_n)
    reg = _import_registry()
    valid = [n for n in names if n in reg.all_factor_names()]
    return valid[:top_n]


def load_fd_vib_features(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """读取 fd_vib 因子面板，列名前缀 fv_。"""
    reg = _import_registry()
    names = resolve_factor_names(df, top_n)
    if not names:
        return pd.DataFrame(index=df.index)
    panel = reg.compute_panel(df, names)
    out = pd.DataFrame(index=df.index)
    for col in panel.columns:
        out[f"fv_{col}"] = _zscore_causal(panel[col].astype(float))
    return out.replace([np.inf, -np.inf], np.nan)


def factor_meta() -> dict[str, Any]:
    return {
        "source": str(_FD_VIB),
        "ic_json": str(_IC_PATH),
        "read_only": True,
        "top_n": config.FD_VIB_TOP_N,
        "names": load_ic_top_names(),
    }
