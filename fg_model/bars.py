"""加载训练用 K 线：主力拼接 / 东财 GM / 自动选择。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import config

_TRAIN_COLS = (
    "datetime", "open", "high", "low", "close",
    "volume", "open_interest", "vt_symbol",
)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.tz_convert(None)
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "open_interest" not in out.columns and "position" in out.columns:
        out["open_interest"] = pd.to_numeric(out["position"], errors="coerce")
    if "open_interest" in out.columns:
        out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce").fillna(0)
    if "vt_symbol" not in out.columns:
        out["vt_symbol"] = "FG609.CZCE"
    out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return out.reset_index(drop=True)


def load_fg_bars(adjust: bool = False) -> pd.DataFrame:
    """fd_vib 主力拼接 30m（默认训练源）。"""
    fd = config.DATA_ROOT / "fd_vib"
    if str(fd) not in sys.path:
        sys.path.insert(0, str(fd))
    from dominant import load_dominant_30m

    df = load_dominant_30m(adjust=adjust)
    if "datetime" not in df.columns:
        df = df.reset_index()
    return _normalize_df(df)


def load_gm_parquet(path: Path | None = None) -> pd.DataFrame:
    """读取东财 GM 下载的 parquet。"""
    path = path or config.TRAIN_BARS_PATH
    if not path.exists():
        # 回退：最新 bars_30m_gm_*.parquet
        candidates = sorted((config.DATA_DIR).glob("bars_30m_gm_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"未找到 GM 数据。请先: python download_gm.py --symbol FG609"
            )
        path = candidates[-1]
    df = pd.read_parquet(path)
    return _normalize_df(df)


def load_train_cache() -> pd.DataFrame:
    """import_gm_to_train 写入的标准训练缓存。"""
    if not config.TRAIN_BARS_PATH.exists():
        raise FileNotFoundError(
            f"无训练缓存 {config.TRAIN_BARS_PATH}，请先: python import_gm_to_train.py"
        )
    return _normalize_df(pd.read_parquet(config.TRAIN_BARS_PATH))


def apply_train_overrides() -> dict | None:
    """读取 train_source.json，临时覆盖 WF 参数（仅 fg_model）。"""
    if not config.TRAIN_SOURCE_META.exists():
        return None
    meta = json.loads(config.TRAIN_SOURCE_META.read_text(encoding="utf-8"))
    if meta.get("wf_oos_start"):
        config.WF_OOS_START = meta["wf_oos_start"]
    if meta.get("wf_min_train_bars"):
        config.WF_MIN_TRAIN_BARS = int(meta["wf_min_train_bars"])
    if meta.get("train_end"):
        config.TRAIN_END = meta["train_end"]
    if meta.get("valid_end"):
        config.VALID_END = meta["valid_end"]
    if meta.get("test_start"):
        config.TEST_START = meta["test_start"]
    return meta


def load_bars(source: str = "auto") -> pd.DataFrame:
    """
    source:
      auto     — 有 train_bars_30m.parquet 则用 GM 缓存，否则主力拼接
      gm       — 东财训练缓存（须先 import_gm_to_train）
      gm_raw   — 直接读 bars_30m_gm_*.parquet
      dominant — fd_vib 主力拼接
    """
    if source == "dominant":
        return load_fg_bars()
    if source == "gm_raw":
        return load_gm_parquet()
    if source == "gm":
        apply_train_overrides()
        return load_train_cache()
    if source == "auto":
        if config.TRAIN_BARS_PATH.exists():
            apply_train_overrides()
            return load_train_cache()
        return load_fg_bars()
    raise ValueError(f"未知 source={source}")
