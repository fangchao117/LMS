"""从 vnpy 数据库加载 1 分钟 K 线并合成 30 分钟。"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database

import config


def load_1m(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    prefer: str = "auto",
) -> pd.DataFrame:
    """prefer: auto=取数据更长者, parquet, vnpy"""
    sym = symbol or config.SYMBOL
    parquet_1m = config.ROOT / "data" / f"bars_1m_{sym}.parquet"

    def _from_parquet() -> pd.DataFrame:
        if not parquet_1m.exists():
            return pd.DataFrame()
        df = pd.read_parquet(parquet_1m)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def _from_vnpy() -> pd.DataFrame:
        db = get_database()
        st = datetime.strptime(start, "%Y-%m-%d") if start else datetime(2020, 1, 1)
        en = datetime.strptime(end, "%Y-%m-%d") if end else datetime(2030, 12, 31)
        bars = db.load_bar_data(sym, Exchange.CZCE, Interval.MINUTE, st, en)
        if not bars:
            return pd.DataFrame()
        rows = [
            {
                "datetime": b.datetime.replace(tzinfo=None),
                "open": b.open_price,
                "high": b.high_price,
                "low": b.low_price,
                "close": b.close_price,
                "volume": b.volume,
                "open_interest": b.open_interest,
            }
            for b in bars
        ]
        return pd.DataFrame(rows).sort_values("datetime").drop_duplicates("datetime")

    if prefer == "parquet":
        df = _from_parquet()
    elif prefer == "vnpy":
        df = _from_vnpy()
    else:
        df_p, df_v = _from_parquet(), _from_vnpy()
        df = df_v if len(df_v) >= len(df_p) else df_p

    if df.empty:
        raise RuntimeError(f"无 1 分钟数据: {sym}，请先 EDB 下载或 CTP 落地")

    if start:
        df = df[df["datetime"] >= pd.Timestamp(start)]
    if end:
        df = df[df["datetime"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def resample_bars(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    pdf = df_1m.set_index("datetime")
    bar = pdf.resample(f"{minutes}min", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
        }
    )
    bar = bar.dropna(subset=["close"]).reset_index()
    bar["vt_symbol"] = config.VT_SYMBOL
    return bar


def resample_30m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return resample_bars(df_1m, config.BAR_MINUTES)


def load_30m(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    sym = symbol or config.SYMBOL
    cache = config.ARTIFACT_PATH / f"bars_30m_{sym}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        df["datetime"] = pd.to_datetime(df["datetime"])
        if start:
            df = df[df["datetime"] >= pd.Timestamp(start)]
        if end:
            df = df[df["datetime"] <= pd.Timestamp(end)]
        if len(df) > 0:
            return df.reset_index(drop=True)

    df_1m = load_1m(sym, start, end)
    df = resample_30m(df_1m)
    if not start and not end:
        config.ensure_dirs()
        df.to_parquet(cache, index=False)
    return df


def load_bars(
    minutes: int,
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    sym = symbol or config.SYMBOL
    cache = config.ARTIFACT_PATH / f"bars_{minutes}m_{sym}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        df["datetime"] = pd.to_datetime(df["datetime"])
        if start:
            df = df[df["datetime"] >= pd.Timestamp(start)]
        if end:
            df = df[df["datetime"] <= pd.Timestamp(end)]
        if len(df) > 0:
            return df.reset_index(drop=True)

    df_1m = load_1m(sym, start, end)
    df = resample_bars(df_1m, minutes)
    if not start and not end:
        config.ensure_dirs()
        df.to_parquet(cache, index=False)
    return df


def load_60m(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    return load_bars(60, symbol, start, end, use_cache)
