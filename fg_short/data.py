"""行情加载。"""
from __future__ import annotations

import pandas as pd
import polars as pl
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

import config


def load_polars_df() -> pl.DataFrame:
    pdf = pd.read_parquet(config.DATA_FILE)
    pdf = pdf.dropna(subset=["open", "high", "low", "close"])
    pdf = pdf.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    pdf["vwap"] = (pdf["high"] + pdf["low"] + pdf["close"]) / 3.0
    df = pl.from_pandas(
        pdf[["datetime", "open", "high", "low", "close", "volume", "open_interest", "vwap"]]
    )
    return df.with_columns(
        pl.col("datetime").cast(pl.Datetime("us")),
        pl.lit(config.VT_SYMBOL).alias("vt_symbol"),
    ).sort("datetime")


def build_bar_data() -> list[BarData]:
    pdf = pd.read_parquet(config.DATA_FILE).drop_duplicates("datetime").sort_values("datetime")
    exchange = Exchange(config.EXCHANGE_STR)
    bars: list[BarData] = []
    for row in pdf.itertuples(index=False):
        bars.append(
            BarData(
                symbol=config.SYMBOL,
                exchange=exchange,
                datetime=row.datetime.to_pydatetime(),
                interval=Interval.DAILY,
                open_price=float(row.open),
                high_price=float(row.high),
                low_price=float(row.low),
                close_price=float(row.close),
                volume=float(row.volume),
                turnover=float(getattr(row, "turnover", 0) or 0),
                open_interest=float(row.open_interest),
                gateway_name="DB",
            )
        )
    return bars
