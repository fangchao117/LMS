"""
数据加载 —— 把原始 parquet 转成两种下游需要的形态：

1. load_polars_df()  -> polars.DataFrame   给 AlphaDataset 做因子工程
   要求列：datetime, vt_symbol, open, high, low, close, volume, vwap ...
2. build_bar_data()  -> list[BarData]      给 AlphaLab / 回测引擎回放行情

注意：原始 close 是主力连续的后复权价，而 turnover 是未复权口径，
两者尺度不一致，故 vwap 不用 turnover/volume 反推，改用典型价 (H+L+C)/3，
与复权价尺度一致，仅用于 vwap/close 这类比值因子。
"""
from __future__ import annotations

import polars as pl
import pandas as pd

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval

import config


def _read_raw() -> pd.DataFrame:
    """读取原始 parquet，按时间排序去重。"""
    df: pd.DataFrame = pd.read_parquet(config.DATA_FILE)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.reset_index(drop=True)
    return df


def load_polars_df() -> pl.DataFrame:
    """构造 AlphaDataset 输入：带 datetime / vt_symbol 索引列的宽表。"""
    pdf: pd.DataFrame = _read_raw()

    # 典型价近似 vwap，尺度与复权收盘一致
    pdf["vwap"] = (pdf["high"] + pdf["low"] + pdf["close"]) / 3.0

    df: pl.DataFrame = pl.from_pandas(
        pdf[["datetime", "open", "high", "low", "close",
             "volume", "turnover", "open_interest", "vwap"]]
    )

    # AlphaDataset 以 (datetime, vt_symbol) 为主键
    df = df.with_columns(
        pl.col("datetime").cast(pl.Datetime("us")),
        pl.lit(config.VT_SYMBOL).alias("vt_symbol"),
    )

    # 把索引列放到最前，特征列在后
    order: list[str] = ["datetime", "vt_symbol", "open", "high", "low",
                        "close", "volume", "turnover", "open_interest", "vwap"]
    return df.select(order).sort("datetime")


def build_bar_data() -> list[BarData]:
    """构造回测行情：list[BarData]，日线。"""
    pdf: pd.DataFrame = _read_raw()
    exchange: Exchange = Exchange(config.EXCHANGE_STR)

    bars: list[BarData] = []
    for row in pdf.itertuples(index=False):
        bar: BarData = BarData(
            symbol=config.SYMBOL,
            exchange=exchange,
            datetime=row.datetime.to_pydatetime(),
            interval=Interval.DAILY,
            open_price=float(row.open),
            high_price=float(row.high),
            low_price=float(row.low),
            close_price=float(row.close),
            volume=float(row.volume),
            turnover=float(row.turnover),
            open_interest=float(row.open_interest),
            gateway_name="DB",
        )
        bars.append(bar)
    return bars


if __name__ == "__main__":
    df = load_polars_df()
    print("polars df:", df.shape)
    print(df.head())
    bars = build_bar_data()
    print(f"bars: {len(bars)}  {bars[0].datetime} -> {bars[-1].datetime}")
