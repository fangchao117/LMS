"""
LMS 战法 —— 规则层：长/中/短三均线趋势判定

在完整历史价格上计算三条均线，输出每根 K 线的趋势状态 regime：
    +1  多头排列（ma_S > ma_M > ma_L 且 收盘 > ma_M）    -> 允许做多
    -1  空头排列（ma_S < ma_M < ma_L 且 收盘 < ma_M）    -> 允许做空
     0  均线缠绕 / 震荡                                    -> 空仓

在全历史上算完再截取回测区间，避免回测起点的均线预热缺口。

⚠️ 这是对 "LMS" 的通用解读。若你手上有博主的确切进出场规则，
   只需改本文件 _classify() 里的条件即可，下游策略/回测无需改动。
"""
from __future__ import annotations

import polars as pl

import config


def compute_lms_regime(
    price_df: pl.DataFrame,
    short: int = config.LMS_SHORT,
    medium: int = config.LMS_MEDIUM,
    long: int = config.LMS_LONG,
) -> pl.DataFrame:
    """
    输入：含 datetime / vt_symbol / close 的价格宽表（全历史）。
    输出：datetime / vt_symbol / lms_regime / ma_s / ma_m / ma_l。
    """
    df = price_df.sort("datetime").with_columns(
        pl.col("close").rolling_mean(short).alias("ma_s"),
        pl.col("close").rolling_mean(medium).alias("ma_m"),
        pl.col("close").rolling_mean(long).alias("ma_l"),
    )

    long_cond = (
        (pl.col("ma_s") > pl.col("ma_m"))
        & (pl.col("ma_m") > pl.col("ma_l"))
        & (pl.col("close") > pl.col("ma_m"))
    )
    short_cond = (
        (pl.col("ma_s") < pl.col("ma_m"))
        & (pl.col("ma_m") < pl.col("ma_l"))
        & (pl.col("close") < pl.col("ma_m"))
    )

    df = df.with_columns(
        pl.when(long_cond).then(1)
        .when(short_cond).then(-1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias(config.LMS_REGIME_COL)
    )

    return df.select(
        ["datetime", "vt_symbol", config.LMS_REGIME_COL, "ma_s", "ma_m", "ma_l"]
    )


def build_signal_df(
    price_df: pl.DataFrame,
    model_signal: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    组装回测用信号表：
      - 始终包含 lms_regime（规则方向）
      - 若给定 model_signal（含 'signal' 列），按 datetime 左连接并入 AI 预测
    """
    regime = compute_lms_regime(price_df)

    if model_signal is not None:
        regime = regime.join(
            model_signal.select(["datetime", "signal"]),
            on="datetime",
            how="left",
        )
    return regime


if __name__ == "__main__":
    import data_loader

    reg = compute_lms_regime(data_loader.load_polars_df())
    counts = reg[config.LMS_REGIME_COL].value_counts().sort(config.LMS_REGIME_COL)
    print("regime 分布（-1空/0震荡/+1多）：")
    print(counts)
