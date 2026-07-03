"""
玻璃期货因子数据集 —— GlassAlphaDataset

基于 vnpy.alpha 的表达式因子引擎（Qlib 风格 DSL），
为单标的商品期货精选一组稳健的量价因子（约 40 个），
相比全套 Alpha158（158 因子）更不易在 3000 根日线上过拟合。

因子分组：
  · K 线形态      —— 当日多空力量
  · 动量 / 均线   —— 趋势方向与强度
  · 波动 / 通道   —— 风险与位置
  · 摆动指标      —— RSI / 随机指标 / ATR
  · 量能 / 持仓   —— 资金与情绪确认

标签、预处理器（稳健 Z-Score 归一化 + 缺失处理）在 build_dataset() 中装配。
"""
from __future__ import annotations

from functools import partial

import polars as pl

from vnpy.alpha import AlphaDataset
from vnpy.alpha.dataset.processor import (
    process_robust_zscore_norm,
    process_fill_na,
    process_drop_na,
)

import config


# 时序窗口（交易日）—— 去掉 5 日减少共线性、降低过拟合
WINDOWS: list[int] = [10, 20, 60]
EPS: str = "1e-12"


class GlassAlphaDataset(AlphaDataset):
    """玻璃期货专用因子集。"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )

        # -------- K 线形态（以 open 归一，量纲无关）--------
        self.add_feature("kmid", "(close - open) / open")                       # 实体
        self.add_feature("klen", "(high - low) / open")                         # 全幅
        self.add_feature("kup", f"(high - ts_greater(open, close)) / open")     # 上影
        self.add_feature("klow", f"(ts_less(open, close) - low) / open")        # 下影
        self.add_feature("ksft", "(close * 2 - high - low) / open")             # 收盘偏移

        # -------- 动量：过去 w 日收益 --------
        for w in WINDOWS:
            self.add_feature(f"roc_{w}", f"close / ts_delay(close, {w}) - 1")

        # -------- 均线乖离：均线相对现价 --------
        for w in WINDOWS:
            self.add_feature(f"ma_{w}", f"ts_mean(close, {w}) / close - 1")

        # -------- 线性趋势斜率（归一化）--------
        for w in WINDOWS:
            self.add_feature(f"beta_{w}", f"ts_slope(close, {w}) / close")

        # -------- 波动率：收益标准差 --------
        for w in WINDOWS:
            self.add_feature(f"std_{w}", f"ts_std(close, {w}) / close")

        # -------- 通道位置（随机指标 RSV）--------
        for w in WINDOWS:
            self.add_feature(
                f"rsv_{w}",
                f"(close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + {EPS})",
            )

        # -------- 距区间高低点 --------
        for w in WINDOWS:
            self.add_feature(f"maxd_{w}", f"ts_max(high, {w}) / close - 1")
        for w in WINDOWS:
            self.add_feature(f"mind_{w}", f"ts_min(low, {w}) / close - 1")

        # -------- 摆动指标 RSI / ATR（保留中长周期）--------
        for w in [14, 24]:
            self.add_feature(f"rsi_{w}", f"ta_rsi(close, {w}) / 100")
        self.add_feature("atr_14", "ta_atr(high, low, close, 14) / close")

        # -------- 量能：成交量均线比、量价相关 --------
        for w in WINDOWS:
            self.add_feature(f"vma_{w}", f"ts_mean(volume, {w}) / (volume + {EPS})")
        for w in WINDOWS:
            self.add_feature(f"vcorr_{w}", f"ts_corr(close, ts_log(volume + 1), {w})")

        # -------- 持仓量变化 --------
        self.add_feature("oi_20", "open_interest / ts_delay(open_interest, 20) - 1")

        # -------- 标签：见 config.LABEL_EXPR --------
        self.set_label(config.LABEL_EXPR)


def build_dataset(df: pl.DataFrame, max_workers: int = 1) -> GlassAlphaDataset:
    """构造并装配预处理器，返回已 prepare 的数据集。

    max_workers=1 为低配机器的温和模式（单进程算因子）；
    机器性能好可传 None 让框架自动并行。
    """
    dataset: GlassAlphaDataset = GlassAlphaDataset(
        df=df,
        train_period=config.TRAIN_PERIOD,
        valid_period=config.VALID_PERIOD,
        test_period=config.TEST_PERIOD,
    )

    train_start, train_end = config.TRAIN_PERIOD

    # 推断链：稳健 Z-Score 归一化（仅用训练窗口统计量拟合，防止泄漏）+ 特征缺失填 0
    dataset.add_processor(
        "infer",
        partial(
            process_robust_zscore_norm,
            fit_start_time=train_start,
            fit_end_time=train_end,
            clip_outlier=True,
        ),
    )
    dataset.add_processor(
        "infer",
        partial(process_fill_na, fill_value=0.0, fill_label=False),
    )

    # 学习链：丢弃标签缺失的样本（尾部未来收益不可得的行）
    dataset.add_processor(
        "learn",
        partial(process_drop_na, names=["label"]),
    )

    dataset.prepare_data(max_workers=max_workers)
    return dataset


if __name__ == "__main__":
    import data_loader
    from vnpy.alpha import Segment

    ds = build_dataset(data_loader.load_polars_df())
    for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        learn = ds.fetch_learn(seg)
        print(f"{seg}: learn={learn.shape}  features={learn.width - 3}")
