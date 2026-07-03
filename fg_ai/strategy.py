"""
AI 多空择时策略 —— GlassAlphaStrategy

消费 LightGBM 经因果 z-score 标准化后的预测（signal），做单标的多空择时：

    pred >  threshold  -> 满仓做多
    pred < -threshold  -> 满仓做空
    |pred| <= threshold -> 空仓

可选三均线趋势过滤（USE_REGIME_FILTER）：
    仅在 lms_regime 与 AI 方向一致时开仓，震荡市（regime=0）空仓。

撮合时序由 vnpy.alpha 回测引擎保证：T 日收盘 on_bars 产生目标仓位并挂单，
订单在 T+1 日 K 线撮合成交，与标签口径严格对齐，无未来函数。
"""
from __future__ import annotations

from vnpy.trader.object import BarData, TradeData
from vnpy.alpha import AlphaStrategy

import config


SIGNAL_COL: str = "signal"


class GlassAlphaStrategy(AlphaStrategy):
    """玻璃期货 AI 多空择时。"""

    signal_threshold: float = config.SIGNAL_THRESHOLD
    position_pct: float = config.POSITION_PCT
    price_add_ticks: int = config.PRICE_ADD_TICKS
    use_regime_filter: bool = config.USE_REGIME_FILTER

    def on_init(self) -> None:
        mode = "趋势过滤+AI" if self.use_regime_filter else "纯 AI"
        self.write_log(f"玻璃 AI 择时策略初始化（{mode}，阈值={self.signal_threshold}）")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        bar: BarData | None = bars.get(config.VT_SYMBOL)
        if bar is None:
            return

        signal_df = self.get_signal()
        if signal_df.is_empty():
            return

        pred: float = float(signal_df[SIGNAL_COL][0])
        if np_isnan(pred):
            return

        regime: int = 0
        if self.use_regime_filter and config.LMS_REGIME_COL in signal_df.columns:
            regime = int(signal_df[config.LMS_REGIME_COL][0])

        direction: int = 0
        if pred > self.signal_threshold:
            direction = 1
        elif pred < -self.signal_threshold:
            direction = -1

        if self.use_regime_filter and direction != 0:
            if regime == 0 or regime != direction:
                direction = 0

        target_lots: int = 0
        if direction != 0:
            portfolio_value: float = self.get_portfolio_value()
            notional_per_lot: float = bar.close_price * config.CONTRACT_SIZE
            if notional_per_lot > 0:
                lots: int = int(portfolio_value * self.position_pct / notional_per_lot)
                target_lots = direction * max(lots, 0)

        self.set_target(config.VT_SYMBOL, float(target_lots))

        price_add: float = 0.0
        if bar.close_price > 0:
            price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price

        self.execute_trading(bars, price_add)

    def on_trade(self, trade: TradeData) -> None:
        pass


def np_isnan(v: float) -> bool:
    return v != v
