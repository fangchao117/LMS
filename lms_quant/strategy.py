"""
LMS 自适应滤波策略（vnpy.alpha 回测框架）

信号 signal = 滤波器对下一根的预测（收益或相对价格变化）：
    signal >  阈值 -> 满仓做多
    signal < -阈值 -> 满仓做空
    |signal| ≤ 阈值 -> 空仓

撮合时序：T 收盘出信号 -> T+1 成交，无未来函数。
"""
from __future__ import annotations

from vnpy.trader.object import BarData, TradeData
from vnpy.alpha import AlphaStrategy

import config


class LmsFilterStrategy(AlphaStrategy):
    """最小均方自适应滤波择时。"""

    signal_threshold: float = config.SIGNAL_THRESHOLD
    position_pct: float = config.POSITION_PCT
    price_add_ticks: int = config.PRICE_ADD_TICKS

    def on_init(self) -> None:
        self.write_log("LMS 自适应滤波策略初始化")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        bar = bars.get(config.VT_SYMBOL)
        if bar is None:
            return

        sig = self.get_signal()
        if sig.is_empty():
            return

        val = sig["signal"][0]
        pred = float(val) if val is not None else 0.0

        direction = 0
        if pred > self.signal_threshold:
            direction = 1
        elif pred < -self.signal_threshold:
            direction = -1

        target_lots = 0
        if direction != 0:
            notional = bar.close_price * config.CONTRACT_SIZE
            if notional > 0:
                lots = int(self.get_portfolio_value() * self.position_pct / notional)
                target_lots = direction * max(lots, 0)

        self.set_target(config.VT_SYMBOL, float(target_lots))

        price_add = 0.0
        if bar.close_price > 0:
            price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price
        self.execute_trading(bars, price_add)

    def on_trade(self, trade: TradeData) -> None:
        pass
