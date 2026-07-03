"""
玻璃短线策略 —— 小资金保证金算手数

signal >  threshold  -> 做多
signal < -threshold  -> 做空
否则空仓

手数 = floor(权益 × 仓位比例 / (现价×乘数×保证金率))，至少 0 手。
"""
from __future__ import annotations

from vnpy.trader.object import BarData, TradeData
from vnpy.alpha import AlphaStrategy

import config


class ShortTermStrategy(AlphaStrategy):
    signal_threshold: float = config.SIGNAL_THRESHOLD
    position_pct: float = config.POSITION_PCT
    margin_rate: float = config.MARGIN_RATE
    max_lots: int = config.MAX_LOTS
    price_add_ticks: int = config.PRICE_ADD_TICKS

    def on_init(self) -> None:
        self.write_log(
            f"玻璃短线策略 资金={config.CAPITAL} 模式={config.SIGNAL_MODE} "
            f"保证金率={self.margin_rate:.0%}"
        )

    def on_bars(self, bars: dict[str, BarData]) -> None:
        bar = bars.get(config.VT_SYMBOL)
        if bar is None:
            return

        signal_df = self.get_signal()
        if signal_df.is_empty():
            return

        raw = signal_df[config.SIGNAL_COL][0]
        if raw is None or raw != raw:
            return
        pred = float(raw)

        direction = 0
        if pred > self.signal_threshold:
            direction = 1
        elif pred < -self.signal_threshold:
            direction = -1

        target_lots = 0
        if direction != 0 and bar.close_price > 0:
            margin_per_lot = bar.close_price * config.CONTRACT_SIZE * self.margin_rate
            if margin_per_lot > 0:
                lots = int(self.get_portfolio_value() * self.position_pct / margin_per_lot)
                lots = min(max(lots, 0), self.max_lots)
                target_lots = direction * lots

        self.set_target(config.VT_SYMBOL, float(target_lots))

        price_add = 0.0
        if bar.close_price > 0:
            price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price
        self.execute_trading(bars, price_add)

    def on_trade(self, trade: TradeData) -> None:
        pass
