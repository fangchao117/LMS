"""
LMS 混合策略 —— 规则层定方向 + AI 模型辅助

AI 两种模式（config.LMS_AI_MODE）：
  · veto（默认，推荐）：趋势明确时默认顺势；仅当 AI z-score 强烈反向时否决
  · confirm：规则定方向后，要求 AI 同向超阈值才开仓（更保守）
  · off：纯三均线趋势，不使用 AI

撮合时序：T 收盘出信号 -> T+1 成交，无未来函数。
"""
from __future__ import annotations

from vnpy.trader.object import BarData, TradeData
from vnpy.alpha import AlphaStrategy

import config


class LmsAlphaStrategy(AlphaStrategy):
    """LMS 三均线 + AI 混合择时。"""

    use_ai: bool = config.LMS_USE_AI
    ai_mode: str = config.LMS_AI_MODE
    signal_threshold: float = config.SIGNAL_THRESHOLD
    veto_threshold: float = config.LMS_VETO_THRESHOLD
    position_pct: float = config.POSITION_PCT
    price_add_ticks: int = config.PRICE_ADD_TICKS

    def on_init(self) -> None:
        if not self.use_ai or self.ai_mode == "off":
            mode = "纯规则趋势"
        elif self.ai_mode == "veto":
            mode = f"趋势+AI否决(>{self.veto_threshold})"
        else:
            mode = f"趋势+AI确认(>{self.signal_threshold})"
        self.write_log(f"LMS 混合策略初始化（{mode}）")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        bar: BarData | None = bars.get(config.VT_SYMBOL)
        if bar is None:
            return

        signal_df = self.get_signal()
        if signal_df.is_empty():
            return

        regime: int = int(signal_df[config.LMS_REGIME_COL][0])

        pred: float = 0.0
        has_ai = self.use_ai and self.ai_mode != "off" and "signal" in signal_df.columns
        if has_ai:
            val = signal_df["signal"][0]
            if val is not None and val == val:
                pred = float(val)

        direction: int = 0
        if regime > 0:
            if not has_ai:
                direction = 1
            elif self.ai_mode == "veto":
                if pred > -self.veto_threshold:
                    direction = 1
            elif self.ai_mode == "confirm" and pred > self.signal_threshold:
                direction = 1
        elif regime < 0:
            if not has_ai:
                direction = -1
            elif self.ai_mode == "veto":
                if pred < self.veto_threshold:
                    direction = -1
            elif self.ai_mode == "confirm" and pred < -self.signal_threshold:
                direction = -1

        target_lots: int = 0
        if direction != 0:
            notional_per_lot: float = bar.close_price * config.CONTRACT_SIZE
            if notional_per_lot > 0:
                lots: int = int(self.get_portfolio_value() * self.position_pct / notional_per_lot)
                target_lots = direction * max(lots, 0)

        self.set_target(config.VT_SYMBOL, float(target_lots))

        price_add: float = 0.0
        if bar.close_price > 0:
            price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price

        self.execute_trading(bars, price_add)

    def on_trade(self, trade: TradeData) -> None:
        pass
