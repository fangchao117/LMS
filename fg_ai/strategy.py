"""
AI 多空择时策略 —— GlassAlphaStrategy

消费 LightGBM 对"未来 HOLD 日收益"的预测值（signal），做单标的多空翻转：

    pred >  threshold  -> 满仓做多
    pred < -threshold  -> 满仓做空
    |pred| <= threshold -> 空仓（死区，默认 0 即纯多空翻转）

撮合时序由 vnpy.alpha 回测引擎保证：T 日收盘 on_bars 产生目标仓位并挂单，
订单在 T+1 日 K 线撮合成交，与标签口径（T+1 -> T+1+HOLD）严格对齐，无未来函数。
仓位规模按当前组合净值的 POSITION_PCT 折算为整数手。
"""
from __future__ import annotations

from vnpy.trader.object import BarData, TradeData
from vnpy.alpha import AlphaStrategy

import config


SIGNAL_COL: str = "signal"


class GlassAlphaStrategy(AlphaStrategy):
    """玻璃期货 AI 多空择时。"""

    # 可被 setting 覆盖的参数
    signal_threshold: float = config.SIGNAL_THRESHOLD
    position_pct: float = config.POSITION_PCT
    price_add_ticks: int = config.PRICE_ADD_TICKS

    def on_init(self) -> None:
        """策略初始化。"""
        self.write_log("玻璃 AI 择时策略初始化")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """每根 K 线：读取模型信号 -> 决定目标仓位 -> 委托调仓。"""
        bar: BarData | None = bars.get(config.VT_SYMBOL)
        if bar is None:
            return

        signal_df = self.get_signal()
        if signal_df.is_empty():
            return  # 无当日预测（预热或缺口），维持现状

        pred: float = float(signal_df[SIGNAL_COL][0])

        # 依据信号方向决定目标手数
        direction: int = 0
        if pred > self.signal_threshold:
            direction = 1
        elif pred < -self.signal_threshold:
            direction = -1

        target_lots: int = 0
        if direction != 0:
            portfolio_value: float = self.get_portfolio_value()
            notional_per_lot: float = bar.close_price * config.CONTRACT_SIZE
            if notional_per_lot > 0:
                lots: int = int(portfolio_value * self.position_pct / notional_per_lot)
                target_lots = direction * max(lots, 0)

        self.set_target(config.VT_SYMBOL, float(target_lots))

        # 超价挂单模拟滑点：以跳数折算成相对比例
        price_add: float = 0.0
        if bar.close_price > 0:
            price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price

        self.execute_trading(bars, price_add)

    def on_trade(self, trade: TradeData) -> None:
        """成交回报（此处无需额外处理，仓位由框架维护）。"""
        pass
