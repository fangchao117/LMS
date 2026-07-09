"""
玻璃短线策略 —— 小资金保证金算手数

· 净仓模式：多 OR 空，max_lots 限制净持仓
· 无「每日开仓次数」硬限制；日线策略自然约 1 次/日调仓
· 手数 = floor(min(权益, capital_max) × 仓位比例 / 保证金)，封顶 max_lots
"""
from __future__ import annotations

from vnpy.trader.constant import Offset
from vnpy.trader.object import BarData, TradeData
from vnpy.alpha import AlphaStrategy

import config


class ShortTermStrategy(AlphaStrategy):
    signal_threshold: float = config.SIGNAL_THRESHOLD
    position_pct: float = config.POSITION_PCT
    margin_rate: float = config.MARGIN_RATE
    max_lots: int = config.MAX_LOTS
    capital_max: int = config.CAPITAL_MAX
    price_add_ticks: int = config.PRICE_ADD_TICKS
    stop_loss_pct: float = config.STOP_LOSS_PCT
    max_drawdown_pct: float = config.MAX_DRAWDOWN_PCT

    entry_price: float = 0.0
    peak_equity: float = 0.0
    dd_halt: bool = False

    def on_init(self) -> None:
        self.peak_equity = float(config.CAPITAL)
        self.write_log(
            f"玻璃短线 资金={config.CAPITAL}~{self.capital_max} "
            f"最多净仓{self.max_lots}手 保证金率={self.margin_rate:.0%}"
        )

    def _calc_lots(self, price: float) -> int:
        if price <= 0:
            return 0
        margin_per_lot = price * config.CONTRACT_SIZE * self.margin_rate
        if margin_per_lot <= 0:
            return 0
        equity = min(float(self.get_portfolio_value()), float(self.capital_max))
        lots = int(equity * self.position_pct / margin_per_lot)
        return min(max(lots, 0), self.max_lots)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        bar = bars.get(config.VT_SYMBOL)
        if bar is None:
            return

        equity = float(self.get_portfolio_value())
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.dd_halt = False

        if self.max_drawdown_pct > 0 and self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd >= self.max_drawdown_pct:
                if not self.dd_halt:
                    self.write_log(f"账户回撤熔断 {dd:.1%} >= {self.max_drawdown_pct:.0%}，强制空仓")
                    self.dd_halt = True
                self.set_target(config.VT_SYMBOL, 0.0)
                price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price if bar.close_price > 0 else 0
                self.execute_trading(bars, price_add)
                self.entry_price = 0.0
                return

        if self._check_stop_loss(bar):
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
        if self.dd_halt:
            direction = 0
        if direction != 0:
            lots = self._calc_lots(bar.close_price)
            target_lots = direction * lots

        self.set_target(config.VT_SYMBOL, float(target_lots))

        price_add = 0.0
        if bar.close_price > 0:
            price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price
        self.execute_trading(bars, price_add)

    def _check_stop_loss(self, bar: BarData) -> bool:
        if self.stop_loss_pct <= 0 or self.entry_price <= 0 or bar.close_price <= 0:
            return False

        pos = self.get_target(config.VT_SYMBOL)
        hit = False
        if pos > 0 and bar.close_price <= self.entry_price * (1 - self.stop_loss_pct):
            hit = True
        elif pos < 0 and bar.close_price >= self.entry_price * (1 + self.stop_loss_pct):
            hit = True

        if not hit:
            return False

        self.set_target(config.VT_SYMBOL, 0.0)
        price_add = self.price_add_ticks * config.PRICE_TICK / bar.close_price
        self.execute_trading({config.VT_SYMBOL: bar}, price_add)
        self.write_log(
            f"止损平仓 entry={self.entry_price:.1f} close={bar.close_price:.1f} "
            f"阈值={self.stop_loss_pct:.1%}"
        )
        self.entry_price = 0.0
        return True

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN:
            self.entry_price = trade.price
        elif trade.offset in (Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY):
            if abs(self.get_target(config.VT_SYMBOL)) < 1e-6:
                self.entry_price = 0.0
