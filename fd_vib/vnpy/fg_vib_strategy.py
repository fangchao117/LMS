"""
玻璃主力 Don 策略 —— VeighNa CTA（界面版仿真 / 实盘）

· 与 fd_vib 回测一致：Don 突破 + ATR 硬止损 / 移动止损 / 最大亏损
· 输入 1 分钟 K 线，BarGenerator 合成 30 分钟（VeighNa 无原生 30m 周期）
· T 收盘算信号 → 下一根 30m 调仓
· 换月：旧合约策略停止 → 新合约重新加载（平旧开新，不平移）

使用前：python install_vnpy.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vnpy_ctastrategy import TargetPosTemplate
from vnpy_ctastrategy.base import EngineType
from vnpy.trader.constant import Direction, Interval, Status
from vnpy.trader.object import BarData, OrderData, TickData
from vnpy.trader.utility import BarGenerator

# 郑商所玻璃 FG 交易时段（本地时间）
_FG_SESSIONS: tuple[tuple[time, time], ...] = (
    (time(9, 0), time(11, 30)),
    (time(13, 30), time(15, 0)),
    (time(21, 0), time(23, 0)),
)


class FgVibDonStrategy(TargetPosTemplate):
    """fd_vib 主力 Don 策略（30 分钟 K 线）"""

    author = "fd_vib"

    fd_vib_root: str = r"D:\LMS\fd_vib"
    profile: str = "max"
    signal_mode: str = "compute"  # compute | json
    live_json_path: str = ""

    donchian: int = 38
    mode: str = "donchian_baseline"
    max_lots: int = 2
    use_risk: bool = True
    atr_stop_mult: float = 3.5
    trail_atr_mult: float = 4.0
    max_loss_pct: float = 6.0
    bar_window: int = 120
    bar_minutes: int = 30
    capital: float = 10_000.0

    pending_target: int = 0
    entry_price: float = 0.0
    peak_pnl: float = 0.0
    last_signal: float = 0.0
    roll_warned: bool = False

    parameters = [
        "fd_vib_root",
        "profile",
        "signal_mode",
        "live_json_path",
        "donchian",
        "mode",
        "max_lots",
        "use_risk",
        "atr_stop_mult",
        "trail_atr_mult",
        "max_loss_pct",
        "bar_window",
        "bar_minutes",
        "capital",
    ]
    variables = [
        "pending_target",
        "entry_price",
        "peak_pnl",
        "last_signal",
        "roll_warned",
    ]

    def on_init(self) -> None:
        self._bars: list[dict[str, Any]] = []
        self._spec: dict[str, Any] = {}
        self._combo = None
        self._dominant = None
        self._startup_sync: bool = False
        self._startup_target: int = 0
        self._tick_seen: bool = False
        self._tick_count: int = 0

        pt = self.get_pricetick()
        if pt and pt > 0:
            self.tick_add = pt

        self._setup_fd_vib()
        self._load_spec()
        self.write_log(
            f"FgVibDon  profile={self.profile}  mode={self.mode}  "
            f"don={self.donchian}  max_lots={self.max_lots}  signal={self.signal_mode}"
        )
        self._check_dominant()
        self.bg = BarGenerator(
            on_bar=self._on_1m_from_tick,
            window=self.bar_minutes,
            on_window_bar=self.on_30m_bar,
            interval=Interval.MINUTE,
        )
        if self._load_bars_from_fg30m():
            pre = self._compute_pending_target() if len(self._bars) >= self.donchian + 5 else 0
            last_dt = self._bars[-1]["datetime"] if self._bars else "?"
            self.write_log(
                f"信号源 fg_30m 缓存 {len(self._bars)} 根 末根={last_dt} 预算pending={pre}"
            )
        else:
            days = max(5, (self.bar_window + self.donchian) // 8 + 3)
            self.load_bar(days)
            self.write_log(f"信号源 vnpy DB 1m 预载 {days} 天（可能与 fd_vib 不一致）")
        self.write_log(f"实盘 tick -> {self.bar_minutes}m  tick_add={self.tick_add}")

    def on_start(self) -> None:
        """vnpy: on_start 时 trading 仍为 False；init 后变量文件会覆盖 pending，须在此重算信号。"""
        self._tick_seen = False
        self._tick_count = 0
        self._check_dominant()
        self._refresh_signal()

        if self.get_engine_type() == EngineType.BACKTESTING:
            self.write_log(f"回测模式 30m_bars={len(self._bars)}")
        else:
            main_engine = getattr(self.cta_engine, "main_engine", None)
            if main_engine:
                contract = main_engine.get_contract(self.vt_symbol)
                if contract:
                    self.write_log(
                        f"合约 OK {contract.vt_symbol}  gateway={contract.gateway_name}  "
                        f"30m_bars={len(self._bars)}"
                    )
                else:
                    self.write_log(
                        f"合约未找到 {self.vt_symbol}！请在「合约查询」确认 SimNow 代码"
                    )
            else:
                self.write_log(f"仿真 30m_bars={len(self._bars)}")

        self.write_log(
            f"启动信号 pending={self.pending_target} sig={self.last_signal} pos={self.pos}"
        )
        if self.pending_target == 0:
            if self.get_engine_type() != EngineType.BACKTESTING:
                self.write_log("当前信号=空仓，不发单；等 30m 收盘变信号或 run.py --live 核对")
        elif self.pending_target != int(self.pos):
            self._startup_sync = True
            self._startup_target = int(self.pending_target)
            if self.get_engine_type() == EngineType.BACKTESTING:
                self.write_log(f"回测待调仓 target={self._startup_target}")
            else:
                self.write_log(f"启动待补单 target={self._startup_target}（等首笔 tick）")

    def _refresh_signal(self) -> None:
        """init 后 vnpy 会从 json 恢复 variables，覆盖 on_init 算出的 pending。"""
        if self.signal_mode == "json":
            self.pending_target = self._target_from_json()
        elif len(self._bars) >= self.donchian + 5:
            self.pending_target = self._compute_pending_target()

    def on_stop(self) -> None:
        self._startup_sync = False
        if self.trading and self.pos != 0:
            self.set_target_pos(0)

    def set_target_pos(self, target_pos: int) -> None:
        self.target_pos = int(target_pos)
        if self.trading:
            self.trade()

    def trade(self) -> None:
        if not self.trading:
            return
        if self.target_pos == int(self.pos):
            return
        if not self._has_valid_price():
            self.write_log(
                f"发单跳过：无有效报价 target={self.target_pos} pos={self.pos} "
                f"tick={self.last_tick is not None} bar={self.last_bar is not None}"
            )
            return
        super().trade()
        if self.target_pos != int(self.pos):
            self.write_log(
                f"发单未完成 target={self.target_pos} pos={self.pos} "
                f"price={self._order_price()} active={self.active_orderids}"
            )

    def send_new_order(self) -> None:
        """仿真盘 ask/bid 常为 0，回退到 last_price + tick_add。"""
        pos_change = self.target_pos - int(self.pos)
        if not pos_change:
            return

        long_price = 0.0
        short_price = 0.0
        add = float(self.tick_add or 1.0)

        if self.last_tick:
            lp = float(self.last_tick.last_price or 0)
            ask = float(self.last_tick.ask_price_1 or lp or 0)
            bid = float(self.last_tick.bid_price_1 or lp or 0)
            if pos_change > 0:
                long_price = (ask or lp) + add
                if self.last_tick.limit_up:
                    long_price = min(long_price, float(self.last_tick.limit_up))
            else:
                short_price = (bid or lp) - add
                if self.last_tick.limit_down:
                    short_price = max(short_price, float(self.last_tick.limit_down))
        elif self.last_bar:
            cp = float(self.last_bar.close_price)
            if pos_change > 0:
                long_price = cp + add
            else:
                short_price = cp - add
        else:
            self.write_log("send_new_order: 无 tick/bar，无法定价")
            return

        if self.get_engine_type() == EngineType.BACKTESTING:
            if pos_change > 0:
                vt_orderids = self.buy(long_price, abs(pos_change))
            else:
                vt_orderids = self.short(short_price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)
            return

        if self.active_orderids:
            return

        vt_orderids: list = []
        if pos_change > 0:
            if self.pos < 0:
                vol = pos_change if pos_change < abs(self.pos) else abs(self.pos)
                vt_orderids = self.cover(long_price, vol)
            else:
                vt_orderids = self.buy(long_price, abs(pos_change))
        else:
            if self.pos > 0:
                vol = abs(pos_change) if abs(pos_change) < self.pos else self.pos
                vt_orderids = self.sell(short_price, vol)
            else:
                vt_orderids = self.short(short_price, abs(pos_change))

        if vt_orderids:
            self.write_log(
                f"已报单 {pos_change:+d}手 @{long_price or short_price:.1f} ids={vt_orderids}"
            )
        else:
            self.write_log(
                f"报单返回空 target={self.target_pos} pos={self.pos} "
                f"price={long_price or short_price:.1f}"
            )
        self.active_orderids.extend(vt_orderids)

    def _sync_startup(self) -> None:
        if not self._startup_sync or not self.trading:
            return
        self._startup_sync = False
        tgt = self._startup_target
        self.write_log(
            f"启动补单 target={tgt} pos={self.pos} price={self._order_price()}"
        )
        self.set_target_pos(tgt)

    def _has_valid_price(self) -> bool:
        pos_change = self.target_pos - int(self.pos)
        if not pos_change:
            return False
        if self.last_tick:
            if pos_change > 0:
                p = self.last_tick.ask_price_1 or self.last_tick.last_price
            else:
                p = self.last_tick.bid_price_1 or self.last_tick.last_price
            return bool(p and p > 0)
        if self.last_bar:
            return self.last_bar.close_price > 0
        return False

    def _order_price(self) -> float:
        if self.last_tick:
            if self.target_pos > self.pos:
                return float(self.last_tick.ask_price_1 or self.last_tick.last_price or 0)
            return float(self.last_tick.bid_price_1 or self.last_tick.last_price or 0)
        if self.last_bar:
            return float(self.last_bar.close_price)
        return 0.0

    @staticmethod
    def _naive_dt(dt: Any) -> datetime:
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    @staticmethod
    def _in_trading_session(dt: datetime) -> bool:
        dt = FgVibDonStrategy._naive_dt(dt)
        t = dt.time()
        for start, end in _FG_SESSIONS:
            if start <= t <= end:
                return True
        return False

    def on_bar(self, bar: BarData) -> None:
        """init 历史回放：引擎推送 1m K 线（仿真实盘不走此路径）。"""
        self._process_1m_bar(bar)

    def _on_1m_from_tick(self, bar: BarData) -> None:
        """BarGenerator 从 tick 合成 1m 后回调。"""
        self._process_1m_bar(bar)

    def _process_1m_bar(self, bar: BarData) -> None:
        super().on_bar(bar)
        self._sync_startup()
        if self.use_risk and self.pos != 0:
            if self._check_stops(bar):
                self.pending_target = 0
                self.set_target_pos(0)
                if self._in_trading_session(bar.datetime):
                    self.bg.update_bar(bar)
                self.put_event()
                return
        if self._in_trading_session(bar.datetime):
            self.bg.update_bar(bar)

    def on_tick(self, tick: TickData) -> None:
        """仿真/实盘唯一实时行情入口：tick -> 1m -> 30m。"""
        super().on_tick(tick)
        self._tick_count += 1

        if not self._tick_seen:
            self._tick_seen = True
            self.write_log(
                f"首笔tick {tick.vt_symbol} last={tick.last_price} "
                f"bid={tick.bid_price_1} ask={tick.ask_price_1} "
                f"策略合约={self.vt_symbol}"
            )

        if tick.vt_symbol != self.vt_symbol:
            if self._tick_count <= 3:
                self.write_log(
                    f"tick合约不匹配: {tick.vt_symbol} != {self.vt_symbol}，"
                    f"请改策略合约或检查 SimNow 主力代码"
                )
            return

        self._sync_startup()
        if not tick.last_price:
            return
        self.bg.update_tick(tick)

    def on_order(self, order: OrderData) -> None:
        super().on_order(order)
        if order.status == Status.REJECTED:
            self.write_log(
                f"委托拒单 {order.vt_symbol} {order.direction.value} "
                f"{order.volume}@{order.price}  {order.reference}"
            )

    def on_30m_bar(self, bar: BarData) -> None:
        if self.trading and self.use_risk and self.pos != 0:
            if self._check_stops(bar):
                self.pending_target = 0
                self.set_target_pos(0)
                self.put_event()
                return

        exec_target = self.pending_target
        if self.signal_mode == "json":
            exec_target = self._target_from_json()
        elif self.trading:
            pass  # compute 模式用 pending_target

        if self.signal_mode == "compute":
            self._append_bar(bar)
            self.pending_target = self._compute_pending_target()
        elif self.trading:
            self.pending_target = exec_target

        if not self.trading:
            self.put_event()
            return

        self.last_bar = bar
        self.set_target_pos(int(exec_target))
        if exec_target != 0 and self.pos != 0 and self.entry_price <= 0:
            self.entry_price = float(bar.open_price or bar.close_price)
            self.peak_pnl = 0.0
        if exec_target != 0 or self.pending_target != 0:
            self.write_log(
                f"30m bar={bar.datetime} exec={exec_target} pos={self.pos} "
                f"close={bar.close_price} next={self.pending_target} sig={self.last_signal}"
            )
        self.put_event()

    # ------------------------------------------------------------------ fd_vib

    def _setup_fd_vib(self) -> None:
        root = Path(self.fd_vib_root).resolve()
        if not root.is_dir():
            self.write_log(f"fd_vib 路径不存在: {root}")
            return
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    def _load_spec(self) -> None:
        try:
            import config as fd_cfg

            tune = fd_cfg.ARTIFACT_PATH / "tune.json"
            if self.profile == "balanced":
                self._spec = dict(fd_cfg.BALANCED_SPEC)
            elif self.profile == "enhanced":
                self._spec = dict(fd_cfg.ENHANCED_SPEC)
            elif tune.exists():
                data = json.loads(tune.read_text(encoding="utf-8"))
                self._spec = dict(data["best_score"]["spec"])
            else:
                self._spec = dict(fd_cfg.DEFAULT_SPEC)
        except Exception as exc:
            self.write_log(f"加载 spec 失败，用类参数: {exc}")
            self._spec = {}

        for key in ("donchian", "mode", "max_lots", "use_risk", "atr_stop_mult", "trail_atr_mult", "max_loss_pct"):
            if key in self._spec:
                setattr(self, key, self._spec[key])

        if not self.live_json_path:
            self.live_json_path = str(Path(self.fd_vib_root) / "artifacts" / "live_signal.json")

        try:
            import combo

            self._combo = combo
        except Exception as exc:
            self.write_log(f"import combo 失败: {exc}")

        try:
            from dominant import current_dominant, next_roll_info

            self._dominant = (current_dominant, next_roll_info)
        except Exception as exc:
            self.write_log(f"import dominant 失败: {exc}")

    def _check_dominant(self) -> None:
        if not self._dominant:
            return
        current_dominant, next_roll_info = self._dominant
        sym, _, _ = current_dominant()
        roll = next_roll_info()
        vt = self.vt_symbol.split(".")[0].upper()
        if vt != sym and not self.roll_warned:
            self.write_log(
                f"合约 {vt} 非当期主力 {sym}；换月请停策略、平旧仓后在 {sym}.CZCE 重新加载"
            )
            self.roll_warned = True
        if roll.get("days_to_roll", 99) <= 3:
            self.write_log(
                f"距换月 {roll['days_to_roll']} 天 -> {roll.get('next_symbol')}，"
                f"换月日平 {sym} 再开新主力"
            )

    # ------------------------------------------------------------------ bars

    def _load_bars_from_fg30m(self) -> bool:
        """与 fd_vib 回测同源：fg_30m/artifacts/bars_30m_*.parquet"""
        sym = self.vt_symbol.split(".")[0].upper()
        cache = Path(self.fd_vib_root).parent / "fg_30m" / "artifacts" / f"bars_30m_{sym}.parquet"
        if not cache.exists():
            self.write_log(f"无 fg_30m 缓存 {cache}，将退回 vnpy DB")
            return False
        try:
            df = pd.read_parquet(cache)
            df["datetime"] = pd.to_datetime(df["datetime"])
            if len(df) > self.bar_window:
                df = df.tail(self.bar_window)
            rows: list[dict[str, Any]] = []
            for row in df.itertuples(index=False):
                dt = self._naive_dt(row.datetime)
                rows.append({
                    "datetime": dt,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(getattr(row, "volume", 0)),
                    "symbol": sym,
                    "vt_symbol": self.vt_symbol,
                })
            self._bars = rows
            return len(self._bars) > 0
        except Exception as exc:
            self.write_log(f"读取 fg_30m 缓存失败: {exc}")
            return False

    def _append_bar(self, bar: BarData) -> None:
        sym = self.vt_symbol.split(".")[0]
        self._bars.append({
            "datetime": self._naive_dt(bar.datetime),
            "open": float(bar.open_price),
            "high": float(bar.high_price),
            "low": float(bar.low_price),
            "close": float(bar.close_price),
            "volume": float(bar.volume),
            "symbol": sym,
            "vt_symbol": self.vt_symbol,
        })
        if len(self._bars) > self.bar_window:
            self._bars = self._bars[-self.bar_window :]

    def _bar_df(self) -> pd.DataFrame:
        if not self._bars:
            return pd.DataFrame()
        df = pd.DataFrame(self._bars)
        df["datetime"] = pd.to_datetime([self._naive_dt(x) for x in df["datetime"]])
        return df

    def _compute_pending_target(self) -> int:
        if self._combo is None:
            return 0
        df = self._bar_df()
        if len(df) < self.donchian + 5:
            return 0
        sig_kw = {
            k: v
            for k, v in self._spec.items()
            if k not in ("use_risk", "max_lots", "atr_stop_mult", "trail_atr_mult", "max_loss_pct")
        }
        sig_kw.setdefault("mode", self.mode)
        sig_kw.setdefault("donchian", self.donchian)
        try:
            sig = self._combo.generate(df, **sig_kw)
            raw = float(sig.iloc[-1])
        except Exception as exc:
            self.write_log(f"信号计算失败: {exc}")
            return 0
        self.last_signal = raw
        lots = int(round(raw))
        return int(max(-self.max_lots, min(self.max_lots, lots)))

    def _target_from_json(self) -> int:
        path = Path(self.live_json_path)
        if not path.exists():
            self.write_log(f"live_signal.json 不存在: {path}，请先运行 run.py --live")
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            vt = self.vt_symbol.split(".")[0].upper()
            dom = str(data.get("dominant_symbol", "")).upper()
            if dom and dom != vt:
                self.write_log(f"JSON 主力 {dom} 与策略合约 {vt} 不一致")
            self.last_signal = float(data.get("signal_raw", 0))
            tgt = int(data.get("target_position", 0))
            ml = int(data.get("max_lots", self.max_lots))
            return int(max(-ml, min(ml, tgt)))
        except Exception as exc:
            self.write_log(f"读取 JSON 失败: {exc}")
            return 0

    # ------------------------------------------------------------------ risk

    def _atr(self, df: pd.DataFrame, n: int = 14) -> float:
        if len(df) < 2:
            return 0.0
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        val = tr.rolling(n, min_periods=1).mean().iloc[-1]
        return float(val) if np.isfinite(val) else 0.0

    def _check_stops(self, bar: BarData) -> bool:
        df = self._bar_df()
        if df.empty or self.entry_price <= 0:
            return False
        atr_i = self._atr(df)
        if atr_i <= 0:
            return False

        size = float(self.get_size() or 20.0)
        pos = int(self.pos)
        h, l = float(bar.high_price), float(bar.low_price)
        if pos > 0:
            pnl_low = (l - self.entry_price) * size * pos
            pnl_high = (h - self.entry_price) * size * pos
        else:
            pnl_low = (self.entry_price - h) * size * (-pos)
            pnl_high = (self.entry_price - l) * size * (-pos)

        self.peak_pnl = max(self.peak_pnl, pnl_high)
        hard_stop = -self.atr_stop_mult * atr_i * size * abs(pos)
        trail_stop = self.peak_pnl - self.trail_atr_mult * atr_i * size * abs(pos)
        max_loss = -self.capital * self.max_loss_pct / 100.0

        if pnl_low <= hard_stop or pnl_low <= trail_stop or pnl_low <= max_loss:
            self.write_log(
                f"止损平仓 entry={self.entry_price:.1f} pos={pos} "
                f"low_pnl={pnl_low:.0f} peak={self.peak_pnl:.0f}"
            )
            self.entry_price = 0.0
            self.peak_pnl = 0.0
            return True
        return False

    def on_trade(self, trade) -> None:
        if self.pos == 0:
            self.entry_price = 0.0
            self.peak_pnl = 0.0
        elif self.entry_price <= 0:
            self.entry_price = float(trade.price)
            self.peak_pnl = 0.0
