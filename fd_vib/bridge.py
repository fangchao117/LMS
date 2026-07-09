"""复用 fg_30m 数据层。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_FG30 = Path(__file__).resolve().parent.parent / "fg_30m"


def _import_fg30() -> tuple[Any, Any, Any, Any]:
    if str(_FG30) not in sys.path:
        sys.path.insert(0, str(_FG30))
    for name in ("data", "regime", "signals", "backtest"):
        mod = sys.modules.get(name)
        fp = getattr(mod, "__file__", "") or ""
        if name in sys.modules and not fp.startswith(str(_FG30)):
            del sys.modules[name]
    if "config" in sys.modules:
        fp = getattr(sys.modules["config"], "__file__", "") or ""
        if not fp.startswith(str(_FG30)):
            del sys.modules["config"]

    fg_data = __import__("data")
    fg_regime = __import__("regime")
    fg_signals = __import__("signals")
    fg_backtest = __import__("backtest")

    _fg30_str = str(_FG30)
    if sys.path and sys.path[0] == _fg30_str:
        sys.path.pop(0)
    elif _fg30_str in sys.path:
        sys.path.remove(_fg30_str)
    return fg_data, fg_regime, fg_signals, fg_backtest


_fg_data, _fg_regime, _fg_signals, _fg_backtest = _import_fg30()

_cfg = sys.modules.get("config")
if _cfg and str(_FG30) in (getattr(_cfg, "__file__", "") or ""):
    del sys.modules["config"]

load_30m = _fg_data.load_30m
load_daily = _fg_regime.load_daily
fg_generate_signal = _fg_signals.generate
run_simple = _fg_backtest.run
