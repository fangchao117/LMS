"""
VeighNa CTA 回测 —— 命令行直接出结果（与 fd_vib 对照）

    cd D:\\LMS\\fd_vib
    python vnpy_backtest.py
    python vnpy_backtest.py --symbol FG605 --profile smart
    python vnpy_backtest.py --symbol FG605 --start 2025-12-15 --end 2026-04-14
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vnpy"))

import combo
import config
from bridge import load_30m
from dominant import DOMINANT_SEGMENTS
from risk_bt import run as run_risk


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s.strip()[:10], "%Y-%m-%d")


def segment_dates(symbol: str) -> tuple[str, str] | None:
    sym = symbol.upper()
    for s, start, end in DOMINANT_SEGMENTS:
        if s == sym:
            return start, end
    return None


def resolve_dates(symbol: str, start: str | None, end: str | None) -> tuple[str, str]:
    seg = segment_dates(symbol)
    if start and end:
        return start, end
    if seg:
        return seg
    if start:
        return start, end or "2026-07-09"
    return "2026-04-15", end or "2026-07-09"


def check_vnpy_bars(symbol: str, start: str, end: str) -> int:
    try:
        from vnpy.trader.constant import Exchange, Interval
        from vnpy.trader.database import get_database

        sym = symbol.upper()
        db = get_database()
        bars = db.load_bar_data(
            sym, Exchange.CZCE, Interval.MINUTE,
            _parse_date(start), _parse_date(end),
        )
        return len(bars) if bars else 0
    except Exception:
        return -1


def run_fd_vib(symbol: str, start: str, end: str, profile: str) -> dict:
    from run import load_best_spec

    spec = load_best_spec(profile)
    kw = {k: v for k, v in spec.items()
          if k not in ("use_risk", "max_lots", "atr_stop_mult", "trail_atr_mult", "max_loss_pct")}
    risk = dict(
        max_lots=spec.get("max_lots", 2),
        atr_stop_mult=spec.get("atr_stop_mult", 3.5),
        trail_atr_mult=spec.get("trail_atr_mult", 4.0),
        max_loss_pct=spec.get("max_loss_pct", 6.0),
        handle_roll=True,
    )
    df = load_30m(symbol, start=start, end=end)
    if len(df) < 50:
        return {
            "error": f"fd_vib 无足够 30m 数据（{len(df)} 根）",
            "bars": len(df),
            "hint": f"请确认 fg_30m/artifacts/bars_30m_{symbol.upper()}.parquet 覆盖 {start}~{end}",
        }
    sig = combo.generate(df, **kw)
    st, _ = run_risk(df, sig, **risk)
    return {"bars": len(df), **st}


def run_vnpy(symbol: str, start: str, end: str, profile: str) -> dict:
    from vnpy_ctastrategy.backtesting import BacktestingEngine
    from vnpy.trader.constant import Interval
    from fg_vib_strategy import FgVibDonStrategy

    n_bars = check_vnpy_bars(symbol, start, end)
    if n_bars == 0:
        return {
            "error": "vnpy 数据库无 1m 数据",
            "bars": 0,
            "hint": (
                f"python import_vnpy_data.py --symbol {symbol.upper()} --from-cache\n"
                f"或: python import_vnpy_data.py --symbol {symbol.upper()} --start {start} --end {end}"
            ),
        }

    vt = f"{symbol.upper()}.CZCE"
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt,
        interval=Interval.MINUTE,
        start=_parse_date(start),
        end=_parse_date(end),
        rate=0.0001,
        slippage=1.0,
        size=20,
        pricetick=1.0,
        capital=float(config.CAPITAL),
    )
    engine.add_strategy(FgVibDonStrategy, {
        "fd_vib_root": str(ROOT),
        "profile": profile,
        "signal_mode": "compute",
    })
    engine.load_data()
    if not engine.history_data:
        return {
            "error": "vnpy 回测加载 0 根 K 线",
            "bars": 0,
            "hint": f"检查 {vt} 在 {start}~{end} 是否有 1m 数据（当前库内约 {n_bars} 根）",
        }
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics() or {}
    if not stats:
        return {"error": "vnpy 回测无成交", "bars": len(engine.history_data)}
    return {
        "bars": len(engine.history_data),
        "total_return_pct": float(stats.get("total_return", 0)),
        "max_ddpercent": float(stats.get("max_ddpercent", 0)),
        "total_trades": int(stats.get("total_trade_count", 0)),
        "total_commission": float(stats.get("total_commission", 0)),
        "total_slippage": float(stats.get("total_slippage", 0)),
        "end_balance": float(stats.get("end_balance", 0)),
        "sharpe_ratio": float(stats.get("sharpe_ratio", 0)),
    }


def _print_result(label: str, r: dict) -> None:
    print(f"=== {label} ===")
    if r.get("error"):
        print(f"  错误: {r['error']}")
        if r.get("hint"):
            print(f"  提示: {r['hint']}")
        return
    print(f"  K线     {r.get('bars', '?')} 根")
    print(f"  收益     {r.get('total_return_pct', 0):+.2f}%")
    print(f"  回撤     {r.get('max_ddpercent', 0):.2f}%")
    print(f"  成交     {r.get('total_trades', 0)} 笔")
    if "total_commission" in r:
        print(f"  手续费   {r['total_commission']:.2f}")
        print(f"  滑点     {r['total_slippage']:.2f}")
        print(f"  结束资金 {r.get('end_balance', 0):.2f}")
    if "stop_hits" in r:
        print(f"  止损     {r.get('stop_hits', 0)} 次")


def main() -> int:
    ap = argparse.ArgumentParser(description="vnpy CTA 回测 + fd_vib 对照")
    ap.add_argument("--symbol", default="FG609")
    ap.add_argument("--start", default="", help="默认按 dominant.py 主力段自动选取")
    ap.add_argument("--end", default="")
    ap.add_argument("--profile", default="smart",
                    choices=["max", "smart", "auto", "regime", "balanced", "lms", "lms_dir", "lms_mom"])
    args = ap.parse_args()

    start, end = resolve_dates(
        args.symbol,
        args.start or None,
        args.end or None,
    )
    seg = segment_dates(args.symbol)
    auto_note = f"（主力段 {seg[0]}~{seg[1]}）" if seg and not args.start and not args.end else ""

    print(f"\n[vnpy_backtest] {args.symbol}  {start} ~ {end}  profile={args.profile} {auto_note}\n")

    vn = run_vnpy(args.symbol, start, end, args.profile)
    fd = run_fd_vib(args.symbol, start, end, args.profile)

    _print_result("VeighNa CTA 回测", vn)
    print()
    _print_result("fd_vib 对照", fd)

    out = {
        "symbol": args.symbol,
        "start": start,
        "end": end,
        "profile": args.profile,
        "segment": dict(zip(("start", "end"), seg)) if seg else None,
        "vnpy": vn,
        "fd_vib": fd,
    }
    path = config.ARTIFACT_PATH / "vnpy_backtest.json"
    config.ensure_dirs()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}\n")

    if vn.get("error") or fd.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
