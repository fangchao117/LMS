"""各主力合约分段稳定性 —— python test_dominant_stability.py"""
from __future__ import annotations

import json

import dominant
import data
import signals
import config
from backtest import run


MODES = [
    ("daily_filter_donchian", {"donchian": 20}),
    ("daily_filter_ema", {}),
]


def main() -> None:
    config.ensure_dirs()
    rows: list[dict] = []

    print("FG609 1万1手  各主力窗口独立回测")
    print("=" * 78)

    for sym, start, end in dominant.DOMINANT_SEGMENTS:
        try:
            df = data.load_30m(sym, start=start, end=end, use_cache=True)
        except RuntimeError as e:
            print(f"{sym} {start}~{end}: 无数据 ({e})")
            continue
        if len(df) < 50:
            print(f"{sym}: 样本过短 {len(df)} bars")
            continue

        print(f"\n--- {sym}  {start} ~ {end}  ({len(df)} bars) ---")
        for mode, extra in MODES:
            kw = dict(ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW, **extra)
            sig = signals.generate(df, mode, **kw)
            st, _ = run(df, sig, max_lots=1)
            rec = {
                "symbol": sym,
                "window": [start, end],
                "bars": len(df),
                "mode": mode,
                **extra,
                "return_pct": st["total_return_pct"],
                "max_dd_pct": st["max_ddpercent"],
                "trades": st["total_trades"],
                "sharpe": st["sharpe_ratio"],
            }
            rows.append(rec)
            print(
                f"  {mode:26} ret={st['total_return_pct']:7.1f}%  "
                f"dd={st['max_ddpercent']:6.1f}%  trades={st['total_trades']:3}"
            )

    # 汇总统计
    print("\n" + "=" * 78)
    print("稳定性汇总（daily_filter_donchian）")
    don = [r for r in rows if r["mode"] == "daily_filter_donchian"]
    rets = [r["return_pct"] for r in don]
    wins = sum(1 for x in rets if x > 0)
    print(f"  分段数: {len(don)}  盈利段: {wins}  亏损段: {len(don)-wins}")
    if rets:
        print(f"  收益: min={min(rets):.1f}%  max={max(rets):.1f}%  avg={sum(rets)/len(rets):.1f}%")

    ema = [r for r in rows if r["mode"] == "daily_filter_ema"]
    rets_e = [r["return_pct"] for r in ema]
    wins_e = sum(1 for x in rets_e if x > 0)
    print("\n对比（daily_filter_ema）")
    print(f"  分段数: {len(ema)}  盈利段: {wins_e}  亏损段: {len(ema)-wins_e}")
    if rets_e:
        print(f"  收益: min={min(rets_e):.1f}%  max={max(rets_e):.1f}%  avg={sum(rets_e)/len(rets_e):.1f}%")

    out = config.ARTIFACT_PATH / "dominant_stability.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
