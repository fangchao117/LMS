"""逐主力合约分段检验 —— 同一策略在各段是否稳定盈利

    python test_per_contract.py
"""
from __future__ import annotations

import json

import config
import data
import dominant
import signals
from backtest import run

STRATEGIES = [
    ("daily_filter_donchian", {"ema_fast": 26, "ema_slow": 46, "donchian": 20}),
    ("daily_filter_ema", {"ema_fast": 26, "ema_slow": 46}),
    ("ema_cross", {"ema_fast": 26, "ema_slow": 46}),
]


def main() -> dict:
    config.ensure_dirs()
    rows: list[dict] = []

    print("逐段主力合约检验（1万1手）")
    print("=" * 88)

    for sym, start, end in dominant.DOMINANT_SEGMENTS:
        df = data.load_30m(sym, start=start, end=end, use_cache=True)
        if df.empty:
            continue
        print(f"\n## {sym}  {start} ~ {end}  ({len(df)} bars)")
        for label, kw in STRATEGIES:
            sig = signals.generate(df, label, **kw)
            st, _ = run(df, sig, max_lots=1)
            row = {
                "symbol": sym,
                "start": start,
                "end": end,
                "bars": len(df),
                "strategy": label,
                **{k: st[k] for k in ("total_return_pct", "max_ddpercent", "total_trades", "end_equity")},
            }
            rows.append(row)
            mark = "OK" if st["total_return_pct"] > 0 else "LOSS"
            print(
                f"  {label:24} {mark:4}  ret={st['total_return_pct']:7.1f}%  "
                f"dd={st['max_ddpercent']:6.1f}%  trades={st['total_trades']:3}"
            )

    # 全段拼接对比
    df_all = dominant.load_dominant_30m()
    print(f"\n## 主力拼接  {dominant.dominant_period()[0]} ~ {dominant.dominant_period()[1]}")
    for label, kw in STRATEGIES:
        st, _ = run(df_all, signals.generate(df_all, label, **kw), max_lots=1)
        print(
            f"  {label:24}      ret={st['total_return_pct']:7.1f}%  "
            f"dd={st['max_ddpercent']:6.1f}%  trades={st['total_trades']:3}"
        )

    # 稳定性汇总
    print("\n" + "=" * 88)
    print("稳定性汇总（各主力段正收益比例）")
    for label, _ in STRATEGIES:
        seg = [r for r in rows if r["strategy"] == label]
        wins = sum(1 for r in seg if r["total_return_pct"] > 0)
        rets = [r["total_return_pct"] for r in seg]
        avg = sum(rets) / len(rets) if rets else 0
        print(f"  {label:24}  盈利段 {wins}/{len(seg)}  段均收益 {avg:+.1f}%  各段 {rets}")

    out = {"segments": rows}
    path = config.ARTIFACT_PATH / "per_contract_stability.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    return out


if __name__ == "__main__":
    main()
