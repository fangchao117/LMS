"""
60 分钟定方向 + 30 分钟交易 —— 与单周期基线对比

    python test_mtf.py
"""
from __future__ import annotations

import json

import config
import data
import signals
from backtest import run


def _slice(df, period: tuple[str, str]):
    t0, t1 = period
    return df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].reset_index(drop=True)


def main() -> dict:
    config.ensure_dirs()
    sym = config.SYMBOL
    df30 = data.load_30m(sym)
    df60 = data.load_60m(sym)
    df_v30, df_t30 = _slice(df30, config.TUNE_PERIOD), _slice(df30, config.TEST_PERIOD)
    df_v60, df_t60 = _slice(df60, config.TUNE_PERIOD), _slice(df60, config.TEST_PERIOD)

    def _eval(name: str, sig_v, sig_t) -> dict:
        st_v, _ = run(df_v30, sig_v)
        st_t, _ = run(df_t30, sig_t)
        return {
            "name": name,
            "valid_return": st_v["total_return_pct"],
            "valid_trades": st_v["total_trades"],
            "test_return": st_t["total_return_pct"],
            "test_trades": st_t["total_trades"],
        }

    rows = [
        _eval(
            "30m EMA 12/48",
            signals.generate(df_v30, "ema_cross", ema_fast=12, ema_slow=48),
            signals.generate(df_t30, "ema_cross", ema_fast=12, ema_slow=48),
        ),
        _eval(
            "30m EMA 24/48",
            signals.generate(df_v30, "ema_cross", ema_fast=24, ema_slow=48),
            signals.generate(df_t30, "ema_cross", ema_fast=24, ema_slow=48),
        ),
        _eval(
            "60m EMA 6/24 (trade on 30m)",
            signals.align_htf_to_ltf(
                df_v30, df_v60,
                signals.generate(df_v60, "ema_cross", ema_fast=6, ema_slow=24),
            ),
            signals.align_htf_to_ltf(
                df_t30, df_t60,
                signals.generate(df_t60, "ema_cross", ema_fast=6, ema_slow=24),
            ),
        ),
        _eval(
            "MTF 60m 6/24 + 30m 24/48",
            signals.generate_mtf(
                df_v30, df_v60,
                ltf_ema_fast=24, ltf_ema_slow=48,
                htf_ema_fast=6, htf_ema_slow=24,
            ),
            signals.generate_mtf(
                df_t30, df_t60,
                ltf_ema_fast=24, ltf_ema_slow=48,
                htf_ema_fast=6, htf_ema_slow=24,
            ),
        ),
        _eval(
            "MTF 60m 4/12 + 30m 5/20",
            signals.generate_mtf(
                df_v30, df_v60,
                ltf_ema_fast=5, ltf_ema_slow=20,
                htf_ema_fast=4, htf_ema_slow=12,
            ),
            signals.generate_mtf(
                df_t30, df_t60,
                ltf_ema_fast=5, ltf_ema_slow=20,
                htf_ema_fast=4, htf_ema_slow=12,
            ),
        ),
    ]

    base = rows[1]
    recommend = any(
        r["valid_return"] >= base["valid_return"]
        and r["test_return"] > base["test_return"]
        for r in rows[3:]
    )

    out = {"symbol": sym, "results": rows, "recommend_mtf": recommend}
    path = config.ARTIFACT_PATH / "mtf_test.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[test_mtf] {sym} 60m direction + 30m execution @ FG609")
    for r in rows:
        print(
            f"  {r['name']:<28} VALID {r['valid_return']:6.1f}%  "
            f"TEST {r['test_return']:6.1f}%  trades {r['test_trades']}"
        )
    if recommend:
        print("[test_mtf] MTF beats 30m 24/48 on both segments -> consider enabling")
    else:
        print("[test_mtf] MTF does not beat 30m EMA 24/48 -> keep single timeframe")
    print(f"[test_mtf] -> {path}")
    return out


if __name__ == "__main__":
    main()
