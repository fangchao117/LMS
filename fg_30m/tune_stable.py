"""
稳定性优先扫参 —— 各主力段都要尽量盈利

评分：最差段收益 > 盈利段数 > TEST 段 > 回撤

    python tune_stable.py
"""
from __future__ import annotations

import json
from itertools import product

import config
import data
import dominant
import signals
from backtest import run


def _seg_returns(spec: dict) -> list[dict]:
    rows = []
    for sym, start, end in dominant.DOMINANT_SEGMENTS:
        df = data.load_30m(sym, start=start, end=end, use_cache=True)
        if len(df) < 50:
            continue
        kw = {k: v for k, v in spec.items() if k not in ("label", "mode")}
        sig = signals.generate(df, spec["mode"], **kw)
        st, _ = run(df, sig, max_lots=1)
        rows.append({
            "symbol": sym,
            "return_pct": st["total_return_pct"],
            "max_dd_pct": st["max_ddpercent"],
            "trades": st["total_trades"],
        })
    return rows


def _stability_score(seg_rets: list[float], test_ret: float, full_dd: float) -> float:
    if not seg_rets:
        return float("-inf")
    min_ret = min(seg_rets)
    wins = sum(1 for r in seg_rets if r > 0)
    n = len(seg_rets)
    score = min_ret * 3.0 + wins * 15.0 + test_ret * 0.8 + full_dd * 0.25
    if wins == n:
        score += 40.0
    if min_ret > 0:
        score += 25.0
    if min_ret > 5:
        score += 10.0
    return score


def _grid() -> list[dict]:
    out: list[dict] = []
    ema_pairs = [(26, 46), (12, 48), (34, 55)]
    adx_ths = [20.0, 25.0, 30.0, 35.0]

    for adx_th in adx_ths:
        for ef, es in ema_pairs:
            out.append({
                "label": f"daily_adx_{int(adx_th)}_ema_{ef}_{es}",
                "mode": "daily_adx_filter_ema",
                "adx_threshold": adx_th,
                "ema_fast": ef,
                "ema_slow": es,
            })

    for adx_th in adx_ths:
        out.append({
            "label": f"daily_stable_adx_{int(adx_th)}",
            "mode": "daily_filter_ema_stable",
            "adx_threshold": adx_th,
            "ema_fast": 26,
            "ema_slow": 46,
            "trend_ma": 50,
        })

    for ef, es in ema_pairs:
        out.append({
            "label": f"daily_filter_ema_{ef}_{es}",
            "mode": "daily_filter_ema",
            "ema_fast": ef,
            "ema_slow": es,
        })

    out.append({
        "label": "ema_stable_26_46",
        "mode": "ema_stable",
        "ema_fast": 26,
        "ema_slow": 46,
        "adx_threshold": 30.0,
    })
    return out


def main() -> dict:
    config.ensure_dirs()
    df_all = dominant.load_dominant_30m()
    t0, t1 = config.TEST_PERIOD
    df_test = df_all[(df_all["datetime"] >= t0) & (df_all["datetime"] <= t1)]

    results: list[dict] = []
    for spec in _grid():
        segs = _seg_returns(spec)
        seg_rets = [s["return_pct"] for s in segs]
        kw = {k: v for k, v in spec.items() if k not in ("label", "mode")}
        st_full, _ = run(df_all, signals.generate(df_all, spec["mode"], **kw), max_lots=1)
        st_test, _ = run(df_test, signals.generate(df_test, spec["mode"], **kw), max_lots=1)
        wins = sum(1 for r in seg_rets if r > 0)
        rec = {
            "label": spec["label"],
            "mode": spec["mode"],
            "params": kw,
            "seg_min": min(seg_rets) if seg_rets else None,
            "seg_avg": sum(seg_rets) / len(seg_rets) if seg_rets else None,
            "seg_wins": wins,
            "seg_n": len(seg_rets),
            "seg_detail": segs,
            "full_ret": st_full["total_return_pct"],
            "full_dd": st_full["max_ddpercent"],
            "test_ret": st_test["total_return_pct"],
            "score": _stability_score(seg_rets, st_test["total_return_pct"], st_full["max_ddpercent"]),
        }
        results.append(rec)

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0] if results else None

    print("稳定性扫参（主力四段 + TEST）")
    print("=" * 90)
    for r in results[:8]:
        print(
            f"  {r['label']:<32} seg {r['seg_wins']}/{r['seg_n']}  "
            f"min={r['seg_min']:6.1f}%  TEST={r['test_ret']:6.1f}%  "
            f"FULL={r['full_ret']:6.1f}%  dd={r['full_dd']:5.1f}%  score={r['score']:.0f}"
        )
    if best:
        print(f"\n[best stable] {best['label']}")
        for s in best["seg_detail"]:
            print(f"    {s['symbol']}: {s['return_pct']:+.1f}%  trades={s['trades']}")

    out = {"best": best, "top10": results[:10]}
    path = config.ARTIFACT_PATH / "tune_stable.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    return out


if __name__ == "__main__":
    main()
