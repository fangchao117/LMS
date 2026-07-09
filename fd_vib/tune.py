"""
参数网格 —— 未复权主力 + 换月成本，收益最大化

    python tune.py
"""
from __future__ import annotations

import itertools
import json

import combo
import config
import ic
import registry
from bridge import fg_generate_signal, load_30m
from dominant import DOMINANT_SEGMENTS, load_dominant_30m
from risk_bt import run as run_risk


def _eval(df, df_test, spec: dict) -> dict:
    sig_kw = {k: v for k, v in spec.items()
              if k not in ("use_risk", "max_lots", "atr_stop_mult", "trail_atr_mult", "max_loss_pct")}
    ml = spec.get("max_lots", 2)
    risk = dict(
        max_lots=ml,
        atr_stop_mult=spec.get("atr_stop_mult", 3.5),
        trail_atr_mult=spec.get("trail_atr_mult", 2.62),
        max_loss_pct=spec.get("max_loss_pct", 6.0),
        handle_roll=True,
    )
    sig_f = combo.generate(df, **sig_kw)
    sig_t = combo.generate(df_test, **sig_kw)
    st_f, _ = run_risk(df, sig_f, **risk)
    st_t, _ = run_risk(df_test, sig_t, **risk)
    score = (
        st_f["total_return_pct"] * 0.42
        + st_t["total_return_pct"] * 0.28
        + st_f["max_ddpercent"] * 0.35
        + st_f.get("sharpe_ratio", 0) * 8.0
    )
    return {
        "full_ret": st_f["total_return_pct"],
        "full_dd": st_f["max_ddpercent"],
        "test_ret": st_t["total_return_pct"],
        "trades": st_f["total_trades"],
        "roll_closes": st_f.get("roll_closes", 0),
        "score": score,
    }


def main() -> dict:
    config.ensure_dirs()
    ic_out = ic.save_screen()
    by_tier = ic_out["by_tier"]
    top12 = [r["name"] for r in ic_out["all"][:12]]
    tier1 = by_tier.get("tier1", [])[:4]
    glass_top = by_tier.get("tier6", [])[:4]

    df = load_dominant_30m(adjust=False)
    t0 = config.TEST_PERIOD[0]
    df_test = df[df["datetime"] >= t0].reset_index(drop=True)

    specs = []
    for mode, dc, daily, ml, stop, trail, loss, th, chop in itertools.product(
        ["donchian_baseline", "don_enhanced", "don_factor_filter", "four_tier_don_filter"],
        [24, 30, 38],
        [False, True],
        [2],
        [3.5, 4.0, 5.0],
        [2.62, 4.0],
        [6.0, 8.0],
        [0.28, 0.38],
        [0.30, 0.40],
    ):
        spec: dict = {
            "mode": mode,
            "donchian": dc,
            "daily_filter": daily,
            "threshold": th,
            "smooth": 3,
            "max_lots": ml,
            "use_risk": True,
            "atr_stop_mult": stop,
            "trail_atr_mult": trail,
            "max_loss_pct": loss,
        }
        if mode == "don_enhanced":
            spec["factors"] = top12
            spec["chop_max"] = chop
            spec["strict"] = True
        elif mode == "don_factor_filter":
            spec["factors"] = list(dict.fromkeys(tier1 + glass_top))
        elif mode == "four_tier_don_filter":
            spec["tier_per"] = 3
        specs.append(spec)

    results = []
    for i, spec in enumerate(specs):
        ev = _eval(df, df_test, spec)
        results.append({"spec": spec, **ev})
        if (i + 1) % 100 == 0:
            print(f"  … {i + 1}/{len(specs)}")

    by_score = sorted(results, key=lambda x: x["score"], reverse=True)
    by_ret = sorted(results, key=lambda x: x["full_ret"], reverse=True)

    print(f"\n[fd_vib tune] {len(specs)} 组  收益+回撤综合评分")
    print(f"{'模式':<22} {'don':>3} {'FULL':>8} {'TEST':>8} {'回撤':>8} {'score':>8}")
    print("-" * 62)
    for r in by_score[:10]:
        s = r["spec"]
        print(f"{s['mode']:<22} {s['donchian']:3d} "
              f"{r['full_ret']:7.1f}% {r['test_ret']:7.1f}% {r['full_dd']:7.1f}% {r['score']:8.1f}")

    out = {
        "period": list(config.BACKTEST_PERIOD),
        "segments": DOMINANT_SEGMENTS,
        "adjust": False,
        "best_score": by_score[0],
        "best_return": by_ret[0],
        "top10_score": by_score[:10],
    }
    path = config.ARTIFACT_PATH / "tune.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    b = by_score[0]
    br = by_ret[0]
    print(f"\n综合最优: {b['spec']['mode']} don={b['spec']['donchian']} "
          f"FULL={b['full_ret']:.1f}% 回撤={b['full_dd']:.1f}%")
    print(f"收益最优: {br['spec']['mode']} don={br['spec']['donchian']} "
          f"FULL={br['full_ret']:.1f}% 回撤={br['full_dd']:.1f}%")
    print(f"-> {path}")
    return out


if __name__ == "__main__":
    main()
