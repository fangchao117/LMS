"""
因子深度挖掘 —— 各维度 IC + 与 Don 组合快测

    python mine.py
"""
from __future__ import annotations

import json

import combo
import config
import ic
import registry
from dominant import DOMINANT_SEGMENTS, load_dominant_30m
from risk_bt import run as run_risk


def main() -> dict:
    config.ensure_dirs()
    ic_out = ic.save_screen()
    df = load_dominant_30m(adjust=False)

    print(f"[mine] 因子总数 {ic_out['n_factors']}  horizon={config.IC_HORIZON}")
    print("\n=== 各梯队 TOP 因子 ===")
    for tid, info in registry.TIERS.items():
        names = ic_out["by_tier"].get(tid, [])
        if not names:
            continue
        print(f"\n{info['label']} ({tid}):")
        ic_map = {r["name"]: r for r in ic_out["all"]}
        for n in names[:5]:
            r = ic_map.get(n, {})
            print(f"  {n:<24} IC={r.get('ic', 0):+.4f}")

    # 玻璃专项
    glass_rows = [r for r in ic_out["all"] if r["name"].startswith("glass_")]
    print(f"\n=== 玻璃专项 ({len(glass_rows)} 个) TOP10 ===")
    for r in glass_rows[:10]:
        print(f"  {r['name']:<24} IC={r['ic']:+.4f}")

    # 快测：Don vs Don增强
    base_spec = {"mode": "donchian_baseline", "donchian": 38, "daily_filter": False}
    enh_spec = {
        "mode": "don_enhanced",
        "donchian": 38,
        "daily_filter": True,
        "threshold": 0.35,
        "smooth": 3,
        "factors": [r["name"] for r in ic_out["all"][:12]],
        "chop_max": 0.35,
        "strict": True,
    }
    risk = dict(max_lots=2, atr_stop_mult=3.5, trail_atr_mult=4.0, max_loss_pct=6.0, handle_roll=True)

    print("\n=== 主力拼接 策略快测 ===")
    for label, spec in [("Don38基准", base_spec), ("Don增强+震荡过滤", enh_spec)]:
        sig = combo.generate(df, **{k: v for k, v in spec.items()})
        st, _ = run_risk(df, sig, **risk)
        print(f"  {label:<20} FULL={st['total_return_pct']:+.1f}%  "
              f"回撤={st['max_ddpercent']:.1f}%  Sharpe={st['sharpe_ratio']:.2f}  "
              f"笔={st['total_trades']}")

    out = {
        "n_factors": ic_out["n_factors"],
        "by_tier": ic_out["by_tier"],
        "glass_top10": glass_rows[:10],
        "global_top20": ic_out["all"][:20],
    }
    path = config.ARTIFACT_PATH / "factor_mine.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    return out


if __name__ == "__main__":
    main()
