"""
因子训练 —— 从数据里「学」出新因子（ML + 模板搜索）

与 fd_vib 规则策略（Don/EMA/动能）不同，本脚本属于 **因子模型** 路线：
  · LightGBM 学价量结构 → 产出 disc_lgb_score
  · GTJA/WorldQuant 风格模板穷举 → IC 筛选 TOP 因子
  · 自动写入 discovered.py 注册到第七梯队

用法:
    cd D:\\LMS\\fd_vib
    python factor_train.py
    python factor_train.py --top 8 --min-ic 0.025
    python run.py --quick --profile smart   # 原规则策略不变

微博博主常见的「多因子模型 / ML 选股」属于本脚本这类；
fd_vib 的 profile=smart 是 **规则组合**，不是训练出来的权重。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
import factor_factory as ff
import ic
from dominant import load_dominant_30m
from risk_bt import run as run_risk
import combo


def _quick_backtest(df: pd.DataFrame, factor_names: list[str]) -> dict:
    """Don38 + 新因子增强，与 mine.py 快测一致。"""
    base = {"mode": "donchian_baseline", "donchian": 38, "daily_filter": False}
    enh = {
        "mode": "don_enhanced",
        "donchian": 38,
        "daily_filter": True,
        "threshold": 0.35,
        "smooth": 3,
        "factors": factor_names,
        "chop_max": 0.35,
        "strict": True,
    }
    risk = dict(max_lots=2, atr_stop_mult=3.5, trail_atr_mult=4.0, max_loss_pct=6.0, handle_roll=True)
    out = {}
    for label, spec in [("Don38", base), ("Don+新因子", enh)]:
        sig = combo.generate(df, **spec)
        st, _ = run_risk(df, sig, **risk)
        out[label] = {
            "return_pct": st["total_return_pct"],
            "max_dd": st["max_ddpercent"],
            "sharpe": st["sharpe_ratio"],
            "trades": st["total_trades"],
        }
    return out


def main() -> dict:
    parser = argparse.ArgumentParser(description="训练 / 搜索玻璃期货新因子")
    parser.add_argument("--horizon", type=int, default=config.IC_HORIZON)
    parser.add_argument("--top", type=int, default=6, help="保留模板因子数量")
    parser.add_argument("--min-ic", type=float, default=0.03, help="模板因子 |IC| 门槛")
    parser.add_argument("--train-end", default="2026-02-28")
    parser.add_argument("--valid-end", default=config.TUNE_PERIOD[1])
    args = parser.parse_args()

    config.ensure_dirs()
    df = load_dominant_30m(adjust=False)
    print(f"[factor_train] 主力拼接 bars={len(df)}  horizon={args.horizon}")

    # 1) 模板搜索
    print("\n=== 模板因子搜索 ===")
    templates = ff.template_candidates(df)
    screened = ff.screen_candidates(df, templates, args.horizon, args.min_ic)
    top_templates = screened[: args.top]
    for c in top_templates[:10]:
        print(f"  {c.name:<28} IC={c.ic:+.4f}")

    # 2) LightGBM 新因子
    print("\n=== LightGBM 训练 disc_lgb_score ===")
    feats = ff.base_feature_frame(df)
    lgb_score, lgb_meta = ff.train_lgb_score(
        df, feats, args.horizon, args.train_end, args.valid_end
    )
    lgb_ic = ic.spearman_ic(lgb_score, ic.forward_return(df["close"].astype(float), args.horizon))
    print(f"  disc_lgb_score  IC={lgb_ic:+.4f}  test_IC={lgb_meta['ic_test']:+.4f}  "
          f"best_iter={lgb_meta['best_iteration']}")
    for f in lgb_meta["top_features"][:5]:
        print(f"    feat {f['name']:<16} gain={f['gain']}")

    # 保存 LGB 序列（discovered.py 读取）
    lgb_path = config.ARTIFACT_PATH / "disc_lgb_score.parquet"
    lgb_score.to_frame().to_parquet(lgb_path)

    # 3) 写出 discovered.py
    disc_path = Path(__file__).resolve().parent / "discovered.py"
    ff.write_discovered_module(disc_path, top_templates, include_lgb=True)
    print(f"\n-> 注册模块 {disc_path}")

    # 4) 对新因子整体 IC 复检（含 registry 已有 + discovered）
    import importlib
    import registry

    try:
        import discovered
        importlib.reload(discovered)
    except Exception as e:
        print(f"警告: discovered 加载失败 {e}")
        discovered = None

    new_names = [c.name for c in top_templates]
    if discovered:
        new_names = list(discovered.all_factors().keys())

    fwd = ic.forward_return(df["close"].astype(float), args.horizon)
    ic_rows = []
    if discovered:
        panel = discovered.compute_panel(df, new_names) if hasattr(discovered, "compute_panel") else None
        if panel is None:
            panel = pd.DataFrame({n: discovered.all_factors()[n](df) for n in new_names})
        for col in panel.columns:
            ic_rows.append({
                "name": col,
                "ic": ic.spearman_ic(panel[col], fwd),
                "kind": "lgb" if col == "disc_lgb_score" else "template",
            })
    ic_rows.sort(key=lambda x: abs(x["ic"]), reverse=True)

    # 5) 快测
    print("\n=== Don vs Don+新因子 快测 ===")
    factor_for_bt = [r["name"] for r in ic_rows if r["name"] != "disc_lgb_score"][:8]
    if "disc_lgb_score" in new_names:
        factor_for_bt = ["disc_lgb_score"] + factor_for_bt
    bt = _quick_backtest(df, factor_for_bt)
    for k, v in bt.items():
        print(f"  {k:<12} 收益={v['return_pct']:+.1f}%  回撤={v['max_dd']:.1f}%  "
              f"Sharpe={v['sharpe']:.2f}  笔={v['trades']}")

    out = {
        "horizon": args.horizon,
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "lgb_meta": lgb_meta,
        "lgb_ic": lgb_ic,
        "template_top": [{"name": c.name, "ic": c.ic} for c in top_templates],
        "discovered_ic": ic_rows,
        "backtest": bt,
        "factor_for_combo": factor_for_bt,
    }
    path = config.ARTIFACT_PATH / "factor_train.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    print("\n说明: 实盘仍建议 profile=smart；新因子供 don_enhanced / 研究对比。")
    return out


if __name__ == "__main__":
    main()
