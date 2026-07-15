"""
Walk-forward 滚动训练 + fd_vib 因子（只读）

    cd D:\\LMS\\fg_model
    python wf_train.py
    python wf_train.py --no-fd-vib     # 对照：不用 fd_vib 因子
    python predict.py --wf             # 用最后一折 WF 模型出信号
"""
from __future__ import annotations

import argparse
import json

import config
from bars import apply_train_overrides, load_bars
from fd_vib_factors import factor_meta, resolve_factor_names
from walk_forward import run_walk_forward, save_wf_results


def main() -> dict:
    parser = argparse.ArgumentParser(description="Walk-forward 滚动训练")
    parser.add_argument("--source", default="auto", choices=["auto", "gm", "gm_raw", "dominant"])
    parser.add_argument("--no-fd-vib", action="store_true", help="不接入 fd_vib 因子")
    parser.add_argument("--top", type=int, default=None, help="fd_vib 因子数量")
    args = parser.parse_args()

    if args.top:
        config.FD_VIB_TOP_N = args.top

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.source in ("gm", "auto") and config.TRAIN_SOURCE_META.exists():
        apply_train_overrides()
    df = load_bars(args.source)
    print(f"[wf] 数据源={args.source}  bars={len(df)}")
    use_fd_vib = not args.no_fd_vib

    if use_fd_vib:
        names = resolve_factor_names(df)
        meta = factor_meta()
        meta["selected"] = names
        (config.ARTIFACTS / "wf_fd_vib_factors.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[wf] fd_vib 因子 {len(names)} 个（只读，不改 fd_vib）")
        for n in names[:8]:
            print(f"    {n}")
        if len(names) > 8:
            print(f"    ... +{len(names) - 8}")

    suffix = "" if use_fd_vib else "_base"
    result, oos_score, eq, _ = run_walk_forward(df, use_fd_vib=use_fd_vib)
    save_wf_results(result, oos_score, eq, df, suffix=suffix)

    bt = result["backtest_oos"]
    print(f"\n=== Walk-forward OOS 汇总 ===")
    print(f"  折数={result['n_folds']}  OOS_IC={result['ic_oos_all']:+.4f}")
    print(f"  收益={bt['total_return_pct']:+.1f}%  回撤={bt['max_ddpercent']:.1f}%  "
          f"Sharpe={bt['sharpe_ratio']:.2f}  笔={bt['total_trades']}")
    print(f"-> {config.ARTIFACTS / f'wf_summary{suffix}.json'}")
    return result


if __name__ == "__main__":
    main()
