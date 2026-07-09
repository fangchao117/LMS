"""
VALID 段参数扫描 —— 收益最大化

在 VALID（2022-01 ~ 2023-06）上扫描均线 / 手数 / 过滤参数，
以总收益最高为优选目标，再在 TEST 段样本外验证。

    python tune_valid.py
    python tune_valid.py --quick          # 仅扫均线+手数
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product

import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

import config
import data
from signals import generate_signals
from strategy import ShortTermStrategy


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _run(
    lab: AlphaLab,
    signal_df: pl.DataFrame,
    start: str,
    end: str,
    *,
    max_lots: int,
    stop_loss_pct: float,
    max_drawdown_pct: float,
) -> dict:
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[config.VT_SYMBOL],
        interval=Interval.DAILY,
        start=_to_dt(start),
        end=_to_dt(end),
        capital=config.CAPITAL,
        annual_days=240,
    )
    engine.add_strategy(
        ShortTermStrategy,
        {
            "signal_threshold": config.SIGNAL_THRESHOLD,
            "position_pct": config.POSITION_PCT,
            "margin_rate": config.MARGIN_RATE,
            "max_lots": max_lots,
            "price_add_ticks": config.PRICE_ADD_TICKS,
            "stop_loss_pct": stop_loss_pct,
            "max_drawdown_pct": max_drawdown_pct,
        },
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics()


def _score_return(stats: dict) -> float:
    """收益最大化：VALID 段总收益为主。"""
    return float(stats.get("total_return", 0) or 0)


def scan(
    lab: AlphaLab,
    bars_df: pl.DataFrame,
    quick: bool = False,
) -> tuple[dict, list[dict]]:
    start, end = config.VALID_PERIOD

    ma_triples = [
        (3, 13, 34), (3, 21, 34), (5, 13, 34), (5, 21, 34),
        (8, 13, 34), (8, 21, 34), (8, 21, 55), (5, 34, 55),
        (10, 21, 55), (5, 13, 55),
    ]
    max_lots_list = [1, 2, 3]
    min_trend_list = [0.0] if quick else [0.0, 0.004, 0.006, 0.008]
    max_dd_list = [0.0] if quick else [0.0, 0.35, 0.50]

    results: list[dict] = []
    best: dict | None = None
    best_score = float("-inf")
    total = len(ma_triples) * len(max_lots_list) * len(min_trend_list) * len(max_dd_list)
    n = 0

    print(f"[tune_valid] VALID {start} ~ {end}  扫描 {total} 组（目标：收益最大化）…")

    for (ms, mm, ml), lots, mt, mdd in product(
        ma_triples, max_lots_list, min_trend_list, max_dd_list
    ):
        n += 1
        sig = generate_signals(
            bars_df,
            mode="trend_ma",
            ma_short=ms,
            ma_medium=mm,
            ma_long=ml,
            min_trend_pct=mt,
        )
        stats = _run(
            lab, sig, start, end,
            max_lots=lots,
            stop_loss_pct=0.0,
            max_drawdown_pct=mdd,
        )
        sc = _score_return(stats)
        row = {
            "ma_short": ms,
            "ma_medium": mm,
            "ma_long": ml,
            "max_lots": lots,
            "min_trend_pct": mt,
            "max_drawdown_pct": mdd,
            "valid_return": sc,
            "valid_sharpe": float(stats.get("sharpe_ratio", 0) or 0),
            "valid_max_dd": float(stats.get("max_ddpercent", 0) or stats.get("max_drawdown", 0) or 0),
            "valid_trades": int(stats.get("total_trade_count", 0) or 0),
        }
        results.append(row)

        if sc > best_score:
            best_score = sc
            best = {**row, "valid_stats": stats}

        if n % 20 == 0 or n == total:
            print(f"    进度 {n}/{total}  当前最优 VALID 收益 {best_score:.1f}%")

    assert best is not None
    results.sort(key=lambda x: x["valid_return"], reverse=True)
    return best, results


def main(quick: bool = False) -> dict:
    config.ensure_dirs()
    bars_df = data.load_polars_df()
    lab = AlphaLab(str(config.LAB_PATH))
    lab.save_bar_data(data.build_bar_data())
    lab.add_contract_setting(
        config.VT_SYMBOL,
        config.LONG_RATE,
        config.SHORT_RATE,
        config.CONTRACT_SIZE,
        config.PRICE_TICK,
    )

    best, all_results = scan(lab, bars_df, quick=quick)

    # TEST 样本外
    sig_test = generate_signals(
        bars_df,
        mode="trend_ma",
        ma_short=best["ma_short"],
        ma_medium=best["ma_medium"],
        ma_long=best["ma_long"],
        min_trend_pct=best["min_trend_pct"],
    )
    test_stats = _run(
        lab, sig_test,
        config.TEST_PERIOD[0], config.TEST_PERIOD[1],
        max_lots=best["max_lots"],
        stop_loss_pct=0.0,
        max_drawdown_pct=best["max_drawdown_pct"],
    )

    def _fmt(v: object) -> object:
        return v.isoformat() if isinstance(v, datetime) else v

    out = {
        "objective": "return_max",
        "valid_period": config.VALID_PERIOD,
        "test_period": config.TEST_PERIOD,
        "best": {k: v for k, v in best.items() if k != "valid_stats"},
        "test_stats": {k: _fmt(v) for k, v in test_stats.items()},
        "top10_valid": all_results[:10],
    }
    out_path = config.ARTIFACT_PATH / "tune_valid.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n[tune_valid] ===== VALID 最优（收益最大化）=====")
    print(
        f"    MA {best['ma_short']}/{best['ma_medium']}/{best['ma_long']}  "
        f"max_lots={best['max_lots']}  min_trend={best['min_trend_pct']:.3f}  "
        f"max_dd熔断={best['max_drawdown_pct']:.0%}"
    )
    print(f"    VALID 总收益   {best['valid_return']:.1f}%")
    print(f"    VALID Sharpe   {best['valid_sharpe']:.2f}")
    print(f"    VALID 最大回撤 {best['valid_max_dd']}")

    tr = float(test_stats.get("total_return", 0) or 0)
    print("\n[tune_valid] ===== TEST 样本外 =====")
    print(f"    总收益         {tr:.1f}%")
    print(f"    Sharpe         {test_stats.get('sharpe_ratio', 0)}")
    print(f"    最大回撤       {test_stats.get('max_ddpercent', test_stats.get('max_drawdown', 0))}")
    print(f"    成交笔数       {test_stats.get('total_trade_count', 0)}")
    print(f"\n[tune_valid] 完整结果 -> {out_path}")

    print("\n[tune_valid] VeighNa 参数建议：")
    print(f"    ma_short={best['ma_short']}  ma_medium={best['ma_medium']}  ma_long={best['ma_long']}")
    print(f"    max_lots={best['max_lots']}  min_trend_pct={best['min_trend_pct']}")
    print(f"    max_drawdown_pct={best['max_drawdown_pct']}  stop_loss_pct=0")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VALID 段收益最大化扫描")
    parser.add_argument("--quick", action="store_true", help="快速模式：仅扫均线+手数")
    args = parser.parse_args()
    main(quick=args.quick)
