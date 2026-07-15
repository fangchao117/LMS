"""
fd_vib 主入口 —— IC → 调参 → 回测 → 实盘信号

    python run.py           # 完整流程
    python run.py --quick   # 跳过 tune（用已有 tune.json）
    python run.py --live    # 仅输出盘后仿真信号
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import combo
import config
import ic
from dominant import DOMINANT_SEGMENTS, dominant_period, load_dominant_30m, next_roll_info
from risk_bt import run as run_risk

_ROOT = Path(__file__).resolve().parent


def load_best_spec(profile: str = "max") -> dict:
    tune_path = config.ARTIFACT_PATH / "tune.json"
    if profile == "balanced":
        return dict(config.BALANCED_SPEC)
    if profile == "enhanced":
        return dict(config.ENHANCED_SPEC)
    if profile == "smart":
        return dict(config.SMART_SPEC)
    if profile == "auto":
        return dict(config.AUTO_SPEC)
    if profile == "regime":
        return dict(config.REGIME_SPEC)
    if profile == "lms":
        return dict(config.LMS_SPEC)
    if profile == "lms_dir":
        return dict(config.LMS_DIR_SPEC)
    if profile == "lms_mom":
        return dict(config.LMS_MOM_SPEC)
    if tune_path.exists():
        data = json.loads(tune_path.read_text(encoding="utf-8"))
        return dict(data["best_score"]["spec"])
    return dict(config.DEFAULT_SPEC)


def run_backtest(spec: dict | None = None) -> dict:
    config.ensure_dirs()
    spec = spec or load_best_spec()
    sig_kw = {k: v for k, v in spec.items()
              if k not in ("use_risk", "max_lots", "atr_stop_mult", "trail_atr_mult", "max_loss_pct")}
    ml = spec.get("max_lots", config.MAX_LOTS)
    risk_kw = dict(
        max_lots=ml,
        atr_stop_mult=spec.get("atr_stop_mult", 3.5),
        trail_atr_mult=spec.get("trail_atr_mult", 2.62),
        max_loss_pct=spec.get("max_loss_pct", 6.0),
        handle_roll=True,
    )

    df = load_dominant_30m(adjust=False)
    sig = combo.generate(df, **sig_kw)
    t0 = config.TEST_PERIOD[0]
    df_test = df[df["datetime"] >= t0].reset_index(drop=True)
    sig_test = sig[df["datetime"] >= t0].reset_index(drop=True)

    st_full, eq = run_risk(df, sig, **risk_kw)
    st_test, _ = run_risk(df_test, sig_test, **risk_kw)

    from bridge import load_30m

    segs = []
    for sym, start, end in DOMINANT_SEGMENTS:
        seg = load_30m(sym, start=start, end=end, use_cache=True)
        if len(seg) < 50:
            continue
        s = combo.generate(seg, **sig_kw)
        st, _ = run_risk(seg, s, **risk_kw)
        segs.append({
            "symbol": sym, "start": start, "end": end,
            "return_pct": st["total_return_pct"],
            "max_dd_pct": st["max_ddpercent"],
            "trades": st["total_trades"],
        })

    p0, p1 = dominant_period()
    print(f"\n[fd_vib] 主力量化  {p0} ~ {p1}  未复权+换月平仓")
    print(f"  策略: {spec.get('mode')}  don={spec.get('donchian')}  {ml}手")
    print(f"  FULL  {st_full['total_return_pct']:.1f}%  回撤 {st_full['max_ddpercent']:.1f}%  "
          f"Sharpe {st_full['sharpe_ratio']:.2f}  换月平仓 {st_full.get('roll_closes', 0)} 次")
    print(f"  TEST  {st_test['total_return_pct']:.1f}%  回撤 {st_test['max_ddpercent']:.1f}%")
    print(f"{'合约':<8} {'收益':>8} {'回撤':>8} {'笔':>5}")
    for s in segs:
        mark = "+" if s["return_pct"] > 0 else "-"
        print(f"{s['symbol']:<8} {s['return_pct']:7.1f}% {s['max_dd_pct']:7.1f}% {s['trades']:5d} {mark}")

    out = {
        "spec": spec,
        "roll_model": "平旧仓→新合约开仓，未复权，换月额外手续费",
        "full_stats": st_full,
        "test_stats": st_test,
        "segments": segs,
        "roll_info": next_roll_info(),
    }
    config.ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACT_PATH / "stats_best.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    eq.to_csv(config.ARTIFACT_PATH / "equity_best.csv", index=False)
    return out


def export_live_signal(spec: dict | None = None) -> dict:
    """盘后仿真：下一根 30m 目标仓位"""
    from bridge import load_30m
    from datetime import datetime

    from dominant import current_dominant

    spec = spec or load_best_spec()
    sig_kw = {k: v for k, v in spec.items()
              if k not in ("use_risk", "max_lots", "atr_stop_mult", "trail_atr_mult", "max_loss_pct")}
    sym, seg_start, seg_end = current_dominant()
    roll = next_roll_info()

    df = load_30m(sym, start=seg_start, use_cache=True)
    if df.empty:
        raise RuntimeError(f"无 {sym} 数据")

    sig = combo.generate(df, **sig_kw)
    last_sig = float(sig.iloc[-1])
    target_lots = int(max(-spec.get("max_lots", 2), min(spec.get("max_lots", 2), round(last_sig))))

    # 换月前一日提示：若 segment 最后 3 天，建议平仓
    dt = df["datetime"].iloc[-1]
    days_left = roll["days_to_roll"]
    roll_hint = "hold"
    if days_left <= 3 and target_lots != 0:
        roll_hint = f"距换月 {days_left} 天，换月日需平 {sym} 再开 {roll.get('next_symbol', '?')}"

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bar_time": str(dt),
        "dominant_symbol": sym,
        "vt_symbol": f"{sym}.CZCE",
        "segment": {"start": seg_start, "end": seg_end},
        "roll": roll,
        "roll_hint": roll_hint,
        "strategy": spec,
        "signal_raw": last_sig,
        "target_position": target_lots,
        "max_lots": spec.get("max_lots", 2),
        "action": {1: "做多", -1: "做空", 0: "空仓"}.get(target_lots, "空仓"),
        "note": "T 收盘信号 → 下一根 30m 开盘执行；换月日平旧开新",
    }
    path = config.ARTIFACT_PATH / "live_signal.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[实盘仿真] {sym}  信号={out['action']}  目标={target_lots}手")
    print(f"  换月: {roll['days_to_roll']} 天后 → {roll.get('next_symbol')}")
    print(f"  -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过 tune")
    ap.add_argument("--live", action="store_true", help="仅输出盘后信号")
    ap.add_argument("--profile", choices=["max", "smart", "auto", "regime", "balanced", "enhanced", "lms", "lms_dir", "lms_mom"], default="smart",
                    help="smart=每段auto/regime(推荐) regime=震荡趋势 auto=自校准Don lms=方向+动能")
    args = ap.parse_args()

    if args.live:
        export_live_signal(load_best_spec(args.profile))
        return

    ic.save_screen()
    subprocess.check_call([sys.executable, str(_ROOT / "mine.py")])
    if not args.quick:
        subprocess.check_call([sys.executable, str(_ROOT / "tune.py")])
    run_backtest(load_best_spec(args.profile))
    export_live_signal(load_best_spec(args.profile))


if __name__ == "__main__":
    main()
