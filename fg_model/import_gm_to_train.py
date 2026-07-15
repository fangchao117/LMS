"""
东财 30m 数据 → fg_model 训练缓存

    python import_gm_to_train.py
    python import_gm_to_train.py --symbol FG609 --run-wf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
from bars import load_gm_parquet


def suggest_splits(df: pd.DataFrame) -> dict:
    """按 GM 数据长度建议 WF / 静态切分日期。"""
    dt = pd.to_datetime(df["datetime"])
    n = len(df)
    min_train = max(300, min(600, int(n * 0.35)))

    # WF：首个 OOS 约在 35% 位置
    wf_oos = dt.iloc[min_train].strftime("%Y-%m-%d")

    # 静态 train/valid/test（后 40% 做 test）
    i70 = int(n * 0.55)
    i85 = int(n * 0.75)
    train_end = dt.iloc[i70].strftime("%Y-%m-%d")
    valid_end = dt.iloc[i85].strftime("%Y-%m-%d")
    test_start = dt.iloc[i85 + 1].strftime("%Y-%m-%d") if i85 + 1 < n else valid_end

    return {
        "wf_oos_start": wf_oos,
        "wf_min_train_bars": min_train,
        "train_end": train_end,
        "valid_end": valid_end,
        "test_start": test_start,
    }


def import_gm(
    symbol: str = "FG609",
    gm_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    sym = symbol.upper().replace(".CZCE", "")
    if gm_path is None:
        gm_path = config.DATA_DIR / f"bars_30m_gm_{sym}.parquet"
        if not gm_path.exists():
            gm_path = config.DATA_DIR / "bars_30m_gm_FG_continuous.parquet"

    df = load_gm_parquet(gm_path)
    splits = suggest_splits(df)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TRAIN_BARS_PATH, index=False)

    meta = {
        "source": "eastmoney_gm",
        "symbol": sym,
        "gm_path": str(gm_path),
        "train_bars_path": str(config.TRAIN_BARS_PATH),
        "bars": len(df),
        "datetime_min": str(df["datetime"].min()),
        "datetime_max": str(df["datetime"].max()),
        **splits,
        "note": "仅 fg_model 训练用，不影响 fd_vib",
    }
    config.TRAIN_SOURCE_META.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return df, meta


def main() -> dict:
    parser = argparse.ArgumentParser(description="东财 GM 30m → fg_model 训练缓存")
    parser.add_argument("--symbol", default="FG609")
    parser.add_argument("--gm-path", default=None, help="指定 GM parquet 路径")
    parser.add_argument("--run-wf", action="store_true", help="导入后立即 Walk-forward 训练")
    parser.add_argument("--no-fd-vib", action="store_true", help="--run-wf 时不接 fd_vib 因子")
    args = parser.parse_args()

    gm_path = Path(args.gm_path) if args.gm_path else None
    df, meta = import_gm(args.symbol, gm_path)

    print(f"[import] GM → 训练缓存  bars={meta['bars']}")
    print(f"  时间 {meta['datetime_min']} ~ {meta['datetime_max']}")
    print(f"  WF  OOS 自 {meta['wf_oos_start']}  min_train={meta['wf_min_train_bars']}")
    print(f"  静态切分 train≤{meta['train_end']}  valid≤{meta['valid_end']}  test≥{meta['test_start']}")
    print(f"-> {config.TRAIN_BARS_PATH}")
    print(f"-> {config.TRAIN_SOURCE_META}")

    if args.run_wf:
        print("\n[import] 启动 Walk-forward ...")
        import wf_train

        argv = ["wf_train.py", "--source", "gm"]
        if args.no_fd_vib:
            argv.append("--no-fd-vib")
        import sys

        old = sys.argv
        try:
            sys.argv = argv
            wf_train.main()
        finally:
            sys.argv = old

    return meta


if __name__ == "__main__":
    main()
