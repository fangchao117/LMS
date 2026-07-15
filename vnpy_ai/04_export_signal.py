"""从已训练模型导出 signal 文件（datetime, vt_symbol, signal）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from vnpy.alpha import AlphaLab, Segment

from fg_dataset import FgAlphaDemo  # noqa: F401 — pickle 反序列化需要

LAB_PATH = Path(__file__).resolve().parent / "lab"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def build_signal(model, dataset, segment: Segment) -> pl.DataFrame:
    infer_df = dataset.fetch_infer(segment)
    pred = model.predict(dataset, segment)
    return infer_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", pred)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="FG609")
    parser.add_argument("--dataset", default=None, help="dataset 名称，默认 {symbol}_demo")
    parser.add_argument("--model", default=None, help="model 名称，默认 {symbol}_lgb")
    parser.add_argument("--segment", default="TEST", choices=["TRAIN", "VALID", "TEST"])
    parser.add_argument("--lab", default=str(LAB_PATH))
    args = parser.parse_args()

    dataset_name = args.dataset or f"{args.symbol}_demo"
    model_name = args.model or f"{args.symbol}_lgb"
    segment = Segment[args.segment]

    lab = AlphaLab(args.lab)
    dataset = lab.load_dataset(dataset_name)
    model = lab.load_model(model_name)
    if dataset is None or model is None:
        raise SystemExit(
            f"找不到 dataset={dataset_name} 或 model={model_name}\n"
            f"请先运行: python 03_alpha_demo_lgb.py --symbol {args.symbol}"
        )

    signal = build_signal(model, dataset, segment)
    signal_name = f"{args.symbol}_lgb_{args.segment.lower()}"
    lab.save_signal(signal_name, signal)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stats = {
        "signal_name": signal_name,
        "rows": int(signal.height),
        "segment": args.segment,
        "signal_mean": float(signal["signal"].mean()),
        "signal_std": float(signal["signal"].std()),
        "path": str(Path(args.lab) / "signal" / f"{signal_name}.parquet"),
    }
    out = ARTIFACTS / f"signal_{signal_name}.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"信号已导出: {signal_name}")
    print(f"  行数={stats['rows']}  mean={stats['signal_mean']:.6f}  std={stats['signal_std']:.6f}")
    print(f"  文件: {stats['path']}")
    print("下一步: python 05_alpha_backtest.py")


if __name__ == "__main__":
    main()
