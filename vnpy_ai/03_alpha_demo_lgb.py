"""vnpy.alpha 最小示例：FG 单合约 Alpha158 + LightGBM 训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnpy.alpha import AlphaLab, Segment
from vnpy.alpha.model.models.lgb_model import LgbModel

from fg_dataset import FgAlphaDemo

LAB_PATH = Path(__file__).resolve().parent / "lab"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="FG609")
    parser.add_argument("--lab", default=str(LAB_PATH))
    args = parser.parse_args()

    vt_symbol = f"{args.symbol}.CZCE"
    lab = AlphaLab(args.lab)

    df = lab.load_bar_df(
        vt_symbols=[vt_symbol],
        interval="1m",
        start="2025-10-01",
        end="2026-07-01",
        extended_days=60,
    )
    if df is None or df.is_empty():
        raise SystemExit(f"AlphaLab 无数据，请先运行: python 02_import_fg_to_alpha_lab.py --symbol {args.symbol}")

    dataset = FgAlphaDemo(
        df=df,
        train_period=("2025-10-01", "2026-02-28"),
        valid_period=("2026-03-01", "2026-05-15"),
        test_period=("2026-05-16", "2026-07-01"),
    )
    dataset.set_label("ts_delay(close, -3) / close - 1")
    dataset.prepare_data(max_workers=1)

    model = LgbModel(learning_rate=0.05, num_leaves=15, num_boost_round=300, early_stopping_rounds=30)
    model.fit(dataset)
    pred = model.predict(dataset, Segment.TEST)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lab.save_dataset(f"{args.symbol}_demo", dataset)
    lab.save_model(f"{args.symbol}_lgb", model)

    stats = {
        "vt_symbol": vt_symbol,
        "train": dataset.data_periods[Segment.TRAIN],
        "valid": dataset.data_periods[Segment.VALID],
        "test": dataset.data_periods[Segment.TEST],
        "pred_count": int(len(pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "note": "demo only; not for live trading",
    }
    out = ARTIFACTS / f"alpha_demo_{args.symbol}.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"训练完成: {vt_symbol}")
    print(f"测试集预测: n={stats['pred_count']}  mean={stats['pred_mean']:.6f}  std={stats['pred_std']:.6f}")
    print(f"模型已保存: {args.lab}/model/{args.symbol}_lgb")
    print(f"统计: {out}")


if __name__ == "__main__":
    main()
