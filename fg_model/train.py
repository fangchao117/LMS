"""
训练玻璃主力 ML 模型

    cd D:\\LMS\\fg_model
    python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from backtest import run_backtest
from bars import apply_train_overrides, load_bars
from dataset import make_xy, split_by_date
from features import build_features, build_label
from model import FgLgbModel
from position import score_to_position, smooth_position


def main() -> dict:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", choices=["auto", "gm", "gm_raw", "dominant"])
    args = parser.parse_args()

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.source in ("gm", "auto") and config.TRAIN_SOURCE_META.exists():
        apply_train_overrides()
    df = load_bars(args.source)
    print(f"[train] 数据源={args.source}  bars={len(df)}")
    feats = build_features(df)
    label = build_label(df)
    masks = split_by_date(df)

    x_train, y_train = make_xy(df, feats, label, masks["train"])
    x_valid, y_valid = make_xy(df, feats, label, masks["valid"])
    x_test, y_test = make_xy(df, feats, label, masks["test"])

    print(f"[train] bars={len(df)}  train={len(x_train)}  valid={len(x_valid)}  test={len(x_test)}")

    model = FgLgbModel()
    report = model.fit(x_train, y_train, x_valid, y_valid)
    ic_test = model.evaluate_test(x_test, y_test)

    model_path = config.ARTIFACTS / "fg_lgb.pkl"
    model.save(model_path)
    model.save_report_json(config.ARTIFACTS / "train_report.json")

    # 测试集预测 + 回测
    score = model.predict_frame(feats.loc[masks["test"]])
    pos = smooth_position(score_to_position(score), bars=2)
    test_df = df.loc[masks["test"]].reset_index(drop=True)
    pos = pos.reset_index(drop=True)
    bt_stats, eq = run_backtest(test_df, pos)

    out = {
        "model": str(model_path),
        "ic_train": report.ic_train,
        "ic_valid": report.ic_valid,
        "ic_test": ic_test,
        "best_iteration": report.best_iteration,
        "top_features": report.top_features[:8],
        "backtest_test": bt_stats,
    }

    pred_path = config.ARTIFACTS / "test_predictions.parquet"
    pd_out = test_df[["datetime", "close"]].copy()
    pd_out["score"] = score.to_numpy()
    pd_out["position"] = pos.to_numpy()
    pd_out.to_parquet(pred_path, index=False)

    eq_path = config.ARTIFACTS / "test_equity.csv"
    eq.to_csv(eq_path, index=False)

    summary_path = config.ARTIFACTS / "summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  IC  train={report.ic_train:+.4f}  valid={report.ic_valid:+.4f}  test={ic_test:+.4f}")
    print(f"  迭代={report.best_iteration}  模型={model_path}")
    print(f"  测试回测  收益={bt_stats['total_return_pct']:+.1f}%  "
          f"回撤={bt_stats['max_ddpercent']:.1f}%  Sharpe={bt_stats['sharpe_ratio']:.2f}")
    print(f"-> {summary_path}")
    return out


if __name__ == "__main__":
    main()
