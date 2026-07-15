"""
加载已训练模型，导出最新信号

    python predict.py
    python predict.py --threshold 0.00015
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import config
from bars import apply_train_overrides, load_bars, load_gm_parquet
from features import build_feature_matrix, build_features
from model import FgLgbModel
from position import score_to_position, smooth_position


def _load_model_and_features(df, use_wf: bool, use_fd_vib: bool):
    if use_wf:
        model_path = config.ARTIFACTS / "fg_lgb_wf_last.pkl"
        feats = build_feature_matrix(df, use_fd_vib=config.USE_FD_VIB_IN_WF)
        tag = "fg_lgb_wf"
    else:
        model_path = config.ARTIFACTS / "fg_lgb.pkl"
        feats = build_features(df)
        tag = "fg_lgb"
    if not model_path.exists():
        raise SystemExit(f"未找到模型 {model_path}，请先训练")
    return FgLgbModel.load(model_path), feats, model_path, tag


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--wf", action="store_true", help="使用 Walk-forward 最后一折模型")
    parser.add_argument("--source", default="auto", choices=["auto", "gm", "gm_raw", "dominant"])
    args = parser.parse_args()

    if args.source in ("gm", "auto") and config.TRAIN_SOURCE_META.exists():
        apply_train_overrides()

    df = load_bars(args.source)
    model, feats, model_path, tag = _load_model_and_features(df, args.wf, False)

    valid = feats.notna().all(axis=1)
    score = model.predict_frame(feats.loc[valid])
    pos = smooth_position(score_to_position(score, args.threshold), bars=2)

    last_i = score.index[-1]
    last_score = float(score.iloc[-1])
    last_pos = int(pos.iloc[-1])
    last_dt = df.loc[last_i, "datetime"]

    live = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "datetime": str(last_dt),
        "score": last_score,
        "target_position": last_pos,
        "action": {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(last_pos, "FLAT"),
        "model": tag,
        "threshold": args.threshold or config.SIGNAL_THRESHOLD,
    }

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = config.ARTIFACTS / "live_signal.json"
    out_path.write_text(json.dumps(live, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"最新信号 {last_dt}  score={last_score:+.6f}  position={last_pos}  ({live['action']})")
    print(f"-> {out_path}")
    return live


if __name__ == "__main__":
    main()
