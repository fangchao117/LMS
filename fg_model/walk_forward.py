"""Walk-forward 滚动训练 —— 样本外拼接评估。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

import config
from backtest import run_backtest
from dataset import make_xy
from features import build_feature_matrix, build_label, spearman_ic
from model import FgLgbModel
from position import score_to_position, smooth_position


@dataclass
class WfFold:
    fold: int
    predict_start: str
    predict_end: str
    train_bars: int
    valid_bars: int
    predict_bars: int
    ic_valid: float
    ic_oos: float
    best_iteration: int


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s)


def build_folds(df: pd.DataFrame) -> list[dict[str, pd.Series]]:
    """按月滚动：train → valid(早停) → predict(OOS)。"""
    dt = _dt(df["datetime"])
    start = pd.Timestamp(config.WF_OOS_START)
    end = dt.max()
    folds: list[dict[str, pd.Series]] = []
    cur = start
    fold_id = 0

    while cur < end:
        pred_end = min(cur + pd.Timedelta(days=config.WF_STEP_DAYS), end + pd.Timedelta(days=1))
        valid_end = cur
        valid_start = cur - pd.Timedelta(days=config.WF_VALID_DAYS)
        train_mask = dt < valid_start
        valid_mask = (dt >= valid_start) & (dt < valid_end)
        predict_mask = (dt >= cur) & (dt < pred_end)

        if predict_mask.sum() == 0:
            break
        if train_mask.sum() < config.WF_MIN_TRAIN_BARS:
            cur = pred_end
            continue
        if valid_mask.sum() < 30:
            cur = pred_end
            continue

        fold_id += 1
        folds.append({
            "fold": fold_id,
            "predict_start": str(cur.date()),
            "predict_end": str((pred_end - pd.Timedelta(days=1)).date()),
            "train": train_mask,
            "valid": valid_mask,
            "predict": predict_mask,
        })
        cur = pred_end

    return folds


def run_walk_forward(
    df: pd.DataFrame,
    use_fd_vib: bool | None = None,
) -> dict[str, Any]:
    use_fv = config.USE_FD_VIB_IN_WF if use_fd_vib is None else use_fd_vib
    feats = build_feature_matrix(df, use_fd_vib=use_fv)
    label = build_label(df)
    folds = build_folds(df)

    oos_score = pd.Series(np.nan, index=df.index, dtype=float)
    fold_rows: list[WfFold] = []

    print(f"[wf] folds={len(folds)}  features={feats.shape[1]}  fd_vib={use_fv}")

    for spec in folds:
        fid = spec["fold"]
        x_train, y_train = make_xy(df, feats, label, spec["train"])
        x_valid, y_valid = make_xy(df, feats, label, spec["valid"])
        x_pred = feats.loc[spec["predict"]].dropna()
        if x_train.empty or x_valid.empty or x_pred.empty:
            continue

        model = FgLgbModel()
        report = model.fit(x_train, y_train, x_valid, y_valid)
        pred_idx = x_pred.index
        pred = model.predict_frame(x_pred)
        oos_score.loc[pred_idx] = pred.to_numpy()

        y_oos = label.loc[pred_idx]
        ic_oos = spearman_ic(pred, y_oos)

        fold_rows.append(WfFold(
            fold=fid,
            predict_start=spec["predict_start"],
            predict_end=spec["predict_end"],
            train_bars=int(spec["train"].sum()),
            valid_bars=int(spec["valid"].sum()),
            predict_bars=int(spec["predict"].sum()),
            ic_valid=report.ic_valid,
            ic_oos=ic_oos,
            best_iteration=report.best_iteration,
        ))
        print(f"  fold {fid:02d}  {spec['predict_start']}~{spec['predict_end']}  "
              f"valid_ic={report.ic_valid:+.3f}  oos_ic={ic_oos:+.3f}  "
              f"pred={len(pred)}")

    oos_mask = oos_score.notna()
    ic_all = spearman_ic(oos_score, label)
    ic_all_masked = spearman_ic(oos_score[oos_mask], label[oos_mask])

    pos = smooth_position(score_to_position(oos_score[oos_mask]), bars=2)
    bt_df = df.loc[oos_mask].reset_index(drop=True)
    pos = pos.reset_index(drop=True)
    bt_stats, eq = run_backtest(bt_df, pos)

    # 最后一折模型用于 predict --wf
    last_model_path = config.ARTIFACTS / "fg_lgb_wf_last.pkl"
    if fold_rows:
        last = folds[-1]
        x_train, y_train = make_xy(df, feats, label, last["train"])
        x_valid, y_valid = make_xy(df, feats, label, last["valid"])
        if not x_train.empty and not x_valid.empty:
            final = FgLgbModel()
            final.fit(x_train, y_train, x_valid, y_valid)
            final.save(last_model_path)

    out = {
        "mode": "walk_forward",
        "use_fd_vib": use_fv,
        "n_folds": len(fold_rows),
        "ic_oos_all": ic_all_masked,
        "ic_oos_full_index": ic_all,
        "backtest_oos": bt_stats,
        "folds": [asdict(f) for f in fold_rows],
        "last_model": str(last_model_path) if last_model_path.exists() else None,
    }
    return out, oos_score, eq, feats


def save_wf_results(
    result: dict[str, Any],
    oos_score: pd.Series,
    eq: pd.DataFrame,
    df: pd.DataFrame,
    suffix: str = "",
) -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    oos_mask = oos_score.notna()
    pred = df.loc[oos_mask, ["datetime", "close"]].copy()
    pred["score"] = oos_score[oos_mask].to_numpy()
    pred["position"] = smooth_position(
        score_to_position(oos_score[oos_mask]), bars=2
    ).to_numpy()
    pred.to_parquet(config.ARTIFACTS / f"wf_predictions{suffix}.parquet", index=False)
    eq.to_csv(config.ARTIFACTS / f"wf_equity{suffix}.csv", index=False)
    (config.ARTIFACTS / f"wf_summary{suffix}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not suffix:
        (config.ARTIFACTS / "wf_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
