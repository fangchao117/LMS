"""
步骤 02：训练 LightGBM 模型并生成预测信号

流程：
  1. 载入数据 -> 构造玻璃因子集（含预处理器）
  2. GlassLgbModel 拟合（训练集内时间切分早停 + 强正则）
  3. 因果滚动 z-score 后处理（消除偏多偏差，统一阈值尺度）
  4. 计算三段样本的 IC / RankIC / 方向命中率
  5. 对 TEST 段生成预测信号，落地到 AlphaLab 供回测使用
  6. 保存模型
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from vnpy.alpha import AlphaLab, Segment

import config
import data_loader
from dataset import build_dataset
from model import GlassLgbModel
from signal_utils import build_processed_predictions, segment_signal_frame
from strategy import SIGNAL_COL


def _ic(pred: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """返回 (Pearson IC, Spearman RankIC)，自动剔除 NaN。"""
    mask = ~np.isnan(pred) & ~np.isnan(label)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    p, l = pred[mask], label[mask]
    ic = float(np.corrcoef(p, l)[0, 1])
    ric = float(spearmanr(p, l).statistic)
    return ic, ric


def _hit_rate(pred: np.ndarray, label: np.ndarray, threshold: float) -> float:
    mask = ~np.isnan(pred) & ~np.isnan(label) & (np.abs(pred) > threshold)
    if mask.sum() < 3:
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(label[mask])))


def main() -> pl.DataFrame:
    config.ensure_dirs()

    print("[02] 构造因子数据集 ...")
    dataset = build_dataset(data_loader.load_polars_df())

    print("[02] 训练 GlassLgbModel（强正则 + 训练集内早停）...")
    model = GlassLgbModel(**config.LGB_PARAMS)
    model.fit(dataset)

    print("[02] 因果滚动 z-score 后处理 ...")
    z_preds = build_processed_predictions(
        dataset,
        model,
        window=config.SIGNAL_ZSCORE_WINDOW,
        min_periods=config.SIGNAL_ZSCORE_MIN_PERIODS,
    )

    print("\n[02] 信息系数（z-score 信号 vs 标签）：")
    for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        infer = dataset.fetch_infer(seg).sort(["datetime", "vt_symbol"])
        pred = z_preds[seg]
        label = infer["label"].to_numpy()
        ic, ric = _ic(pred, label)
        hit = _hit_rate(pred, label, config.SIGNAL_THRESHOLD)
        active = int(np.sum(~np.isnan(pred) & (np.abs(pred) > config.SIGNAL_THRESHOLD)))
        print(
            f"    {seg.name:<6}  IC={ic:+.4f}   RankIC={ric:+.4f}   "
            f"hit@{config.SIGNAL_THRESHOLD}={hit:.3f}   active={active}   n={len(pred)}"
        )

    signal_df = segment_signal_frame(dataset, Segment.TEST, z_preds[Segment.TEST], SIGNAL_COL)

    lab = AlphaLab(str(config.LAB_PATH))
    lab.save_signal(config.SIGNAL_NAME, signal_df)
    lab.save_model(config.MODEL_NAME, model)
    print(
        f"\n[02] 已保存 z-score 信号 '{config.SIGNAL_NAME}'（{signal_df.height} 行）"
        f"与模型 '{config.MODEL_NAME}'"
    )

    return signal_df


if __name__ == "__main__":
    main()
