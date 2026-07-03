"""
步骤 02：训练 LightGBM 模型并生成预测信号

流程：
  1. 载入数据 -> 构造玻璃因子集（含预处理器）
  2. LightGBM 拟合训练集，用验证集早停
  3. 计算三段样本的 IC / RankIC（信息系数，衡量预测与真实收益的相关性）
  4. 对 TEST 段生成预测信号，落地到 AlphaLab 供回测使用
  5. 保存模型
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from vnpy.alpha import AlphaLab, Segment
from vnpy.alpha.model.models.lgb_model import LgbModel

import config
import data_loader
from dataset import build_dataset
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


def main() -> pl.DataFrame:
    config.ensure_dirs()

    # 1. 数据 + 因子
    print("[02] 构造因子数据集 ...")
    dataset = build_dataset(data_loader.load_polars_df())

    # 2. 训练
    print("[02] 训练 LightGBM ...")
    model = LgbModel(**config.LGB_PARAMS)
    model.fit(dataset)

    # 3. 样本内外 IC
    print("\n[02] 信息系数（IC / RankIC）：")
    for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        infer = dataset.fetch_infer(seg).sort(["datetime", "vt_symbol"])
        pred = model.predict(dataset, seg)
        label = infer["label"].to_numpy()
        ic, ric = _ic(pred, label)
        print(f"    {seg.name:<6}  IC={ic:+.4f}   RankIC={ric:+.4f}   n={len(pred)}")

    # 4. TEST 段信号落地
    infer_test = dataset.fetch_infer(Segment.TEST).sort(["datetime", "vt_symbol"])
    pred_test = model.predict(dataset, Segment.TEST)
    signal_df = infer_test.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series(SIGNAL_COL, pred_test)
    )

    lab = AlphaLab(str(config.LAB_PATH))
    lab.save_signal(config.SIGNAL_NAME, signal_df)
    lab.save_model(config.MODEL_NAME, model)
    print(f"\n[02] 已保存信号 '{config.SIGNAL_NAME}'（{signal_df.height} 行）"
          f"与模型 '{config.MODEL_NAME}'")

    return signal_df


if __name__ == "__main__":
    main()
