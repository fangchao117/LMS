"""
玻璃期货 LightGBM 模型 —— 强正则 + 训练/验证合并拟合 + 时间切分早停。
"""
from __future__ import annotations

from typing import cast

import lightgbm as lgb
import numpy as np
import polars as pl

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model.models.lgb_model import LgbModel


class GlassLgbModel(LgbModel):
    """带正则化的 LightGBM，在 TRAIN+VALID 合并数据上拟合。"""

    def __init__(
        self,
        learning_rate: float = 0.02,
        num_leaves: int = 8,
        num_boost_round: int = 400,
        early_stopping_rounds: int = 60,
        log_evaluation_period: int = 0,
        seed: int | None = None,
        min_data_in_leaf: int = 80,
        lambda_l1: float = 0.5,
        lambda_l2: float = 0.5,
        max_depth: int = 4,
        feature_fraction: float = 0.7,
        bagging_fraction: float = 0.7,
        bagging_freq: int = 5,
        valid_ratio: float = 0.15,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            log_evaluation_period=log_evaluation_period,
            seed=seed,
        )
        self.params.update(
            {
                "min_data_in_leaf": min_data_in_leaf,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "max_depth": max_depth,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "bagging_freq": bagging_freq,
                "verbose": -1,
            }
        )
        self.valid_ratio = valid_ratio

    def fit(self, dataset: AlphaDataset) -> None:
        """TRAIN+VALID 合并后，用末尾 valid_ratio 做早停。"""
        tr = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        va = dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"])
        df = pl.concat([tr, va])

        n_valid = max(int(df.height * self.valid_ratio), 60)
        train_df = df.head(df.height - n_valid)
        valid_df = df.tail(n_valid)

        def _to_lgb(part: pl.DataFrame) -> lgb.Dataset:
            x = part.select(part.columns[2:-1]).to_pandas()
            y = np.array(part["label"])
            return lgb.Dataset(x, label=y)

        ds_train = _to_lgb(train_df)
        ds_valid = _to_lgb(valid_df)

        self.model = lgb.train(
            self.params,
            ds_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[ds_train, ds_valid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(self.log_evaluation_period),
            ],
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        df = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
        data = df.select(df.columns[2:-1]).to_numpy()
        return cast(np.ndarray, self.model.predict(data))
