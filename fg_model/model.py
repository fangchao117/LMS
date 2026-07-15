"""LightGBM 模型封装。"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from features import spearman_ic


@dataclass
class TrainReport:
    ic_train: float
    ic_valid: float
    ic_test: float
    best_iteration: int
    n_features: int
    top_features: list[dict[str, Any]]


class FgLgbModel:
    """玻璃主力 30m 收益预测模型。"""

    def __init__(self) -> None:
        self.booster: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self.report: TrainReport | None = None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> TrainReport:
        self.feature_names = list(x_train.columns)
        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_valid, label=y_valid)

        self.booster = lgb.train(
            config.LGB_PARAMS,
            dtrain,
            num_boost_round=config.NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING, verbose=False)],
        )

        ic_train = spearman_ic(self.predict_frame(x_train), y_train)
        ic_valid = spearman_ic(self.predict_frame(x_valid), y_valid)
        imp = sorted(
            zip(self.feature_names, self.booster.feature_importance(importance_type="gain")),
            key=lambda t: t[1],
            reverse=True,
        )
        self.report = TrainReport(
            ic_train=ic_train,
            ic_valid=ic_valid,
            ic_test=0.0,
            best_iteration=int(self.booster.best_iteration or 0),
            n_features=len(self.feature_names),
            top_features=[{"name": n, "gain": int(g)} for n, g in imp[:12]],
        )
        return self.report

    def predict_frame(self, x: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("模型未训练")
        cols = [c for c in self.feature_names if c in x.columns]
        pred = self.booster.predict(x[cols])
        return pd.Series(pred, index=x.index, name="score")

    def evaluate_test(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        ic = spearman_ic(self.predict_frame(x_test), y_test)
        if self.report:
            self.report.ic_test = ic
        return ic

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "report": asdict(self.report) if self.report else None,
            "booster": self.booster,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "FgLgbModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        m = cls()
        m.feature_names = payload["feature_names"]
        m.booster = payload["booster"]
        if payload.get("report"):
            m.report = TrainReport(**payload["report"])
        return m

    def save_report_json(self, path: Path) -> None:
        if not self.report:
            return
        path.write_text(
            json.dumps(asdict(self.report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
