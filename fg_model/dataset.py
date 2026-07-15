"""按日期切分 train / valid / test。"""
from __future__ import annotations

import pandas as pd

import config


def split_by_date(df: pd.DataFrame) -> dict[str, pd.Series]:
    dt = pd.to_datetime(df["datetime"])
    return {
        "train": dt <= pd.Timestamp(config.TRAIN_END),
        "valid": (dt > pd.Timestamp(config.TRAIN_END)) & (dt <= pd.Timestamp(config.VALID_END)),
        "test": dt >= pd.Timestamp(config.TEST_START),
    }


def make_xy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    label: pd.Series,
    mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    m = mask & label.notna() & features.notna().all(axis=1)
    return features.loc[m], label.loc[m]
