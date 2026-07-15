"""训练 / 搜索新因子 —— LightGBM 打分 + 模板穷举 + IC 筛选。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ops import delay, delta, ts_corr, ts_mean, ts_rank, ts_std
from ic import forward_return, spearman_ic


FactorFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class Candidate:
    name: str
    ic: float
    kind: str  # template | lgb
    desc: str


def _oi(df: pd.DataFrame) -> pd.Series:
    if "open_interest" in df.columns:
        return df["open_interest"].astype(float)
    return df["volume"].astype(float) * 0.0


def base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """LGB 输入：价量持仓 + 常用衍生。"""
    c = df["close"].astype(float)
    v = df["volume"].astype(float).replace(0, np.nan)
    oi = _oi(df)
    ret = c.pct_change()
    hl = (df["high"] - df["low"]) / (c + 1e-12)
    out = pd.DataFrame(index=df.index)
    for w in (5, 10, 20, 30):
        out[f"ret_{w}"] = c / delay(c, w) - 1.0
        out[f"vol_r_{w}"] = v / (ts_mean(v, w) + 1e-12) - 1.0
        out[f"oi_r_{w}"] = oi / (ts_mean(oi, w) + 1e-12) - 1.0
        out[f"std_{w}"] = ts_std(ret, w)
        out[f"rank_ret_{w}"] = ts_rank(ret, w)
        out[f"rank_vol_{w}"] = ts_rank(v, w)
        out[f"corr_ro_{w}"] = ts_corr(ret, oi.pct_change(), w)
    out["hl_ratio"] = hl / (ts_mean(hl, 20) + 1e-12) - 1.0
    out["body"] = (c - df["open"].astype(float)) / (c + 1e-12)
    return out.replace([np.inf, -np.inf], np.nan)


def template_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """GTJA / WorldQuant 风格模板组合（单品种时序版）。"""
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    oi = _oi(df)
    ret = c.pct_change()
    oi_ret = oi.pct_change()
    hl = (df["high"] - df["low"]) / (c + 1e-12)
    out: dict[str, pd.Series] = {}

    for w in (5, 10, 20, 30, 40):
        out[f"disc_rank_ret_vol_{w}"] = -ts_rank(ret, w) * ts_rank(v, w)
        out[f"disc_corr_ret_oi_{w}"] = ts_corr(ret, oi_ret, w)
        out[f"disc_mom_vol_{w}"] = (c / delay(c, w) - 1.0) * ts_rank(v, w)
        out[f"disc_oi_lead_{w}"] = ts_rank(oi_ret, w) * np.sign(delay(ret, 1))
        out[f"disc_range_vol_{w}"] = -ts_rank(hl, w) * ts_rank(v, w)
        out[f"disc_vol_shock_{w}"] = ts_std(ret, w) / (ts_std(ret, w * 2) + 1e-12) - 1.0
        out[f"disc_accel_{w}"] = delta(c / delay(c, 5) - 1.0, w // 5)
        out[f"disc_trend_vol_{w}"] = np.sign(c - delay(c, w)) * ts_rank(v, w)

    # Don 邻近结构
    for n in (20, 30, 38):
        hi = df["high"].rolling(n, min_periods=1).max()
        lo = df["low"].rolling(n, min_periods=1).min()
        pos = (c - lo) / (hi - lo + 1e-12) - 0.5
        out[f"disc_don_pos_{n}"] = pos * (hi - lo) / (ts_mean(hi - lo, n) + 1e-12)

    return out


def screen_candidates(
    df: pd.DataFrame,
    candidates: dict[str, pd.Series],
    horizon: int,
    min_abs_ic: float = 0.03,
) -> list[Candidate]:
    fwd = forward_return(df["close"].astype(float), horizon)
    rows: list[Candidate] = []
    for name, ser in candidates.items():
        ic = spearman_ic(ser.astype(float), fwd)
        if abs(ic) < min_abs_ic:
            continue
        rows.append(Candidate(name=name, ic=ic, kind="template", desc="template_combo"))
    rows.sort(key=lambda x: abs(x.ic), reverse=True)
    return rows


def train_lgb_score(
    df: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
    train_end: str,
    valid_end: str,
) -> tuple[pd.Series, dict]:
    """Walk-forward LightGBM，输出逐 bar 预测分（新因子 disc_lgb_score）。"""
    import lightgbm as lgb

    fwd = forward_return(df["close"].astype(float), horizon)
    data = features.copy()
    data["label"] = fwd
    data = data.replace([np.inf, -np.inf], np.nan)

    idx = pd.to_datetime(df["datetime"]) if "datetime" in df.columns else df.index
    data["_dt"] = idx
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)

    train_mask = data["_dt"] <= train_end_ts
    valid_mask = (data["_dt"] > train_end_ts) & (data["_dt"] <= valid_end_ts)
    test_mask = data["_dt"] > valid_end_ts

    feat_cols = [c for c in features.columns]
    train_df = data.loc[train_mask].dropna(subset=["label"] + feat_cols)
    valid_df = data.loc[valid_mask].dropna(subset=["label"] + feat_cols)
    if len(train_df) < 200 or len(valid_df) < 50:
        raise ValueError("训练/验证样本不足，请检查日期区间或数据长度")

    dtrain = lgb.Dataset(train_df[feat_cols], label=train_df["label"])
    dvalid = lgb.Dataset(valid_df[feat_cols], label=valid_df["label"])

    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    # 全样本推断（因果：仅用当时可得特征，模型在 train+valid 上拟合）
    infer = data[feat_cols].copy()
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    valid_idx = infer.notna().all(axis=1)
    if valid_idx.any():
        pred.loc[valid_idx] = booster.predict(infer.loc[valid_idx])

    ic_all = spearman_ic(pred, fwd)
    ic_test = spearman_ic(pred[test_mask], fwd[test_mask]) if test_mask.any() else 0.0
    imp = sorted(
        zip(feat_cols, booster.feature_importance(importance_type="gain")),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    meta = {
        "ic_all": ic_all,
        "ic_test": ic_test,
        "best_iteration": int(booster.best_iteration or 0),
        "top_features": [{"name": n, "gain": int(g)} for n, g in imp],
        "train_end": train_end,
        "valid_end": valid_end,
        "horizon": horizon,
    }
    return pred.rename("disc_lgb_score"), meta


def write_discovered_module(
    path,
    template_top: list[Candidate],
    include_lgb: bool = True,
) -> None:
    """把 TOP 模板因子写成可 import 的 discovered.py。"""
    lines = [
        '"""',
        "第七梯队 —— 因子训练产出（factor_train.py 自动生成，勿手改）",
        '"""',
        "from __future__ import annotations",
        "",
        "from typing import Callable",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        "from ops import delay, delta, ts_corr, ts_mean, ts_rank, ts_std",
        "",
        "FactorFn = Callable[[pd.DataFrame], pd.Series]",
        "_REGISTRY: dict[str, FactorFn] = {}",
        "",
        "",
        "def register(name: str):",
        "    def wrap(fn: FactorFn) -> FactorFn:",
        "        _REGISTRY[name] = fn",
        "        return fn",
        "    return wrap",
        "",
        "",
        "def all_factors() -> dict[str, FactorFn]:",
        "    return dict(_REGISTRY)",
        "",
        "",
        "def compute_panel(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:",
        "    names = names or list(_REGISTRY.keys())",
        "    return pd.DataFrame({n: _REGISTRY[n](df) for n in names if n in _REGISTRY})",
        "",
        "",
        "def _oi(df: pd.DataFrame) -> pd.Series:",
        '    if "open_interest" in df.columns:',
        '        return df["open_interest"].astype(float)',
        '    return df["volume"].astype(float) * 0.0',
        "",
    ]

    # 从 template_candidates 反查实现：按名称匹配写函数体
    # 为简洁，训练时把 series 公式存 JSON，这里用 exec 式模板映射
    name_set = {c.name for c in template_top}

    for cand in template_top:
        fn_body = _codegen_factor_fn(cand.name)
        lines.append(fn_body)
        lines.append("")

    if include_lgb:
        lines.extend([
            "",
            "@register(\"disc_lgb_score\")",
            "def disc_lgb_score(df: pd.DataFrame) -> pd.Series:",
            "    \"\"\"ML 预测分；完整重算请运行 factor_train.py。\"\"\"",
            "    import json",
            "    from pathlib import Path",
            "    import config",
            "    p = config.ARTIFACT_PATH / \"disc_lgb_score.parquet\"",
            "    if not p.exists():",
            "        return pd.Series(0.0, index=df.index)",
            "    ser = pd.read_parquet(p)[\"disc_lgb_score\"]",
            "    ser.index = df.index[: len(ser)]",
            "    return ser.reindex(df.index).fillna(0.0)",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def _codegen_factor_fn(name: str) -> str:
    """按命名约定生成因子函数源码。"""
    parts = name.split("_")
    if name.startswith("disc_rank_ret_vol_"):
        w = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c, v = df['close'].astype(float), df['volume'].astype(float)\n"
            f"    ret = c.pct_change()\n"
            f"    return -ts_rank(ret, {w}) * ts_rank(v, {w})"
        )
    if name.startswith("disc_corr_ret_oi_"):
        w = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c = df['close'].astype(float)\n"
            f"    ret, oi_ret = c.pct_change(), _oi(df).pct_change()\n"
            f"    return ts_corr(ret, oi_ret, {w})"
        )
    if name.startswith("disc_mom_vol_"):
        w = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c, v = df['close'].astype(float), df['volume'].astype(float)\n"
            f"    return (c / delay(c, {w}) - 1.0) * ts_rank(v, {w})"
        )
    if name.startswith("disc_oi_lead_"):
        w = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    ret = df['close'].astype(float).pct_change()\n"
            f"    return ts_rank(_oi(df).pct_change(), {w}) * np.sign(delay(ret, 1))"
        )
    if name.startswith("disc_range_vol_"):
        w = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c = df['close'].astype(float)\n"
            f"    hl = (df['high'] - df['low']) / (c + 1e-12)\n"
            f"    v = df['volume'].astype(float)\n"
            f"    return -ts_rank(hl, {w}) * ts_rank(v, {w})"
        )
    if name.startswith("disc_vol_shock_"):
        w = parts[-1]
        w2 = int(w) * 2
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    ret = df['close'].astype(float).pct_change()\n"
            f"    return ts_std(ret, {w}) / (ts_std(ret, {w2}) + 1e-12) - 1.0"
        )
    if name.startswith("disc_accel_"):
        w = parts[-1]
        d = max(1, int(w) // 5)
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c = df['close'].astype(float)\n"
            f"    mom5 = c / delay(c, 5) - 1.0\n"
            f"    return delta(mom5, {d})"
        )
    if name.startswith("disc_trend_vol_"):
        w = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c, v = df['close'].astype(float), df['volume'].astype(float)\n"
            f"    return np.sign(c - delay(c, {w})) * ts_rank(v, {w})"
        )
    if name.startswith("disc_don_pos_"):
        n = parts[-1]
        return (
            f'@register("{name}")\n'
            f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
            f"    c = df['close'].astype(float)\n"
            f"    hi = df['high'].rolling({n}, min_periods=1).max()\n"
            f"    lo = df['low'].rolling({n}, min_periods=1).min()\n"
            f"    pos = (c - lo) / (hi - lo + 1e-12) - 0.5\n"
            f"    return pos * (hi - lo) / (ts_mean(hi - lo, {n}) + 1e-12)"
        )
    return (
        f'@register("{name}")\n'
        f"def {name}(df: pd.DataFrame) -> pd.Series:\n"
        f"    return pd.Series(0.0, index=df.index)"
    )
