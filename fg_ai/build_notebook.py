"""
生成器：产出「玻璃期货AI多模型.ipynb」

只构建 notebook 的 JSON 结构，不做任何训练，弱机器可放心运行：
    python build_notebook.py
生成后用 Jupyter / VSCode 打开 .ipynb，逐格运行即可看到中文图表。
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import config


nb = new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


# ============================================================ 封面
md(
    "# 🔮 玻璃期货（FG）AI 多模型择时研究\n"
    "\n"
    "> 基于 **vnpy 4.4 · `vnpy.alpha`** 框架，对玻璃主力连续 `FG00.CZCE` 做机器学习择时。\n"
    "\n"
    "本 Notebook 一站式演示 **数据 → 因子 → 多模型 → 评估 → 回测** 全流程，全中文、多图表：\n"
    "\n"
    "| 环节 | 内容 |\n"
    "|---|---|\n"
    "| 📈 数据 | 行情/成交量/持仓量可视化 |\n"
    "| 🧮 因子 | 精选量价因子 + 相关性热力图 |\n"
    "| 🤖 模型 | **Lasso · GlassLgbModel · MLP** 三模型对比 |\n"
    "| 🎯 评估 | IC / RankIC、因果 z-score 信号、特征重要性 |\n"
    "| 🏆 回测 | **趋势 + AI 否决**（推荐）vs 纯规则 vs AI 确认 |\n"
    "\n"
    "⚠️ **低配机器**：顶部 `QUICK=True` 为轻量档；单标的日线上 **纯 AI 多空翻转样本外易亏损**，"
    "推荐 `LMS_AI_MODE=\"veto\"`（三均线定方向，AI 仅在强烈反向时否决）。"
)

# ============================================================ 环境
md("## 0️⃣ 环境与配置")
code(
    """
import sys, os, warnings, importlib
from pathlib import Path
warnings.filterwarnings("ignore")

_cwd = Path.cwd()
FG_DIR = _cwd if (_cwd / "config.py").exists() else _cwd / "fg_ai"
if not (FG_DIR / "config.py").exists():
    raise FileNotFoundError(f"找不到 config.py，请 cd 到 fg_ai 再打开 notebook。当前: {_cwd}")
if str(FG_DIR) not in sys.path:
    sys.path.insert(0, str(FG_DIR))
os.chdir(FG_DIR)

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

for _mod in list(sys.modules):
    if _mod == "config" or _mod.startswith("config."):
        del sys.modules[_mod]
import config
importlib.reload(config)
import data_loader
from dataset import build_dataset
from model import GlassLgbModel
from signal_utils import build_processed_predictions
from strategy import SIGNAL_COL
from lms_strategy import LmsAlphaStrategy

QUICK = True
print("工作目录:", FG_DIR)
print("配置就绪 ✅  合约:", config.VT_SYMBOL)
print("  AI 模式:", config.LMS_AI_MODE, "| 否决阈值:", config.LMS_VETO_THRESHOLD)
print("  QUICK =", QUICK)
"""
)

# ============================================================ 数据
md("## 1️⃣ 数据加载与行情可视化 📈")
code(
    """
raw = pd.read_parquet(config.DATA_FILE).sort_values("datetime").reset_index(drop=True)
print(f"数据区间：{raw['datetime'].min().date()} ~ {raw['datetime'].max().date()}  共 {len(raw)} 根日线")
raw.tail()
"""
)
code(
    """
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})

axes[0].plot(raw["datetime"], raw["close"], color="#c0392b", lw=1.1, label="收盘价")
axes[0].plot(raw["datetime"], raw["close"].rolling(60).mean(), color="#2980b9", lw=1, label="MA60")
axes[0].set_title("玻璃期货 FG00.CZCE 主力连续 · 收盘价", fontsize=13)
axes[0].legend(loc="upper right"); axes[0].grid(alpha=.3)

axes[1].bar(raw["datetime"], raw["volume"], color="#7f8c8d", width=1.0)
axes[1].set_ylabel("成交量"); axes[1].grid(alpha=.3)

axes[2].plot(raw["datetime"], raw["open_interest"], color="#27ae60", lw=1)
axes[2].set_ylabel("持仓量"); axes[2].grid(alpha=.3)

plt.tight_layout(); plt.show()
"""
)

# ============================================================ 因子
md(
    "## 2️⃣ 因子工程 🧮\n"
    "复用 `dataset.py` 的 `GlassAlphaDataset`：精选量价因子（K线形态/动量/均线/波动/通道/摆动/量能/持仓），"
    "训练窗口拟合的稳健归一化，杜绝数据泄漏。"
)
code(
    """
from vnpy.alpha import Segment

df_pl = data_loader.load_polars_df()
dataset = build_dataset(df_pl, max_workers=1)

train_learn = dataset.fetch_learn(Segment.TRAIN)
feature_cols = train_learn.columns[2:-1]
print(f"因子数：{len(feature_cols)}")
for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
    d = dataset.fetch_learn(seg)
    print(f"  {seg.name:<6} 样本 {d.height:>5}")
train_learn.select(['datetime'] + feature_cols[:8]).tail(5).to_pandas()
"""
)
code(
    """
sub = train_learn.select(feature_cols[:24]).to_pandas()
corr = sub.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title("因子相关性热力图（前24个）", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()
"""
)

# ============================================================ 多模型
md(
    "## 3️⃣ 多模型训练 🤖\n"
    "- **Lasso**：L1 正则线性基线\n"
    "- **GlassLgbModel**：强正则 LightGBM，TRAIN+VALID 合并拟合 + 时间切分早停（`model.py`）\n"
    "- **MLP**：多层感知机，捕捉非线性"
)
code(
    """
from vnpy.alpha.model.models.lasso_model import LassoModel
from vnpy.alpha.model.models.mlp_model import MlpModel

n_feat = len(feature_cols)
lgb_params = dict(config.LGB_PARAMS)
if QUICK:
    lgb_params.update(num_boost_round=200, early_stopping_rounds=40)
    mlp_kw = dict(input_size=n_feat, hidden_sizes=(64,), lr=1e-3, n_epochs=60,
                  batch_size=512, early_stop_rounds=20, eval_steps=10, device="cpu", seed=42)
else:
    lgb_params.update(num_boost_round=400, early_stopping_rounds=60)
    mlp_kw = dict(input_size=n_feat, hidden_sizes=(128, 32), lr=1e-3, n_epochs=200,
                  batch_size=512, early_stop_rounds=40, eval_steps=20, device="cpu", seed=42)

models = {
    "Lasso":    LassoModel(alpha=5e-4, max_iter=2000, random_state=42),
    "LightGBM": GlassLgbModel(**lgb_params),
    "MLP":      MlpModel(**mlp_kw),
}

for name, m in models.items():
    print(f"—— 训练 {name} ——")
    m.fit(dataset)
print("\\n三模型训练完成 ✅")
"""
)

# ============================================================ IC
md(
    "## 4️⃣ 预测有效性：IC / RankIC 📊\n"
    "LightGBM 额外展示 **因果滚动 z-score** 后信号（与 `02_train.py` 一致，无未来函数）。"
)
code(
    """
def calc_ic(pred, label):
    mask = ~np.isnan(pred) & ~np.isnan(label)
    if mask.sum() < 3:
        return np.nan, np.nan
    return (float(np.corrcoef(pred[mask], label[mask])[0, 1]),
            float(spearmanr(pred[mask], label[mask]).statistic))

records = []
preds_cache = {}
for name, m in models.items():
    for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
        infer = dataset.fetch_infer(seg).sort(["datetime", "vt_symbol"])
        pred = m.predict(dataset, seg)
        preds_cache[(name, seg.name)] = (infer, pred)
        ic, ric = calc_ic(pred, infer["label"].to_numpy())
        records.append(dict(模型=name, 信号="原始", 样本=seg.name, IC=ic, RankIC=ric))

# LightGBM z-score 信号
z_preds = build_processed_predictions(
    dataset, models["LightGBM"],
    window=config.SIGNAL_ZSCORE_WINDOW,
    min_periods=config.SIGNAL_ZSCORE_MIN_PERIODS,
)
preds_cache[("LightGBM_z", "TEST")] = (
    dataset.fetch_infer(Segment.TEST).sort(["datetime", "vt_symbol"]),
    z_preds[Segment.TEST],
)
for seg in (Segment.TRAIN, Segment.VALID, Segment.TEST):
    infer = dataset.fetch_infer(seg).sort(["datetime", "vt_symbol"])
    ic, ric = calc_ic(z_preds[seg], infer["label"].to_numpy())
    records.append(dict(模型="LightGBM", 信号="z-score", 样本=seg.name, IC=ic, RankIC=ric))

ic_df = pd.DataFrame(records)
display(ic_df.pivot_table(index=["模型", "信号"], columns="样本", values=["IC", "RankIC"]).round(4))
"""
)
code(
    """
test_ic = ic_df[(ic_df["样本"] == "TEST") & (ic_df["信号"] == "原始")]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(test_ic)); w = 0.35
ax.bar(x - w/2, test_ic["IC"], w, label="IC", color="#e67e22")
ax.bar(x + w/2, test_ic["RankIC"], w, label="RankIC", color="#16a085")
ax.set_xticks(x); ax.set_xticklabels(test_ic["模型"])
ax.axhline(0, color="k", lw=.8)
ax.set_title("样本外(TEST) 各模型原始预测 IC / RankIC", fontsize=12)
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()
"""
)

# ============================================================ 特征重要性
md("## 5️⃣ LightGBM 特征重要性 🌳")
code(
    """
lgb = models["LightGBM"].model
imp = pd.Series(lgb.feature_importance(importance_type="gain"), index=feature_cols)
imp = imp.sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(imp.index, imp.values, color="#2e86de")
ax.set_title("GlassLgbModel 因子重要性 Top20（gain）", fontsize=12)
ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.show()
"""
)

# ============================================================ 回测
md(
    "## 6️⃣ 策略回测：趋势 + AI 否决 🏆\n"
    "推荐策略 **`LMS_AI_MODE=\"veto\"`**：三均线定方向，AI z-score 仅在强烈反向（`|z|>LMS_VETO_THRESHOLD`）时否决。\n"
    "\n"
    "对比：**纯规则** · **趋势+AI否决** · **趋势+AI确认**（`confirm` 模式，更保守）。"
)
code(
    """
from datetime import datetime
import lms
from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

config.ensure_dirs()
lab = AlphaLab(str(config.LAB_PATH))
lab.save_bar_data(data_loader.build_bar_data())
lab.add_contract_setting(config.VT_SYMBOL, config.LONG_RATE, config.SHORT_RATE,
                         config.CONTRACT_SIZE, config.PRICE_TICK)

infer_lgb, _ = preds_cache[("LightGBM", "TEST")]
z_test = z_preds[Segment.TEST]
model_signal = infer_lgb.select(["datetime", "vt_symbol"]).with_columns(pl.Series(SIGNAL_COL, z_test))

def run_lms(mode: str):
    use_ai = mode != "off"
    sig = lms.build_signal_df(df_pl, model_signal if use_ai else None)
    eng = BacktestingEngine(lab)
    eng.set_parameters(
        vt_symbols=[config.VT_SYMBOL], interval=Interval.DAILY,
        start=datetime.strptime(config.TEST_PERIOD[0], "%Y-%m-%d"),
        end=datetime.strptime(config.TEST_PERIOD[1], "%Y-%m-%d"),
        capital=config.CAPITAL, annual_days=240,
    )
    eng.add_strategy(LmsAlphaStrategy, dict(
        use_ai=use_ai,
        ai_mode=mode,
        signal_threshold=config.SIGNAL_THRESHOLD,
        veto_threshold=config.LMS_VETO_THRESHOLD,
        position_pct=config.POSITION_PCT,
        price_add_ticks=config.PRICE_ADD_TICKS,
    ), sig)
    eng.load_data(); eng.run_backtesting()
    return eng.calculate_result(), eng.calculate_statistics()

lms_results = {
    "纯规则 LMS": run_lms("off"),
    f"趋势+AI否决(>{config.LMS_VETO_THRESHOLD})": run_lms("veto"),
    f"趋势+AI确认(>{config.SIGNAL_THRESHOLD})": run_lms("confirm"),
}
for name, (_, st) in lms_results.items():
    print(f"{name:<22} 总收益 {st.get('total_return',0):>6.2f}%  "
          f"年化 {st.get('annual_return',0):>5.2f}%  夏普 {st.get('sharpe_ratio',0):>5.2f}  "
          f"成交 {st.get('total_trade_count',0)} 笔")
"""
)
code(
    """
fig, ax = plt.subplots(figsize=(12, 6))
palette = ["#27ae60", "#c0392b", "#8e44ad"]
for (name, (daily, _)), c in zip(lms_results.items(), palette):
    if daily is None or daily.is_empty():
        continue
    d = daily.to_pandas()
    d["balance"] = config.CAPITAL + d["net_pnl"].cumsum()
    ax.plot(d["date"], d["balance"], label=name, color=c, lw=1.6)

bh = raw[(raw["datetime"] >= config.TEST_PERIOD[0]) & (raw["datetime"] <= config.TEST_PERIOD[1])].copy()
ax.plot(bh["datetime"], config.CAPITAL * bh["close"] / bh["close"].iloc[0],
        label="买入持有", color="#95a5a6", ls="--", lw=1.2)
ax.axhline(config.CAPITAL, color="k", lw=.6, alpha=.5)
ax.set_title("样本外(TEST) 策略资金曲线对比", fontsize=13)
ax.set_ylabel("账户净值"); ax.legend(loc="best", fontsize=9); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)
code(
    """
metric_keys = ["total_return", "annual_return", "max_drawdown", "sharpe_ratio",
               "return_drawdown_ratio", "total_trade_count", "daily_return"]
name_cn = {"total_return": "总收益%", "annual_return": "年化%", "max_drawdown": "最大回撤",
           "sharpe_ratio": "夏普", "return_drawdown_ratio": "收益回撤比",
           "total_trade_count": "成交笔数", "daily_return": "日均收益%"}
rows = {name: {name_cn[k]: round(st.get(k, float('nan')), 3) for k in metric_keys}
        for name, (_, st) in lms_results.items()}
display(pd.DataFrame(rows).T)
"""
)

# ============================================================ LMS 可视化
md(
    "## 7️⃣ LMS 三均线趋势可视化 🧭\n"
    "> LMS = **L/M/S 长/中/短三均线**。多头排列只做多，空头排列只做空，缠绕时空仓。"
)
code(
    """
reg = lms.compute_lms_regime(df_pl).to_pandas()
mask = (reg["datetime"] >= pd.Timestamp(config.TEST_PERIOD[0])) & (reg["datetime"] <= pd.Timestamp(config.TEST_PERIOD[1]))
r = reg[mask].merge(raw[["datetime", "close"]], on="datetime")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(r["datetime"], r["close"], color="#2c3e50", lw=1.2, label="收盘价")
ax.plot(r["datetime"], r["ma_s"], color="#e74c3c", lw=.9, label=f"MA{config.LMS_SHORT}(短)")
ax.plot(r["datetime"], r["ma_m"], color="#f39c12", lw=.9, label=f"MA{config.LMS_MEDIUM}(中)")
ax.plot(r["datetime"], r["ma_l"], color="#2980b9", lw=.9, label=f"MA{config.LMS_LONG}(长)")
lo, hi = r["close"].min(), r["close"].max()
ax.fill_between(r["datetime"], lo, hi, where=r["lms_regime"] > 0, color="#2ecc71", alpha=.12, label="多头区")
ax.fill_between(r["datetime"], lo, hi, where=r["lms_regime"] < 0, color="#e74c3c", alpha=.12, label="空头区")
ax.set_title("LMS 三均线趋势 · 多空区间（TEST 段）", fontsize=13)
ax.legend(loc="upper left", ncol=3, fontsize=9); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

# ============================================================ 小结
md(
    "## 8️⃣ 小结与后续 📝\n"
    "- **模型**：`GlassLgbModel` 强正则 + TRAIN/VALID 合并训练，配合 **因果 z-score** 后处理，缓解过拟合与偏多偏差。\n"
    "- **策略**：单标的日线上 **纯 AI 多空翻转** 样本外易亏；推荐 **趋势 + AI 否决**（`LMS_AI_MODE=\"veto\"`）。\n"
    "- **调参**：`config.LMS_VETO_THRESHOLD`（1.0~1.5）、`LGB_PARAMS`、`LABEL_HOLD`。\n"
    "- **流水线**：`python 02_train.py` → `python 03_backtest.py` 与 Notebook 逻辑一致。\n"
    "\n"
    "> ⚠️ 本 Notebook 为量化研究示例，历史回测不代表未来；实盘需考虑真实成本、换月、涨跌停不可成交等因素。"
)

# ============================================================ 写盘
nb["cells"] = cells
nb["metadata"] = {
    "language_info": {"name": "python"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
}

out = config.ROOT / "fg_ai" / "玻璃期货AI多模型.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("已生成：", out)
print("单元格数：", len(cells))
