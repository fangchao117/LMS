"""
生成器：产出「LMS自适应滤波量化.ipynb」
    python build_notebook.py
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import config

nb = new_notebook()
cells: list = []


def md(t: str) -> None:
    cells.append(new_markdown_cell(t))


def code(t: str) -> None:
    cells.append(new_code_cell(t.strip("\n")))


weights_str = ", ".join(f"{k}={v}" for k, v in config.FACTOR_WEIGHTS.items() if v > 0)

md(
    "# 📡 LMS 自适应滤波 + 多因子融合量化\n"
    "\n"
    "> **LMS** = Least Mean Squares 最小均方自适应滤波，叠加动量 / 三均线 / 量价 / 持仓 / RSI 等因子加权融合。\n"
    "\n"
    f"- 信号模式：`{config.SIGNAL_MODE}`\n"
    f"- 因子权重：{weights_str}\n"
    f"- 仓位：`POSITION_PCT={config.POSITION_PCT}`（按净值动态满仓，无手数上限）\n"
    f"- 阈值：`SIGNAL_THRESHOLD={config.SIGNAL_THRESHOLD}`（VALID 段可自动扫描最优）\n"
    "\n"
    "全中文、多图表：**原理 → 因子 → 融合信号 → 回测 → 参数分析**。"
)

md("## 0️⃣ 环境与配置")
code(
    """
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")

FG_DIR = Path.cwd()
if str(FG_DIR) not in sys.path:
    sys.path.insert(0, str(FG_DIR))

import numpy as np, pandas as pd, polars as pl
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

import config, data, factors, signal_gen
from lms_filter import NLMS, LmsEnsemble
from strategy import LmsFilterStrategy
print("就绪 ✅", config.VT_SYMBOL, "| 模式:", config.SIGNAL_MODE)
print("因子权重:", {k:v for k,v in config.FACTOR_WEIGHTS.items() if v>0})
"""
)

md(
    "## 1️⃣ 原理：NLMS 自适应滤波\n"
    "$$ y(k)=\\mathbf{w}^T\\mathbf{x}(k),\\quad "
    "\\mathbf{w}(k{+}1)=(1-\\text{leak})\\,\\mathbf{w}(k)+\\mu\\,"
    "\\frac{e(k)\\,\\mathbf{x}(k)}{\\varepsilon+\\lVert\\mathbf{x}(k)\\rVert^2} $$\n"
    "在线走窗更新，无需批量训练；与多因子 z-score 加权融合后做多空择时。"
)
code(
    """
rng = np.random.default_rng(42)
n = 300
true = np.sin(np.linspace(0, 6*np.pi, n))
noisy = true + rng.normal(0, 0.3, n)
f = NLMS(order=8, mu=0.3, leak=1e-4)
pred = np.full(n, np.nan)
for t in range(8, n-1):
    x = noisy[t-8:t][::-1]
    pred[t] = f.predict(x)
    f.adapt(x, noisy[t+1])
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(true, color="#2ecc71", lw=1.6, label="真实")
ax.plot(noisy, color="#bdc3c7", lw=.8, label="含噪")
ax.plot(pred, color="#e74c3c", lw=1.3, label="NLMS 预测")
ax.set_title("NLMS 自适应滤波演示"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

md("## 2️⃣ 玻璃期货数据")
code(
    """
pdf = data.load_dataframe()
dt, close, returns = data.load_arrays()
print(f"区间 {pdf['datetime'].min()} ~ {pdf['datetime'].max()} | {len(pdf)} 根")
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(pdf["datetime"], pdf["close"], color="#c0392b", lw=1); ax[0].set_title("FG00 收盘价"); ax[0].grid(alpha=.3)
ax[1].plot(pdf["datetime"], returns, color="#7f8c8d", lw=.6); ax[1].set_ylabel("收益"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

md(
    "## 3️⃣ 多因子分解 🔬\n"
    "各因子独立计算后因果 z-score，再按 `config.FACTOR_WEIGHTS` 加权融合。"
)
code(
    """
lms_raw = signal_gen.generate_trend(close)
raw = factors.compute_raw_factors(pdf, lms_raw)
composite, factor_z = factors.fuse_factors(raw)

fig, axes = plt.subplots(len(factor_z)+1, 1, figsize=(12, 2*(len(factor_z)+1)), sharex=True)
for ax, (name, z) in zip(axes[:-1], factor_z.items()):
    ax.plot(pdf["datetime"], z, lw=.7, color="#8e44ad")
    ax.set_ylabel(name, fontsize=9); ax.grid(alpha=.3)
axes[-1].plot(pdf["datetime"], composite, lw=.8, color="#c0392b")
axes[-1].set_ylabel("合成"); axes[-1].set_title("多因子 z-score 与合成信号")
axes[-1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

md("## 4️⃣ 融合信号与阈值")
code(
    """
signal_df, dt, close, returns = signal_gen.build_signal_df()
sig = signal_df.to_pandas()
print(signal_df.tail(3))

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(dt, close, color="#2c3e50", lw=1); ax[0].set_title("收盘价 + 融合信号"); ax[0].grid(alpha=.3)
ax[1].plot(sig["datetime"], sig["signal"], color="#8e44ad", lw=.8)
ax[1].axhline(config.SIGNAL_THRESHOLD, color="g", ls="--", lw=.8, label=f"+阈值")
ax[1].axhline(-config.SIGNAL_THRESHOLD, color="r", ls="--", lw=.8)
ax[1].axhline(0, color="k", lw=.5); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

md(
    "## 5️⃣ 样本外回测 🏆\n"
    "signal>阈值做多、<−阈值做空；手数 = floor(净值×仓位比例/名义价值)，随净值动态缩放。"
)
code(
    """
from datetime import datetime
from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

config.ensure_dirs()
lab = AlphaLab(str(config.LAB_PATH))
lab.save_bar_data(data.build_bars())
lab.add_contract_setting(config.VT_SYMBOL, config.LONG_RATE, config.SHORT_RATE,
                         config.CONTRACT_SIZE, config.PRICE_TICK)

def run_bt(threshold, start, end):
    eng = BacktestingEngine(lab)
    eng.set_parameters(
        vt_symbols=[config.VT_SYMBOL], interval=Interval.DAILY,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
        capital=config.CAPITAL, annual_days=240,
    )
    eng.add_strategy(LmsFilterStrategy, dict(
        signal_threshold=threshold,
        position_pct=config.POSITION_PCT,
        price_add_ticks=config.PRICE_ADD_TICKS,
    ), signal_df)
    eng.load_data(); eng.run_backtesting()
    return eng.calculate_result(), eng.calculate_statistics()

# VALID 扫描最优阈值
best_th, best_ret = config.SIGNAL_THRESHOLD, -1e9
rows = []
for th in config.THRESHOLD_CANDIDATES:
    _, st = run_bt(th, config.VALID_PERIOD[0], config.VALID_PERIOD[1])
    rows.append(dict(阈值=th, VALID总收益=round(st["total_return"],1), 夏普=round(st["sharpe_ratio"],2)))
    if st["total_return"] > best_ret:
        best_ret, best_th = st["total_return"], th
print("VALID 阈值扫描:"); display(pd.DataFrame(rows))
print(f"最优阈值: {best_th}")

daily, stats = run_bt(best_th, config.TEST_PERIOD[0], config.TEST_PERIOD[1])
print(f"TEST 总收益 {stats['total_return']:.1f}%  年化 {stats['annual_return']:.1f}%  "
      f"夏普 {stats['sharpe_ratio']:.2f}  成交 {stats['total_trade_count']} 笔")
"""
)
code(
    """
d = daily.to_pandas(); d["balance"] = config.CAPITAL + d["net_pnl"].cumsum()
bh = pdf[(pdf["datetime"]>=pd.Timestamp(config.TEST_PERIOD[0])) & (pdf["datetime"]<=pd.Timestamp(config.TEST_PERIOD[1]))].copy()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(d["date"], d["balance"], color="#c0392b", lw=1.6, label="LMS+多因子")
ax.plot(bh["datetime"], config.CAPITAL*bh["close"]/bh["close"].iloc[0], color="#95a5a6", ls="--", lw=1.2, label="买入持有")
ax.axhline(config.CAPITAL, color="k", lw=.6, alpha=.5)
ax.set_title(f"资金曲线（阈值={best_th}, TEST 段）"); ax.set_ylabel("净值"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

md("## 6️⃣ 因子权重一览")
code(
    """
w = {k:v for k,v in config.FACTOR_WEIGHTS.items() if v>0}
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(list(w.keys()), list(w.values()), color="#3498db")
ax.set_xlabel("权重"); ax.set_title("多因子融合权重"); ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.show()
"""
)

md(
    "## 7️⃣ 小结\n"
    "- **LMS** 提供低滞后自适应趋势；**多因子** 提供动量/均线/量价/持仓确认。\n"
    "- 仓位按净值比例动态计算，**无固定手数上限**。\n"
    "- VALID 段扫描阈值以**收益最大化**；TEST 段为严格样本外。\n"
    "\n"
    "> ⚠️ 历史回测≠未来收益。实盘需考虑换月、涨跌停、冲击成本。"
)

nb["cells"] = cells
nb["metadata"] = {
    "language_info": {"name": "python"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
}
out = config.HERE / "LMS自适应滤波量化.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("已生成:", out, "| 单元格:", len(cells))
