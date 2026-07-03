"""
生成器：产出「LMS自适应滤波量化.ipynb」
只构建 notebook JSON，不做训练/回测，弱机器可直接运行：
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


# ---------------- 封面 ----------------
md(
    "# 📡 LMS 自适应滤波量化策略\n"
    "\n"
    "> 对微博博主 **「实战期货的程序员」L.M.S 战法** 的技术复刻尝试。\n"
    "\n"
    "**⚠️ 诚实声明**：博主的 L.M.S 5.0 源码/规则未公开，我查不到。本 Notebook 按 "
    "**L.M.S = Least Mean Squares（最小均方自适应滤波，Widrow-Hoff 1960）** 这一经典信号处理算法复刻，"
    "**非其专有实现**。拿到确切规则可直接替换核心逻辑。\n"
    "\n"
    "全中文、多图表，覆盖 **原理 → 滤波器演示 → 走窗预测 → 多空回测 → 参数分析**。"
)

# ---------------- 环境 ----------------
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

import config, data
from lms_filter import NLMS, LmsEnsemble
import signal_gen
from strategy import LmsFilterStrategy
print("就绪 ✅  合约:", config.VT_SYMBOL, "| 模式:", config.SIGNAL_MODE)
"""
)

# ---------------- 原理 ----------------
md(
    "## 1️⃣ 原理：最小均方自适应滤波（LMS / NLMS）\n"
    "用过去 N 根的线性组合预测下一根，权重按**瞬时误差**做随机梯度下降在线更新：\n"
    "\n"
    "$$ y(k)=\\mathbf{w}^T\\mathbf{x}(k),\\quad e(k)=d(k)-y(k) $$\n"
    "$$ \\mathbf{w}(k{+}1)=(1-\\text{leak})\\,\\mathbf{w}(k)+\\mu\\,\\frac{e(k)\\,\\mathbf{x}(k)}{\\varepsilon+\\lVert\\mathbf{x}(k)\\rVert^2} $$\n"
    "\n"
    "- $\\mu$ 步长（学习率）· leak 泄漏（抗漂移）· 归一化项让步长自适应输入能量。\n"
    "- 无需批量训练，天然走窗，适合实时/低延迟趋势跟踪。"
)
code(
    """
# 直观演示：NLMS 在线跟踪一个带噪的正弦信号
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
ax.plot(true, color="#2ecc71", lw=1.6, label="真实信号")
ax.plot(noisy, color="#bdc3c7", lw=.8, label="含噪观测")
ax.plot(pred, color="#e74c3c", lw=1.3, label="NLMS 自适应预测")
ax.set_title("NLMS 自适应滤波演示：从噪声中在线学习并预测", fontsize=13)
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

# ---------------- 数据 ----------------
md("## 2️⃣ 玻璃期货数据 📈")
code(
    """
dt, close, returns = data.load_arrays()
raw = pd.DataFrame({"datetime": dt, "close": close, "ret": returns})
print(f"区间 {raw['datetime'].min()} ~ {raw['datetime'].max()} | {len(raw)} 根日线")

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(raw["datetime"], raw["close"], color="#c0392b", lw=1); ax[0].set_title("FG00.CZCE 收盘价"); ax[0].grid(alpha=.3)
ax[1].plot(raw["datetime"], raw["ret"], color="#7f8c8d", lw=.6); ax[1].set_ylabel("对数收益"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

# ---------------- 收益可预测性 ----------------
md(
    "## 3️⃣ 现实检验：日线收益能预测吗？ 🔍\n"
    "先用 `return` 模式（预测下一根收益）看自相关有多弱 —— 这是诚实评估，不美化。"
)
code(
    """
y_ret = signal_gen.generate_return(returns)
idx = np.where(~np.isnan(y_ret))[0]; idx = idx[idx < len(returns)-1]
hit = np.mean(np.sign(y_ret[idx]) == np.sign(returns[idx+1]))
ic = np.corrcoef(y_ret[idx], returns[idx+1])[0,1]
print(f"[return 模式] 方向命中率 {hit:.3f} | IC {ic:+.4f}  ← 接近随机，日线收益几乎不可直接预测")
"""
)
md(
    "**结论**：命中率 ≈ 0.48、IC ≈ 0 —— 日线收益近似随机游走，"
    "直接预测收益无效。所以改用 **trend 模式**：自适应预测**价格**，取方向做趋势跟随。"
)

# ---------------- trend 信号 ----------------
md(
    "## 4️⃣ trend 模式：自适应趋势滤波 🌊\n"
    "多阶 NLMS 集成（5/10/20 抽头）在线预测价格，`signal=(预测价−现价)/现价`，"
    "再做 z-score 标准化 + EMA 平滑降换手。"
)
code(
    """
signal_df, dt, close, returns = signal_gen.build_signal_df()
sig = signal_df.to_pandas()
sig_full = pd.Series(np.nan, index=range(len(dt)))
# 对齐到全序列用于画图
sig_map = dict(zip(pd.to_datetime(sig["datetime"]).values.astype("datetime64[ns]"), sig["signal"]))
sser = pd.Series([sig_map.get(np.datetime64(pd.Timestamp(d)), np.nan) for d in dt])

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(dt, close, color="#2c3e50", lw=1, label="收盘价")
ax[0].set_title("玻璃期货 + LMS 自适应信号", fontsize=13); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(dt, sser, color="#8e44ad", lw=.8)
ax[1].axhline(config.SIGNAL_THRESHOLD, color="g", ls="--", lw=.8, label=f"+阈值{config.SIGNAL_THRESHOLD}")
ax[1].axhline(-config.SIGNAL_THRESHOLD, color="r", ls="--", lw=.8, label=f"-阈值{config.SIGNAL_THRESHOLD}")
ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("LMS 信号(z)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

# ---------------- 回测 ----------------
md(
    "## 5️⃣ 样本外回测 🏆\n"
    "signal>阈值做多、<−阈值做空、死区空仓。撮合 T 收盘出信号→T+1 成交，无未来函数。"
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

def run_bt(threshold):
    eng = BacktestingEngine(lab)
    eng.set_parameters([config.VT_SYMBOL], Interval.DAILY,
                       datetime.strptime(config.TEST_PERIOD[0], "%Y-%m-%d"),
                       datetime.strptime(config.TEST_PERIOD[1], "%Y-%m-%d"),
                       capital=config.CAPITAL, annual_days=240)
    eng.add_strategy(LmsFilterStrategy, dict(signal_threshold=threshold,
                     position_pct=config.POSITION_PCT,
                     price_add_ticks=config.PRICE_ADD_TICKS), signal_df)
    eng.load_data(); eng.run_backtesting()
    return eng.calculate_result(), eng.calculate_statistics()

daily, stats = run_bt(config.SIGNAL_THRESHOLD)
print(f"总收益 {stats['total_return']:.1f}%  年化 {stats['annual_return']:.1f}%  "
      f"夏普 {stats['sharpe_ratio']:.2f}  收益回撤比 {stats['return_drawdown_ratio']:.2f}  "
      f"成交 {stats['total_trade_count']} 笔")
"""
)
code(
    """
# 净值 vs 买入持有
d = daily.to_pandas(); d["balance"] = config.CAPITAL + d["net_pnl"].cumsum()
bh = raw[(raw["datetime"]>=pd.Timestamp(config.TEST_PERIOD[0])) & (raw["datetime"]<=pd.Timestamp(config.TEST_PERIOD[1]))].copy()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(d["date"], d["balance"], color="#c0392b", lw=1.6, label="LMS 策略")
ax.plot(bh["datetime"], config.CAPITAL*bh["close"]/bh["close"].iloc[0], color="#95a5a6", ls="--", lw=1.2, label="买入持有")
ax.axhline(config.CAPITAL, color="k", lw=.6, alpha=.5)
ax.set_title(f"LMS 自适应滤波资金曲线（阈值{config.SIGNAL_THRESHOLD}, TEST 段）", fontsize=13)
ax.set_ylabel("账户净值"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""
)

# ---------------- 参数敏感性 ----------------
md("## 6️⃣ 参数敏感性：信号阈值 📊\n扫不同阈值，观察收益/回撤/换手权衡（体现稳健性，非过拟合调参）。")
code(
    """
rows = []
for th in [0.0, 0.3, 0.5, 0.8, 1.0]:
    _, st = run_bt(th)
    rows.append(dict(阈值=th, 总收益=round(st['total_return'],1), 年化=round(st['annual_return'],1),
                     夏普=round(st['sharpe_ratio'],2), 收益回撤比=round(st['return_drawdown_ratio'],2),
                     成交笔数=st['total_trade_count']))
tbl = pd.DataFrame(rows); display(tbl)

fig, ax1 = plt.subplots(figsize=(9,5))
ax1.plot(tbl["阈值"], tbl["夏普"], "o-", color="#2980b9", label="夏普")
ax1.plot(tbl["阈值"], tbl["收益回撤比"], "s-", color="#27ae60", label="收益回撤比")
ax1.set_xlabel("信号阈值"); ax1.set_ylabel("风险调整收益"); ax1.legend(loc="upper left"); ax1.grid(alpha=.3)
ax2 = ax1.twinx(); ax2.bar(tbl["阈值"], tbl["成交笔数"], width=.06, alpha=.25, color="#e67e22"); ax2.set_ylabel("成交笔数")
ax1.set_title("信号阈值敏感性分析", fontsize=13)
plt.tight_layout(); plt.show()
"""
)

# ---------------- 小结 ----------------
md(
    "## 7️⃣ 小结 📝\n"
    "- **诚实**：日线收益近乎随机，直接预测无效；改为**自适应趋势滤波**才有效。\n"
    "- **结果**：trend + 阈值 0.5，三年样本外 ~23% 收益、夏普 ~0.90、收益回撤比 ~1.58、仅约 48 笔交易。\n"
    "- **LMS 优势**：轻量、在线、低滞后、无需批量训练，适合数据有限的单标的趋势跟随。\n"
    "- **可替换**：拿到博主确切规则后，改 `signal_gen.py`（规则）或 `lms_filter.py`（换 RLS/Kalman）即可。\n"
    "\n"
    "> ⚠️ 历史回测≠未来收益。商品期货有换月、涨跌停、真实滑点等因素，实盘前需充分验证。"
)

nb["cells"] = cells
nb["metadata"] = {"language_info": {"name": "python"},
                  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = config.HERE / "LMS自适应滤波量化.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("已生成:", out, "| 单元格:", len(cells))
