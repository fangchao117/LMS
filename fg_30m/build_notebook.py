"""
生成「玻璃30分钟短线.ipynb」

    python build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import config

nb = new_notebook()
cells: list = []


def md(t: str) -> None:
    cells.append(new_markdown_cell(t))


def code(t: str) -> None:
    cells.append(new_code_cell(t.strip("\n")))


md(
    "# 玻璃 30 分钟：日线定方向 + 30m 执行\n"
    "\n"
    "> **组合策略**（regime）：日线 EMA 定多空，30 分钟 EMA 开平仓\n"
    "\n"
    f"- 仿真合约：`{config.VT_SYMBOL}`\n"
    f"- 日线参考：`{config.DAILY_SYMBOL}.CZCE`（FG00 连续）\n"
    f"- 30m K 线：vnpy 1 分钟合成\n"
    f"- 信号：`{config.SIGNAL_MODE}`，EMA **{config.EMA_FAST}/{config.EMA_SLOW}**\n"
    f"- 资金：`{config.CAPITAL:,}` 元，最多 `{config.MAX_LOTS}` 手\n"
    "\n"
    "| 层级 | 周期 | 作用 |\n"
    "|------|------|------|\n"
    "| 方向 | 日线 EMA | 只允许多/空/空仓 |\n"
    "| 执行 | 30m EMA | 开平仓时机 |"
)

md("## 0️⃣ 环境")
code(
    """
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

import config, data, signals, regime
from backtest import run
print("就绪", config.VT_SYMBOL, config.SIGNAL_MODE)
"""
)

md("## 1️⃣ 加载 30 分钟 + 日线")
code(
    f"""
SYMBOL = "{config.SYMBOL}"
df = data.load_30m(SYMBOL)
df_daily = regime.load_daily(str(df["datetime"].min().date()), str(df["datetime"].max().date()))
print(f"30m {{SYMBOL}}: {{len(df)}} 根  {{df['datetime'].min()}} -> {{df['datetime'].max()}}")
print(f"日线 FG00: {{len(df_daily)}} 根")
df.tail(3)
"""
)

md("## 2️⃣ 日线趋势方向")
code(
    f"""
trend_tbl = regime.daily_trend_table(df_daily, {config.EMA_FAST}, {config.EMA_SLOW})
daily_trend = regime.align_daily_trend(df, trend_tbl)
print("日线方向分布:", daily_trend.value_counts().to_dict())

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
axes[0].plot(df_daily["datetime"], df_daily["close"], lw=1)
axes[0].set_title("FG00 日线收盘")
d = df_daily.copy()
d["e1"] = d["close"].ewm(span={config.EMA_FAST}, adjust=False).mean()
d["e2"] = d["close"].ewm(span={config.EMA_SLOW}, adjust=False).mean()
axes[0].plot(d["datetime"], d["e1"], label=f"EMA{config.EMA_FAST}")
axes[0].plot(d["datetime"], d["e2"], label=f"EMA{config.EMA_SLOW}")
axes[0].legend()
sample = df.tail(800).copy()
sample["dt"] = daily_trend.tail(800)
axes[1].plot(sample["datetime"], sample["close"], lw=1)
axes[1].set_title("30m 收盘（末800根）")
plt.tight_layout()
plt.show()
"""
)

md("## 3️⃣ 组合信号 vs 纯 30m EMA")
code(
    f"""
sig_combo = signals.generate(df, "{config.SIGNAL_MODE}", ema_fast={config.EMA_FAST}, ema_slow={config.EMA_SLOW})
sig_pure = signals.generate(df, "ema_cross", ema_fast={config.EMA_FAST}, ema_slow={config.EMA_SLOW})
print("组合信号:", sig_combo.value_counts().to_dict())
print("纯30m EMA:", sig_pure.value_counts().to_dict())
"""
)

md("## 4️⃣ 全样本回测对比")
code(
    """
stats_combo, eq_combo = run(df, sig_combo, max_lots=1)
stats_pure, eq_pure = run(df, sig_pure, max_lots=1)
print(f"组合  收益 {stats_combo['total_return_pct']:.1f}%  回撤 {stats_combo['max_ddpercent']:.1f}%  成交 {stats_combo['total_trades']}")
print(f"纯EMA 收益 {stats_pure['total_return_pct']:.1f}%  回撤 {stats_pure['max_ddpercent']:.1f}%  成交 {stats_pure['total_trades']}")
"""
)

md("## 5️⃣ TEST 段样本外")
code(
    f"""
t0, t1 = config.TEST_PERIOD
df_test = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].reset_index(drop=True)
sig_c = signals.generate(df_test, config.SIGNAL_MODE, ema_fast={config.EMA_FAST}, ema_slow={config.EMA_SLOW})
sig_p = signals.generate(df_test, "ema_cross", ema_fast={config.EMA_FAST}, ema_slow={config.EMA_SLOW})
st_c, eq_c = run(df_test, sig_c, max_lots=1)
st_p, eq_p = run(df_test, sig_p, max_lots=1)
print(f"TEST {{t0}} ~ {{t1}}")
print(f"  组合  收益 {{st_c['total_return_pct']:.1f}}%  回撤 {{st_c['max_ddpercent']:.1f}}%  成交 {{st_c['total_trades']}}")
print(f"  纯EMA 收益 {{st_p['total_return_pct']:.1f}}%  回撤 {{st_p['max_ddpercent']:.1f}}%  成交 {{st_p['total_trades']}}")
"""
)

md("## 6️⃣ 权益曲线")
code(
    """
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(eq_c["datetime"], eq_c["equity"], lw=1.2, label="组合策略")
ax.plot(eq_p["datetime"], eq_p["equity"], lw=1.0, ls="--", label="纯30m EMA", alpha=0.8)
ax.axhline(config.CAPITAL, color="gray", ls=":", lw=0.8)
ax.set_title("TEST 段权益曲线")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    "## 7️⃣ VeighNa 仿真\n"
    "\n"
    "```powershell\n"
    "cd D:\\LMS\\vnpy_sim\n"
    "python install_30m_strategy.py   # 复制策略，重启 VeighNa\n"
    "```\n"
    "\n"
    "- 策略类：`Fg30mRegimeStrategy`\n"
    "- 合约：`FG609.CZCE`（具体月份）\n"
    "- **先连 CTP → 再初始化 → 再启动**"
)

md("## 8️⃣ 数据维护")
code(
    """
# 有新 1 分钟数据时重建 30m 缓存
# !python refresh_cache.py --symbol FG609

# 从信易 EDB 补 1 分钟数据
# !python ../data/download_edb.py --symbol CZCE.FG609 --period 1m --update-fg30m
"""
)

nb["cells"] = cells
out = Path(__file__).resolve().parent / "玻璃30分钟短线.ipynb"
nbf.write(nb, out)
print(f"已生成 -> {out}")
