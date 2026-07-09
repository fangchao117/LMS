"""Analyze open-interest / volume vs price on FG00 daily history."""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

df = pd.read_parquet("data/FG00.CZCE.parquet")
df = df.sort_values("datetime").reset_index(drop=True)
df["ret"] = df["close"].pct_change()
df["ret_open"] = df["open"] / df["close"].shift(1) - 1
df["ret_1d_fwd"] = df["close"].shift(-1) / df["close"] - 1
df["ret_3d_fwd"] = df["close"].shift(-3) / df["close"] - 1
df["ret_5d_fwd"] = df["close"].shift(-5) / df["close"] - 1
df["price_chg"] = df["close"].diff()
df["oi_chg"] = df["open_interest"].diff()
df["oi_pct"] = df["open_interest"].pct_change()
df["vol_pct"] = df["volume"].pct_change()
df["vol_ma20"] = df["volume"].rolling(20).mean()
df["vol_ratio"] = df["volume"] / df["vol_ma20"]

print("=" * 60)
print("1. 同期相关系数 (全样本 2013-2026, n=%d)" % len(df))
print("=" * 60)
sub = df.dropna(subset=["ret", "oi_pct", "vol_pct", "vol_ratio"])
for x in ["oi_pct", "oi_chg", "vol_pct", "vol_ratio", "volume"]:
    c = sub["ret"].corr(sub[x])
    print(f"  ret vs {x:12s}: {c:+.4f}")

print("\n" + "=" * 60)
print("2. 四象限：价格涨跌 x 持仓量涨跌 -> 未来收益")
print("=" * 60)
valid = df.dropna(subset=["price_chg", "oi_chg", "ret_1d_fwd"]).copy()
valid["regime"] = np.select(
    [
        (valid["price_chg"] > 0) & (valid["oi_chg"] > 0),
        (valid["price_chg"] > 0) & (valid["oi_chg"] <= 0),
        (valid["price_chg"] <= 0) & (valid["oi_chg"] > 0),
        (valid["price_chg"] <= 0) & (valid["oi_chg"] <= 0),
    ],
    ["价涨+增仓", "价涨+减仓", "价跌+增仓", "价跌+减仓"],
    default="NA",
)
for r in ["价涨+增仓", "价涨+减仓", "价跌+增仓", "价跌+减仓"]:
    s = valid[valid["regime"] == r]
    wr = (s["ret_1d_fwd"] > 0).mean() * 100
    print(
        f"  {r}: n={len(s):4d} | 当日 {s['ret'].mean()*100:+.3f}%"
        f" | 次日 {s['ret_1d_fwd'].mean()*100:+.3f}%"
        f" | 3日 {s['ret_3d_fwd'].mean()*100:+.3f}%"
        f" | 5日 {s['ret_5d_fwd'].mean()*100:+.3f}%"
        f" | 次日胜率 {wr:.1f}%"
    )

print("\n" + "=" * 60)
print("3. 成交量：放量/缩量 x 价格方向 -> 次日收益")
print("=" * 60)
valid2 = df.dropna(subset=["ret", "vol_ratio", "ret_1d_fwd"]).copy()
valid2["vol_regime"] = np.where(
    valid2["vol_ratio"] > 1.2,
    "放量(>1.2x)",
    np.where(valid2["vol_ratio"] < 0.8, "缩量(<0.8x)", "正常"),
)
valid2["dir"] = np.where(valid2["ret"] > 0, "涨", "跌")
for vr in ["放量(>1.2x)", "正常", "缩量(<0.8x)"]:
    for d in ["涨", "跌"]:
        s = valid2[(valid2["vol_regime"] == vr) & (valid2["dir"] == d)]
        if len(s) < 20:
            continue
        wr = (s["ret_1d_fwd"] > 0).mean() * 100
        print(f"  {vr}+{d}: n={len(s):4d} | 次日 {s['ret_1d_fwd'].mean()*100:+.3f}% | 胜率 {wr:.1f}%")

print("\n" + "=" * 60)
print("4. 持仓量日变化分位 -> 未来5日收益")
print("=" * 60)
valid3 = df.dropna(subset=["oi_pct", "ret_5d_fwd"]).copy()
valid3["oi_q"] = pd.qcut(valid3["oi_pct"], 5, labels=["Q1大减", "Q2", "Q3", "Q4", "Q5大增"])
for q in ["Q1大减", "Q2", "Q3", "Q4", "Q5大增"]:
    s = valid3[valid3["oi_q"] == q]
    wr = (s["ret_5d_fwd"] > 0).mean() * 100
    print(f"  {q}: n={len(s):4d} | 5日 {s['ret_5d_fwd'].mean()*100:+.3f}% | 胜率 {wr:.1f}%")

print("\n" + "=" * 60)
print("5. 近2年(2024-06~) 四象限")
print("=" * 60)
recent = valid[valid["datetime"] >= pd.Timestamp("2024-06-01")]
for r in ["价涨+增仓", "价涨+减仓", "价跌+增仓", "价跌+减仓"]:
    s = recent[recent["regime"] == r]
    wr = (s["ret_1d_fwd"] > 0).mean() * 100
    print(
        f"  {r}: n={len(s):4d} | 次日 {s['ret_1d_fwd'].mean()*100:+.3f}%"
        f" | 5日 {s['ret_5d_fwd'].mean()*100:+.3f}% | 次日胜率 {wr:.1f}%"
    )

print("\n" + "=" * 60)
print("6. 量价配合 vs 量价背离 -> 未来收益")
print("=" * 60)
valid4 = df.dropna(subset=["ret", "vol_pct", "ret_1d_fwd"]).copy()
valid4["vp"] = np.select(
    [
        (valid4["ret"] > 0) & (valid4["vol_pct"] > 0),
        (valid4["ret"] > 0) & (valid4["vol_pct"] <= 0),
        (valid4["ret"] <= 0) & (valid4["vol_pct"] > 0),
        (valid4["ret"] <= 0) & (valid4["vol_pct"] <= 0),
    ],
    ["价涨量增", "价涨量缩", "价跌量增", "价跌量缩"],
    default="NA",
)
for v in ["价涨量增", "价涨量缩", "价跌量增", "价跌量缩"]:
    s = valid4[valid4["vp"] == v]
    wr = (s["ret_1d_fwd"] > 0).mean() * 100
    print(
        f"  {v}: n={len(s):4d} | 次日 {s['ret_1d_fwd'].mean()*100:+.3f}%"
        f" | 5日 {s['ret_5d_fwd'].mean()*100:+.3f}% | 胜率 {wr:.1f}%"
    )

print("\n" + "=" * 60)
print("7. 20日趋势段：价格 vs 持仓量")
print("=" * 60)
df["oi_20"] = df["open_interest"] / df["open_interest"].shift(20) - 1
df["ret_20"] = df["close"] / df["close"].shift(20) - 1
valid5 = df.dropna(subset=["oi_20", "ret_20", "ret_5d_fwd"]).copy()
valid5["trend"] = np.select(
    [
        (valid5["ret_20"] > 0) & (valid5["oi_20"] > 0),
        (valid5["ret_20"] > 0) & (valid5["oi_20"] <= 0),
        (valid5["ret_20"] <= 0) & (valid5["oi_20"] > 0),
        (valid5["ret_20"] <= 0) & (valid5["oi_20"] <= 0),
    ],
    ["20d价涨+持仓增", "20d价涨+持仓减", "20d价跌+持仓增", "20d价跌+持仓减"],
    default="NA",
)
for t in ["20d价涨+持仓增", "20d价涨+持仓减", "20d价跌+持仓增", "20d价跌+持仓减"]:
    s = valid5[valid5["trend"] == t]
    wr = (s["ret_5d_fwd"] > 0).mean() * 100
    print(f"  {t}: n={len(s):4d} | 未来5日 {s['ret_5d_fwd'].mean()*100:+.3f}% | 胜率 {wr:.1f}%")

print("\n" + "=" * 60)
print("8. 四象限下次日跳空(gap)分布")
print("=" * 60)
valid6 = valid.dropna(subset=["ret_open"])
for r in ["价涨+增仓", "价涨+减仓", "价跌+增仓", "价跌+减仓"]:
    s = valid6[valid6["regime"] == r]
    g = s["ret_open"] * 100
    print(
        f"  {r}: gap均值 {g.mean():+.3f}%"
        f" | std {g.std():.3f}% | P5 {g.quantile(0.05):+.2f}% | P95 {g.quantile(0.95):+.2f}%"
    )

print("\n" + "=" * 60)
print("9. 持仓量水平(20日Z) vs 未来5日收益")
print("=" * 60)
df["oi_z20"] = (df["open_interest"] - df["open_interest"].rolling(20).mean()) / (
    df["open_interest"].rolling(20).std() + 1e-12
)
valid7 = df.dropna(subset=["oi_z20", "ret_5d_fwd"]).copy()
valid7["oi_level"] = pd.qcut(valid7["oi_z20"], 5, labels=["极低", "偏低", "中", "偏高", "极高"])
for lv in ["极低", "偏低", "中", "偏高", "极高"]:
    s = valid7[valid7["oi_level"] == lv]
    wr = (s["ret_5d_fwd"] > 0).mean() * 100
    print(f"  持仓{lv}: n={len(s):4d} | 5日 {s['ret_5d_fwd'].mean()*100:+.3f}% | 胜率 {wr:.1f}%")
