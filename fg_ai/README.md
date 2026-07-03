# 玻璃期货（FG）AI 择时策略

基于 **vnpy 4.4 的 `vnpy.alpha` 模块**（AI 量化框架）搭建的单标的机器学习择时系统：
用 LightGBM 学习玻璃主力连续（`FG00.CZCE`）的量价因子，预测未来收益，
在 vnpy 回测引擎中做多空翻转，全流程杜绝未来函数。

---

## 目录结构

```
fg_ai/
├── config.py           # 全局配置：路径 / 合约参数 / 样本划分 / 超参
├── data_loader.py      # parquet -> polars(因子输入) / BarData(回测行情)
├── dataset.py          # GlassAlphaDataset：约 40 个量价因子 + 预处理器
├── strategy.py         # GlassAlphaStrategy：多空翻转择时
├── 01_prepare_lab.py   # 步骤1：行情写入 AlphaLab + 登记合约参数
├── 02_train.py         # 步骤2：训练 LightGBM + 计算 IC + 生成信号
├── 03_backtest.py      # 步骤3：样本外回测 + 绩效统计
├── run_all.py          # 一键跑完 01->02->03
├── requirements.txt
└── artifacts/          # 输出：净值 csv、绩效 json（运行后生成）

data/FG00.CZCE.parquet  # 原始日线（2013-01 ~ 2026-06，3273 根）
lab/                    # AlphaLab 工作目录（运行 01 后生成）
```

> 注：原空占位文件 `01_load_data.py` 已被 `01_prepare_lab.py` 取代，可删除。

---

## 运行

```bash
cd D:\LMS\fg_ai

# 低配机器建议先限制线程，避免 CPU 占满卡死
set OMP_NUM_THREADS=2          # Windows CMD
# export OMP_NUM_THREADS=2     # bash

python 01_prepare_lab.py       # 只需一次（数据更新后重跑）
python 02_train.py             # 训练 + 生成信号
python 03_backtest.py          # 回测
# 或： python run_all.py
```

**低配 / 易宕机时的建议**
- `config.LGB_PARAMS` 已调轻（`num_boost_round=500`，配合早停），3000 多根日线训练一般数十秒。
- 因子计算默认单进程（`build_dataset(..., max_workers=1)`），不会 fork 多进程。
- 仍吃力时可进一步减小 `num_boost_round`，或把样本区间缩短。

---

## 设计要点

### 1. 数据与尺度
主力连续 `close` 为**后复权价**，而 `turnover` 为未复权口径，两者尺度不一致，
因此 `vwap` 不用 `turnover/volume` 反推，改用典型价 `(H+L+C)/3`，仅参与 `vwap/close` 类比值因子。

### 2. 因子集（`dataset.py`）
Qlib 风格表达式 DSL，精选约 40 个稳健因子（比全套 Alpha158 更抗过拟合）：

| 组别 | 因子 | 含义 |
|---|---|---|
| K 线形态 | kmid / klen / kup / klow / ksft | 当日多空力量 |
| 动量 | roc_{5,10,20,60} | 过去 w 日收益 |
| 均线 | ma_{w} / beta_{w} | 乖离与趋势斜率 |
| 波动 | std_{w} / atr_{14,24} | 风险度量 |
| 通道 | rsv_{w} / maxd_{w} / mind_{w} | 区间位置 |
| 摆动 | rsi_{6,14,24} | 超买超卖 |
| 量能 | vma_{w} / vcorr_{w} | 成交量确认 |
| 持仓 | oi_{5,20} | 资金流向 |

预处理：**稳健 Z-Score 归一化**（仅用训练窗口统计量拟合，防泄漏）+ 特征缺失填 0 + 丢弃标签缺失行。

### 3. 标签与撮合时序（无未来函数）
```
T 日收盘 → 生成信号并挂单 → T+1 日撮合成交 → 持有 LABEL_HOLD 日
标签 = close[T+1+HOLD] / close[T+1] - 1
```
信号日收盘不参与建仓价，标签口径与实际持仓区间严格对齐。

### 4. 样本划分
| 段 | 区间 | 用途 |
|---|---|---|
| TRAIN | 2014-01 ~ 2021-12 | 拟合 |
| VALID | 2022-01 ~ 2023-06 | 早停 |
| TEST | 2023-07 ~ 2026-06 | **样本外回测** |

（2013 年数据保留作 60 日因子预热）

### 5. 策略（`strategy.py`）
```
pred >  阈值  → 满仓做多
pred < -阈值  → 满仓做空
|pred| ≤ 阈值 → 空仓（阈值默认 0，即纯多空翻转）
```
仓位按当前组合净值的 `POSITION_PCT` 折算为整数手；下单超价 `PRICE_ADD_TICKS` 跳模拟滑点。
合约参数：乘数 20 吨/手、最小变动 1 元/吨、综合成本约 2.5‱（含手续费+滑点近似）。

### 6. 输出
- 控制台：训练/验证/测试三段 **IC / RankIC**（预测有效性），回测**年化/回撤/夏普/胜率**等。
- `artifacts/backtest_daily.csv`：逐日净值。
- `artifacts/backtest_stats.json`：完整绩效指标。

---

## 调参入口（都在 `config.py`）

| 参数 | 作用 |
|---|---|
| `LABEL_HOLD` | 持有天数 / 预测跨度 |
| `LGB_PARAMS` | 模型复杂度与训练轮数 |
| `SIGNAL_THRESHOLD` | 信号死区，>0 时低置信度不开仓 |
| `POSITION_PCT` | 单边仓位占资比例 |
| `TRAIN/VALID/TEST_PERIOD` | 样本划分 |

---

## 免责声明
本项目为量化研究/教学示例。历史回测不代表未来收益，实盘前需考虑
真实手续费、冲击成本、主力换月、涨跌停无法成交等因素，并做充分的
样本外与稳健性检验。
