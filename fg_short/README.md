# 玻璃短线（fg_short）

小资金 **1 万元** 玻璃主力 **FG00** 日线级短线波段策略。

## 说明

- **数据**：`data/FG00.CZCE.parquet`（日线，无分钟线）
- **持仓**：日线级短线波段（约 3~10 日），默认 **三均线趋势**（如 3/21/34）
- **手数**：按保证金估算，默认 **最多 1 手**（玻璃保证金 **14%**）
- **手续费**：开仓/平仓各 **2 元/手**，一开一平 **4 元/手**
- **样本**：TRAIN 调参，TEST 2023H2~2026 样本外（默认 3/21/34 约 +293%）

## 运行

```bash
cd fg_short
python backtest.py           # 默认 3/21/34，TEST 样本外
python backtest.py --tune    # 可选：TRAIN 段扫描均线
```

或直接打开 **`玻璃短线.ipynb`** 交互运行。

## 文件

| 文件 | 作用 |
|------|------|
| `config.py` | 资金、合约、信号参数 |
| `data.py` | 加载 parquet |
| `signals.py` | 短线信号 |
| `strategy.py` | vnpy AlphaStrategy |
| `backtest.py` | 调参 + 回测 |
| `artifacts/` | 日度曲线与统计 JSON |

## 回测结果（TEST 2023-07 ~ 2026-06）

| 指标 | 数值 |
|------|------|
| 起始资金 | 10,000 元 |
| 期末资金 | ~39,269 元 |
| 总收益 | **+292.7%** |
| 年化 | ~96.9% |
| Sharpe | ~0.99 |
| 最大回撤 | ~57% |
| 成交笔数 | 65 |

策略：三均线 **3/21/34**，最多 **1 手**，T 日信号 T+1 成交。

- `CAPITAL = 10_000`
- `MARGIN_RATE = 0.14`（玻璃一手保证金 14%）
- `COMMISSION_YUAN_PER_LOT = 2`（开/平各 2 元/手，往返 4 元）
- `SIGNAL_MODE`: `trend_ma`（默认）| `breakout` | `ema_cross` | `rsi_momo`
- `MA_SHORT / MA_MEDIUM / MA_LONG`: 三均线周期
- `MAX_LOTS`: 单笔手数上限（默认 1）
