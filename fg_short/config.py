"""
玻璃短线 —— 全局配置

小资金（1 万）玻璃主力 FG00 日线级短线波段：
  · 持仓 1~5 日，信号基于快 EMA / 通道突破 / RSI 动量
  · 按保证金估算可开手数（非全额名义）
"""
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
DATA_FILE: Path = ROOT / "data" / "FG00.CZCE.parquet"
LAB_PATH: Path = ROOT / "lab"
ARTIFACT_PATH: Path = Path(__file__).resolve().parent / "artifacts"

# 合约
SYMBOL: str = "FG00"
EXCHANGE_STR: str = "CZCE"
VT_SYMBOL: str = f"{SYMBOL}.{EXCHANGE_STR}"
CONTRACT_SIZE: float = 20.0
PRICE_TICK: float = 1.0

# 手续费（郑商所玻璃）：开仓/平仓（含平今）各 2 元/手，一开一平 4 元/手
# vnpy 按成交额×费率计费，以下用参考价折算（实际随价格略有浮动）
COMMISSION_YUAN_PER_LOT: float = 2.0       # 元/手/边
COMMISSION_REF_PRICE: float = 1000.0         # 折算参考价（元/吨）
LONG_RATE: float = COMMISSION_YUAN_PER_LOT / (COMMISSION_REF_PRICE * CONTRACT_SIZE)
SHORT_RATE: float = LONG_RATE

# 小资金 → 5万规模（按权益动态算手数，封顶 max_lots）
CAPITAL: int = 10_000
CAPITAL_MAX: int = 50_000          # 算手数时权益上限（1万起步、最多按5万规模）
MARGIN_RATE: float = 0.14          # 郑商所玻璃保证金比例 14%
POSITION_PCT: float = 0.95         # 可用资金用于保证金的比例
MAX_LOTS: int = 5                  # 净持仓上限（手）；1万约1~3手，5万可顶格5手
MAX_TOTAL_LOTS: int = 10           # 锁仓模式下 多+空 合计上限（如各1手=2）
PRICE_ADD_TICKS: int = 1

# 主力月份（1/5/9），换月见 vnpy_sim/dominant_contract.py
DOMINANT_MONTHS: tuple[int, ...] = (1, 5, 9)

# 样本划分（统一样本）
BACKTEST_PERIOD: tuple[str, str] = ("2024-12-20", "2025-04-20")
TRAIN_PERIOD: tuple[str, str] = BACKTEST_PERIOD
VALID_PERIOD: tuple[str, str] = BACKTEST_PERIOD
TEST_PERIOD: tuple[str, str] = BACKTEST_PERIOD

# 短线信号（VALID+TEST 双段调优默认：三均线趋势 1 手）
SIGNAL_MODE: str = "trend_ma"      # "trend_ma" | "breakout" | "ema_cross" | "rsi_momo"
MA_SHORT: int = 3
MA_MEDIUM: int = 21
MA_LONG: int = 34
EMA_FAST: int = 5
EMA_SLOW: int = 13
DONCHIAN: int = 5
RSI_PERIOD: int = 6
RSI_OB: float = 70.0             # 超买
RSI_OS: float = 30.0             # 超卖
SIGNAL_THRESHOLD: float = 0.0    # 复合得分死区（0=纯翻转）
MIN_HOLD_DAYS: int = 1           # 最短持有（策略层由信号自然换仓）

# 风控（收益最大化默认关闭；需要控回撤时在 VeighNa / tune_valid 里开）
MIN_TREND_PCT: float = 0.0       # 0=不过滤，追求收益
STOP_LOSS_PCT: float = 0.0
MAX_DRAWDOWN_PCT: float = 0.0    # 0=关闭账户熔断

SIGNAL_COL: str = "signal"


def ensure_dirs() -> None:
    LAB_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
