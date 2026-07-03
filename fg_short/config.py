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

# 小资金
CAPITAL: int = 10_000
MARGIN_RATE: float = 0.14          # 郑商所玻璃保证金比例 14%
POSITION_PCT: float = 0.95         # 可用资金用于保证金的比例
MAX_LOTS: int = 1                  # 小资金默认 1 手封顶（防爆仓）
PRICE_ADD_TICKS: int = 1

# 样本划分
TRAIN_PERIOD: tuple[str, str] = ("2014-01-01", "2021-12-31")
VALID_PERIOD: tuple[str, str] = ("2022-01-01", "2023-06-30")
TEST_PERIOD: tuple[str, str] = ("2023-07-01", "2026-06-30")

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

SIGNAL_COL: str = "signal"


def ensure_dirs() -> None:
    LAB_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
