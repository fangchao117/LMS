"""
玻璃 30 分钟短线 —— 配置

· 数据：vnpy SQLite 1 分钟 → 合成 30 分钟
· 目标：收益最大化，接受日内多笔交易
· 资金：1 万起步
"""
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
ARTIFACT_PATH: Path = Path(__file__).resolve().parent / "artifacts"

# 回测：默认主力拼接（FG509→601→605→609）；仿真仍用具体月份 SYMBOL
USE_DOMINANT_BACKTEST: bool = True
SYMBOL: str = "FG609"
BACKTEST_PERIOD: tuple[str, str] = ("2025-06-20", "2026-07-08")
TUNE_PERIOD: tuple[str, str] = ("2025-06-20", "2026-03-31")
TEST_PERIOD: tuple[str, str] = ("2026-04-01", "2026-07-08")
EXCHANGE_STR: str = "CZCE"
VT_SYMBOL: str = f"{SYMBOL}.{EXCHANGE_STR}"
CONTRACT_SIZE: float = 20.0
PRICE_TICK: float = 1.0
COMMISSION_YUAN_PER_LOT: float = 2.0
SLIPPAGE_TICKS: int = 1

CAPITAL: int = 10_000
MARGIN_RATE: float = 0.14
POSITION_PCT: float = 0.95
MAX_LOTS: int = 3

# 样本（FG605 约 2025-06 ~ 2026-05；按日历切）
# 已由 BACKTEST_PERIOD 统一，以下保留供 tune 脚本参考

# 策略档位：stable=稳定（少交易、控回撤）  aggressive=追求收益（此前唐奇安方案）
STRATEGY_PROFILE: str = "stable"

# stable：日线 EMA 定方向 + 30m EMA + ADX/发散度过滤（四段中相对最稳）
# aggressive：日线 EMA + 30m 唐奇安突破（全段收益更高但 FG605 易大亏）
SIGNAL_MODE: str = "daily_filter_ema_stable"
EMA_FAST: int = 26
EMA_SLOW: int = 46
DONCHIAN: int = 20
ADX_THRESHOLD: float = 35.0
TREND_MA: int = 50
MA_DIVERGENCE_THRESHOLD: float = 0.015
DAILY_SYMBOL: str = "FG00"
BAR_MINUTES: int = 30
HTF_MINUTES: int = 60

# 多周期：60m 看方向，30m 交易（EMA 周期按 K 线根数，非分钟）
HTF_EMA_FAST: int = 6
HTF_EMA_SLOW: int = 24

SIGNAL_COL: str = "signal"


def ensure_dirs() -> None:
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
