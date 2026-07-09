"""fd_vib —— 玻璃主力量化（实盘仿真）"""
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
ARTIFACT_PATH: Path = Path(__file__).resolve().parent / "artifacts"

BACKTEST_PERIOD: tuple[str, str] = ("2025-06-20", "2026-07-08")
TUNE_PERIOD: tuple[str, str] = ("2025-06-20", "2026-03-31")
TEST_PERIOD: tuple[str, str] = ("2026-04-01", "2026-07-08")

CAPITAL: int = 10_000
MAX_LOTS: int = 2
MARGIN_RATE: float = 0.14
CONTRACT_SIZE: float = 20.0
PRICE_TICK: float = 1.0
COMMISSION_YUAN_PER_LOT: float = 2.0
SLIPPAGE_TICKS: int = 1
ROLL_EXTRA_COMM_MULT: float = 1.0  # 换月平仓额外 1x 手续费

IC_HORIZON: int = 4
IC_MIN_ABS: float = 0.02
TOP_FACTORS: int = 8

# 调参产出覆盖；默认收益最大化
DEFAULT_SPEC: dict = {
    "mode": "donchian_baseline",
    "donchian": 38,
    "daily_filter": False,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# 低回撤备选：Don30，回撤约 -62%，收益约 +121%
BALANCED_SPEC: dict = {
    "mode": "donchian_baseline",
    "donchian": 30,
    "daily_filter": False,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# 因子增强备选（震荡过滤，收益较低但过滤假突破）
ENHANCED_SPEC: dict = {
    "mode": "don_enhanced",
    "donchian": 38,
    "daily_filter": False,
    "threshold": 0.35,
    "smooth": 3,
    "chop_max": 0.50,
    "vol_min": -1.2,
    "strict": True,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}


def ensure_dirs() -> None:
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
