"""fd_vib —— 玻璃主力量化（实盘仿真）"""
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
ARTIFACT_PATH: Path = Path(__file__).resolve().parent / "artifacts"

BACKTEST_PERIOD: tuple[str, str] = ("2025-06-20", "2026-07-08")
TUNE_PERIOD: tuple[str, str] = ("2025-06-20", "2026-03-31")
TEST_PERIOD: tuple[str, str] = ("2026-04-01", "2026-07-08")

CAPITAL: int = 10_000
MAX_LOTS: int = 3
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


# 每主力段 smart：warm 窗 auto↔regime 自动切换（推荐，弱化段如 FG605 用 regime）
SMART_SPEC: dict = {
    "mode": "don_smart",
    "donchian": 38,
    "chop_hi": 0.38,
    "warm_bars": 320,
    "don_candidates": (25, 30, 35, 38, 45),
    "fallback_don": 38,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# 震荡/趋势自适应：震荡=方向+Don，趋势=每段自校准 Don
REGIME_SPEC: dict = {
    "mode": "don_regime",
    "donchian": 38,
    "chop_hi": 0.38,
    "warm_bars": 320,
    "don_candidates": (25, 30, 35, 38, 45),
    "fallback_don": 38,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# 每主力段自动校准 Don（无震荡切换；FG605 等震荡段偏弱）
AUTO_SPEC: dict = {
    "mode": "don_auto",
    "donchian": 38,
    "warm_bars": 320,
    "don_candidates": (25, 30, 35, 38, 45),
    "fallback_don": 38,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# L.M.S —— 空间(Don) + 方向(日线EMA) + 动能(近端K线/量价/持仓)
LMS_SPEC: dict = {
    "mode": "don_lms",
    "donchian": 38,
    "require_daily": True,
    "mom_mode": "score",
    "mom_min": 0.05,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# 仅方向（消融对比）
LMS_DIR_SPEC: dict = {
    "mode": "don_lms",
    "donchian": 38,
    "require_daily": True,
    "mom_mode": "off",
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}

# 旧版硬阈值动能（保留）
LMS_MOM_SPEC: dict = {
    "mode": "don_lms",
    "donchian": 38,
    "require_daily": True,
    "mom_mode": "soft",
    "adx_min": 15.0,
    "chop_max": 0.52,
    "vol_min": -0.45,
    "max_lots": 2,
    "use_risk": True,
    "atr_stop_mult": 3.5,
    "trail_atr_mult": 4.0,
    "max_loss_pct": 6.0,
}


def ensure_dirs() -> None:
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
