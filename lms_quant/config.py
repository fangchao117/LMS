"""
LMS 量化 —— 全局配置

LMS 自适应滤波 + 多因子融合择时。
仓位按账户净值比例动态计算，不限定固定手数上限（收益最大化取向）。
"""
from pathlib import Path

# ---------------- 路径 ----------------
HERE: Path = Path(__file__).resolve().parent          # D:\LMS\lms_quant
ROOT: Path = HERE.parent                              # D:\LMS
DATA_FILE: Path = ROOT / "data" / "FG00.CZCE.parquet"
LAB_PATH: Path = ROOT / "lab"                         # 复用已建好的 AlphaLab
ARTIFACT_PATH: Path = HERE / "artifacts"

# ---------------- 合约（郑商所 玻璃 FG）----------------
SYMBOL: str = "FG00"
EXCHANGE_STR: str = "CZCE"
VT_SYMBOL: str = f"{SYMBOL}.{EXCHANGE_STR}"
CONTRACT_SIZE: float = 20.0
PRICE_TICK: float = 1.0
COMMISSION_YUAN_PER_LOT: float = 2.0
COMMISSION_REF_PRICE: float = 1000.0
LONG_RATE: float = COMMISSION_YUAN_PER_LOT / (COMMISSION_REF_PRICE * CONTRACT_SIZE)
SHORT_RATE: float = LONG_RATE

# ---------------- LMS 自适应滤波参数 ----------------
FILTER_ORDERS: tuple[int, ...] = (3, 5, 10, 20)
MU: float = 0.05
EPS: float = 1e-6
LEAK: float = 1e-4
VOL_WINDOW: int = 20
RETURN_MODE: str = "log"

# 信号模式："trend" | "return" | "multi"（LMS + 多因子融合，推荐）
SIGNAL_MODE: str = "multi"
TREND_ORDERS: tuple[int, ...] = (5, 10, 20)
TREND_MU: float = 0.5
TREND_LEAK: float = 1e-5
SIGNAL_SMOOTH: int = 5

# ---------------- 多因子融合 ----------------
USE_MULTI_FACTOR: bool = True
FACTOR_WEIGHTS: dict[str, float] = {
    "lms": 0.30,         # NLMS 自适应趋势（主因子）
    "momentum": 0.20,    # 多周期动量
    "ma_regime": 0.25,   # 三均线趋势
    "volume": 0.10,      # 量价配合
    "oi": 0.10,          # 持仓量
    "rsi": 0.05,         # RSI 摆动
    "atr_trend": 0.00,   # ATR 趋势（默认关闭，可开）
}
MA_SHORT: int = 8
MA_MEDIUM: int = 21
MA_LONG: int = 55
MOM_WINDOWS: tuple[int, ...] = (5, 10, 20)
RSI_PERIOD: int = 14
OI_WINDOW: int = 5
FACTOR_ZSCORE_MIN: int = 20
# 至少 N 个子因子方向与合成信号一致才开仓；0=不过滤（收益最大化）
MIN_FACTOR_AGREE: int = 0

# ---------------- 策略 / 回测 ----------------
CAPITAL: int = 1_000_000          # 回测初始净值（仅作起点，手数随净值动态缩放）
POSITION_PCT: float = 0.98        # 满仓比例，不限定固定手数
SIGNAL_THRESHOLD: float = 0.0       # 0=纯多空翻转，最大化换手与收益
PRICE_ADD_TICKS: int = 1
TUNE_THRESHOLD: bool = True         # VALID 段扫描阈值，取收益最大者用于 TEST
THRESHOLD_CANDIDATES: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)

VALID_PERIOD: tuple[str, str] = ("2022-01-01", "2023-06-30")
TEST_PERIOD: tuple[str, str] = ("2023-07-01", "2026-06-30")

SIGNAL_NAME: str = "lms_multi_signal"


def ensure_dirs() -> None:
    LAB_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
