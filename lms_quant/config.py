"""
LMS 量化 —— 全局配置

对微博博主"实战期货的程序员"的 L.M.S 战法的**技术复刻尝试**：
将 L.M.S 解读为 Least Mean Squares（最小均方自适应滤波），
在线预测下一根 K 线收益并做多空择时。非其专有源码，规则可替换。
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
# 手续费：开仓/平仓各 2 元/手，一开一平 4 元/手（参考价折算费率）
COMMISSION_YUAN_PER_LOT: float = 2.0
COMMISSION_REF_PRICE: float = 1000.0
LONG_RATE: float = COMMISSION_YUAN_PER_LOT / (COMMISSION_REF_PRICE * CONTRACT_SIZE)
SHORT_RATE: float = LONG_RATE

# ---------------- LMS 自适应滤波参数 ----------------
# 多阶集成：不同滤波器阶数（抽头数）捕捉不同时间尺度的自相关，取平均更稳健。
FILTER_ORDERS: tuple[int, ...] = (3, 5, 10, 20)
MU: float = 0.05           # NLMS 步长（学习率），越大越灵敏、越易震荡；建议 1e-3 ~ 0.1
EPS: float = 1e-6          # NLMS 归一化项，防除零
LEAK: float = 1e-4         # 泄漏因子（权重轻微衰减，抗漂移/过拟合）
VOL_WINDOW: int = 20       # 输入收益的滚动波动率归一化窗口（尺度不变）
RETURN_MODE: str = "log"   # "log" 对数收益 / "pct" 百分比收益

# 信号模式：
#   "return" —— 预测下一根收益（纯自相关，日线上通常很弱）
#   "trend"  —— 自适应线性预测价格，取"预测价 vs 现价"方向做趋势跟随（推荐）
SIGNAL_MODE: str = "trend"
TREND_ORDERS: tuple[int, ...] = (5, 10, 20)   # 趋势滤波用的抽头数
TREND_MU: float = 0.5      # 价格预测的 NLMS 步长（NLMS 已按 ‖x‖² 归一，可偏大）
TREND_LEAK: float = 1e-5
SIGNAL_SMOOTH: int = 5     # 信号 EMA 平滑跨度（降换手；越大越平滑、越慢）

# ---------------- 策略 / 回测 ----------------
CAPITAL: int = 1_000_000
POSITION_PCT: float = 0.90
SIGNAL_THRESHOLD: float = 0.5   # 信号(z-score)死区；实测 0.5 收益回撤比最优，0=纯翻转
PRICE_ADD_TICKS: int = 2

# 与前面项目一致的样本外回测区间（滤波器在此之前已在线预热）
TEST_PERIOD: tuple[str, str] = ("2023-07-01", "2026-06-30")

SIGNAL_NAME: str = "lms_adaptive_signal"


def ensure_dirs() -> None:
    LAB_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
