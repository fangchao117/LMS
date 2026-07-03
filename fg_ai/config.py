"""
玻璃期货（FG）AI 择时 —— 全局配置

集中管理路径、合约参数、样本划分、因子/模型/策略超参数。
所有脚本都从这里读取配置，改参数只改这一处。
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent          # D:\LMS
DATA_FILE: Path = ROOT / "data" / "FG00.CZCE.parquet"        # 原始日线数据
LAB_PATH: Path = ROOT / "lab"                                # vnpy AlphaLab 工作目录
ARTIFACT_PATH: Path = ROOT / "fg_ai" / "artifacts"           # 图表 / 结果输出

# ----------------------------------------------------------------------------
# 合约参数（郑商所 玻璃 FG）
# ----------------------------------------------------------------------------
SYMBOL: str = "FG00"
EXCHANGE_STR: str = "CZCE"
VT_SYMBOL: str = f"{SYMBOL}.{EXCHANGE_STR}"                  # "FG00.CZCE"

CONTRACT_SIZE: float = 20.0        # 合约乘数：20 吨/手
PRICE_TICK: float = 1.0            # 最小变动价位：1 元/吨
# 手续费近似为成交额比例（FG 约 6 元/手，名义额 ~2 万/手 -> ~3e-4）。
# 这里放大到含滑点的综合成本，多空各按成交额比例收取。
LONG_RATE: float = 2.5e-4
SHORT_RATE: float = 2.5e-4

# ----------------------------------------------------------------------------
# 样本划分（预留 2013 年数据做 60 日因子预热，训练标签从 2014 起）
#   train : 模型拟合
#   valid : 早停 + 超参监控
#   test  : 严格样本外，回测在此区间
# ----------------------------------------------------------------------------
TRAIN_PERIOD: tuple[str, str] = ("2014-01-01", "2021-12-31")
VALID_PERIOD: tuple[str, str] = ("2022-01-01", "2023-06-30")
TEST_PERIOD: tuple[str, str] = ("2023-07-01", "2026-06-30")

# ----------------------------------------------------------------------------
# 标签（预测目标）
#   回测撮合时序：T 日收盘产生信号 -> 委托在 T+1 日成交 -> 持有 HOLD 日。
#   因此标签取 "从 T+1 收盘到 T+1+HOLD 收盘" 的收益，避免使用信号日收盘、杜绝未来函数。
#     label = close[T+1+HOLD] / close[T+1] - 1
# ----------------------------------------------------------------------------
LABEL_HOLD: int = 2
LABEL_EXPR: str = f"ts_delay(close, -{1 + LABEL_HOLD}) / ts_delay(close, -1) - 1"

# ----------------------------------------------------------------------------
# LightGBM 超参数
#   已按"低配机器"调轻：迭代轮数不高、学习率略大、配合早停，
#   3000 多根日线 + 约 40 因子，训练通常在数十秒内完成，内存占用很小。
#   若想更充分拟合，可把 num_boost_round 调到 1500~2000、learning_rate 降到 0.02。
#   限制并行线程避免弱 CPU 卡死：运行前可设环境变量 OMP_NUM_THREADS=2。
# ----------------------------------------------------------------------------
LGB_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
    "log_evaluation_period": 50,
    "seed": 20260703,
}

# ----------------------------------------------------------------------------
# 策略参数
# ----------------------------------------------------------------------------
CAPITAL: int = 1_000_000           # 初始资金
POSITION_PCT: float = 0.90         # 单边最大仓位占用资金比例
SIGNAL_THRESHOLD: float = 0.0      # 信号死区：|pred|<=阈值 则空仓；0 表示纯多空翻转
PRICE_ADD_TICKS: int = 2           # 下单超价（滑点）跳数，PRICE_TICK 的倍数

# 命名，便于落盘 / 复用
DATASET_NAME: str = "glass_alpha"
MODEL_NAME: str = "glass_lgb"
SIGNAL_NAME: str = "glass_lgb_signal"


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    LAB_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
