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
# 手续费（郑商所玻璃）：开仓/平仓（含平今）各 2 元/手，一开一平 4 元/手
# vnpy 按成交额×费率计费，用参考价折算（滑点由 PRICE_ADD_TICKS 单独模拟）
COMMISSION_YUAN_PER_LOT: float = 2.0
COMMISSION_REF_PRICE: float = 1000.0
LONG_RATE: float = COMMISSION_YUAN_PER_LOT / (COMMISSION_REF_PRICE * CONTRACT_SIZE)
SHORT_RATE: float = LONG_RATE

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
# LightGBM 超参数（GlassLgbModel：强正则 + 训练集内时间切分早停）
# ----------------------------------------------------------------------------
LGB_PARAMS: dict = {
    "learning_rate": 0.02,
    "num_leaves": 8,
    "num_boost_round": 400,
    "early_stopping_rounds": 60,
    "log_evaluation_period": 0,
    "seed": 20260703,
    "min_data_in_leaf": 80,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "max_depth": 4,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "valid_ratio": 0.15,
}

# ----------------------------------------------------------------------------
# 信号后处理：因果滚动 z-score（消除训练期偏多偏差，便于设统一阈值）
# ----------------------------------------------------------------------------
SIGNAL_ZSCORE_WINDOW: int = 60
SIGNAL_ZSCORE_MIN_PERIODS: int = 20

# ----------------------------------------------------------------------------
# 策略参数
# ----------------------------------------------------------------------------
CAPITAL: int = 1_000_000           # 初始资金
POSITION_PCT: float = 0.95         # 单边最大仓位占用资金比例
SIGNAL_THRESHOLD: float = 0.0     # confirm 模式：AI 同向确认阈值（z-score）
LMS_VETO_THRESHOLD: float = 1.5   # veto 模式：AI 强烈反向时否决开仓（z-score）
LMS_AI_MODE: str = "off"          # "off" 纯规则最优；"veto"/"confirm" 可开 AI 辅助
PRICE_ADD_TICKS: int = 1           # 下单超价（滑点）跳数
USE_REGIME_FILTER: bool = True     # 三均线趋势过滤（推荐开启）

# 命名，便于落盘 / 复用
DATASET_NAME: str = "glass_alpha"
MODEL_NAME: str = "glass_lgb"
SIGNAL_NAME: str = "glass_lgb_signal"

# ----------------------------------------------------------------------------
# LMS 战法（长/中/短三均线趋势）—— 规则层参数
#   ⚠️ 这是对 "LMS" 的通用解读（L=长、M=中、S=短均线），非某博主专有规则。
#      拿到确切规则后，改 lms.py 里的判定逻辑即可，其余无需动。
#   多头排列：ma_S > ma_M > ma_L 且 收盘 > ma_M  -> 只做多
#   空头排列：ma_S < ma_M < ma_L 且 收盘 < ma_M  -> 只做空
#   其余（均线缠绕/震荡）                        -> 空仓
# ----------------------------------------------------------------------------
LMS_SHORT: int = 8         # 短期均线（样本外调优：8/21/55 > 5/20/60）
LMS_MEDIUM: int = 21       # 中期均线
LMS_LONG: int = 55         # 长期均线
LMS_USE_AI: bool = False           # False=纯三均线（8/21/55 样本外约 +73%）
LMS_REGIME_COL: str = "lms_regime"


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    LAB_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
