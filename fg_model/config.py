"""fg_model —— 玻璃主力 ML 因子模型配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
DATA_ROOT = ROOT.parent

# 东财 GM 导入后的训练缓存（仅 fg_model，不写 fd_vib）
TRAIN_BARS_PATH = DATA_DIR / "train_bars_30m.parquet"
TRAIN_SOURCE_META = DATA_DIR / "train_source.json"

# 与 fd_vib 对齐的主力区间
TRAIN_END = "2026-02-28"
VALID_END = "2026-03-31"
TEST_START = "2026-04-01"
BACKTEST_END = "2026-07-08"

LABEL_HORIZON = 4          # 预测未来 4 根 30m K 线收益
SIGNAL_THRESHOLD = 0.00012 # 预测值 → 多空阈值

CAPITAL = 300_000
MAX_LOTS = 2
CONTRACT_SIZE = 20.0
COMMISSION_PER_LOT = 2.0
SLIPPAGE_TICKS = 1
PRICE_TICK = 1.0

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 23,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 800
EARLY_STOPPING = 60

# Walk-forward 滚动训练
WF_OOS_START = "2025-11-15"   # 首个样本外预测起点
WF_STEP_DAYS = 28             # 每折预测窗口（约 1 月）
WF_VALID_DAYS = 14            # 早停验证窗
WF_MIN_TRAIN_BARS = 1000      # 最少训练 K 线数（GM 短数据 import 时会下调）

# fd_vib 因子（只读）
FD_VIB_TOP_N = 24
USE_FD_VIB_IN_WF = True
