# fg_model —— 玻璃主力 ML 模型

独立于 `fd_vib` 规则策略的 **机器学习模型**（LightGBM 预测未来 30m 收益）。

## 与 fd_vib 的区别

| | fd_vib | fg_model |
|--|--------|----------|
| 类型 | 规则策略（Don/EMA/动能） | **训练型模型** |
| 信号 | 固定公式 | LightGBM 打分 |
| 实盘 | `profile=smart` | 需先验证 IC / 回测 |

## 快速开始

```powershell
cd D:\LMS\fg_model
python train.py        # 静态切分训练（基础特征）
python wf_train.py       # Walk-forward + fd_vib 因子（推荐研究）
python wf_train.py --no-fd-vib   # WF 对照组
python predict.py        # 静态模型信号
python predict.py --wf   # WF 最后一折模型信号
python run.py            # 静态 train + predict
```

## 东财掘金下载 30m 数据

```powershell
pip install gm
# 复制 gm_token.example.txt → gm_token.txt，填入 Token

python download_gm.py --symbol FG609
python download_gm.py --symbol FG609 --start 2025-06-20 --end 2026-07-08
python download_gm.py --continuous              # 主力连续 CZCE.FG
python download_gm.py --symbol FG609 --sync-fg30m  # 另存 fg_30m 对比
```

产出：`fg_model/data/bars_30m_gm_FG609.parquet`（**不写 fd_vib**）

### 一键流程（东财 → 训练 → 信号）

```powershell
python download_gm.py --symbol FG609          # 1. 下载 30m
python import_gm_to_train.py --run-wf         # 2. 导入 + WF 训练
python predict.py --wf --source gm            # 3. 导出信号

# 或合并为一步（需已下载 GM parquet）
python run_gm.py --symbol FG609
```

训练缓存：
- `data/train_bars_30m.parquet`
- `data/train_source.json`（自动建议 WF/切分日期）

| 掘金参数 | 值 |
|----------|-----|
| 合约 | `CZCE.FG609` 或连续 `CZCE.FG` |
| 频率 | `1800s`（30 分钟） |
| 历史范围 | 郑商所分钟约 2017 至今（以账号权限为准） |

Token 申请：https://emquant.18.cn/ → 用户 → 密钥管理

## Walk-forward + fd_vib 因子

- **只读**加载 `fd_vib/registry.py` 与 `artifacts/factor_ic.json`
- **不修改** fd_vib 策略、combo、实盘配置
- 因子列加前缀 `fv_`，仅在 fg_model 特征矩阵中使用

| 脚本 | 说明 |
|------|------|
| `wf_train.py` | 按月滚动训练，拼接样本外预测 |
| `wf_summary.json` | 每折 IC + OOS 回测 |
| `wf_predictions.parquet` | 样本外 score/position |
| `fg_lgb_wf_last.pkl` | 最后一折模型（`predict --wf`） |

### WF 参数（config.py）

| 参数 | 默认 |
|------|------|
| WF_OOS_START | 2025-11-15 |
| WF_STEP_DAYS | 28 |
| WF_VALID_DAYS | 14 |
| FD_VIB_TOP_N | 24 |

## 目录

```
fg_model/
├── config.py       # 日期切分、LGB 参数
├── bars.py         # 主力 30m 拼接
├── features.py        # 价量 + build_feature_matrix
├── fd_vib_factors.py  # 只读 fd_vib 因子
├── walk_forward.py    # WF 核心
├── wf_train.py        # WF 训练入口
├── train.py           # 静态训练
├── predict.py         # 盘后信号
├── backtest.py        # 回测
├── run.py             # 一键静态训练
└── artifacts/
    ├── fg_lgb.pkl / fg_lgb_wf_last.pkl
    ├── wf_summary.json / wf_predictions.parquet
    └── live_signal.json
```

## 数据切分（防过拟合）

| 集合 | 截止 |
|------|------|
| train | ≤ 2026-02-28 |
| valid | 2026-02-28 ~ 2026-03-31 |
| test  | ≥ 2026-04-01 |

以 **test IC** 和 **test 回测** 为准，不看全样本 IC。

## 注意

- 本模型 **不自动替代** fd_vib 实盘
- 若 test IC ≈ 0，说明暂未学到稳定规律，继续调特征/标签/正则
- 有效后可写 VeighNa CTA 读 `artifacts/live_signal.json`
