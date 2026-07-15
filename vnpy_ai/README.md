# VeighNa「AI-Powered」使用指南

[vnpy](https://github.com/vnpy/vnpy) 4.x 的 **AI-Powered** 实际包含 **两条独立能力**，不要混为一谈：

| 能力 | 入口 | 你需要什么 | 适合做什么 |
|------|------|------------|------------|
| **VeighNa Assistant** | VeighNa **Station** 聊天窗口 | 安装 [VeighNa Studio](https://download.vnpy.com/veighna_studio-4.4.0.exe) + 大模型 API Key | 问文档、写策略、调 VeighNa、Agent 任务 |
| **vnpy.alpha** | Python 代码 `from vnpy.alpha import AlphaLab` | `pip install vnpy` + `lightgbm` 等 | 因子工程 → ML 训练 → Alpha 回测 |

你当前环境（`pip install vnpy 4.4.0`）**已有 vnpy.alpha**，但 **没有 VeighNa Station**，因此 **Assistant 聊天助手需要另装 Studio**。

---

## 一、VeighNa Assistant（图形界面 AI 助手）

> 官方说明：[VeighNa 4.3.0 - VeighNa Assistant](https://www.vnpy.com/forum/topic/34426-veighnafa-bu-v4-3-0-veighna-assistant)

### 1.0 社区版 Trader 里没有 Assistant（重要）

你截图里的 **「VeighNa Trader 社区版」** 是交易终端，**本身没有 AI 聊天窗口**。

Assistant 在上一级程序 **VeighNa Station** 里：

```
VeighNa Station（AI 助手在这里）
    └─ 左侧【交易】→ 勾选 CTP / CTA →【启动】→ VeighNa Trader 社区版
```

| 程序 | Assistant |
|------|-----------|
| **VeighNa Station** | ✅ 有（4.3+，菜单「功能 → AI服务配置」） |
| **VeighNa Trader** | ❌ 无 |

**操作**：先 **关闭 Trader**，回到桌面打开 **VeighNa Station**，再按 1.2 配置 API；配置完成后在 Station **主界面聊天区** 对话，需要交易时再点【交易】→【启动】进 Trader。

若 Station 菜单里没有「AI服务配置」：版本低于 4.3，请左侧【更新】或重装 [VeighNa Studio 4.4.0](https://download.vnpy.com/veighna_studio-4.4.0.exe)。

### 1.1 安装

1. 下载并安装 **VeighNa Studio 4.4.0**（内置 Python 3.13 + VeighNa + Station）
2. 用社区账号登录 Station（论坛账号即 Station 账号）
3. 大版本升级（如 3.x → 4.x）需 **卸载后重装**，不能仅靠 pip 升级

### 1.2 配置大模型 API

1. 启动 **VeighNa Station**（桌面快捷方式，或 `python -m veighna_station`）
2. 菜单 **功能 → AI服务配置**
3. 选择服务商（新手推荐 **Bailian / 阿里云百炼**，有免费额度）
4. 填入 **API Key**，保存后 **重启 Station**
5. 菜单 **功能 → 模型浏览器**，添加模型（百炼推荐 `qwen3-max-preview` 或 `kimi-k2-thinking`）
6. 保存后在主界面聊天区使用

### 1.3 能帮你做什么

- 查 VeighNa 官方文档、示例策略、源码
- 用 **Agentic AI** 调用工具完成多步任务（写 CTA 策略、改回测参数等）
- 自然语言描述需求，让助手生成/修改 Python 策略代码

### 1.4 与 fd_vib 结合示例（在 Assistant 里提问）

```
请根据 D:\LMS\fd_vib\vnpy\fg_vib_strategy.py 的 FgVibDonStrategy，
说明 VeighNa CTA 模块里 profile=smart、signal_mode=json 的参数含义，
并检查我实盘配置 vt_symbol=FG609.CZCE 是否正确。
```

### 1.5 常见问题

| 问题 | 处理 |
|------|------|
| Station 闪退 / API 填错 | 把用户目录下 `.vnag` 文件夹改名，更新 Station 后再改回 |
| 只有 pip vnpy，没有 Station | 正常；Assistant **不随 pip 分发**，需装 Studio |
| 本地 Ollama DeepSeek | 官方模型浏览器主要对接云 API；本地模型需自行对接 OpenAI 兼容接口 |

---

## 二、vnpy.alpha（机器学习量化模块）

> 源码：[vnpy/alpha](https://github.com/vnpy/vnpy/tree/master/vnpy/alpha)

### 2.1 标准流程

```
行情数据 → AlphaLab 落盘
    → AlphaDataset 因子 + 标签
    → AlphaModel 训练（Lasso / LightGBM / MLP）
    → 预测 signal → AlphaStrategy 回测
```

### 2.2 依赖

```powershell
pip install vnpy polars lightgbm alphalens-reloaded matplotlib
```

检查环境：

```powershell
cd D:\LMS\vnpy_ai
python 01_check_env.py
```

### 2.3 本仓库示例（玻璃期货 FG）

| 脚本 | 作用 |
|------|------|
| `01_check_env.py` | 检查 vnpy / alpha / 可选依赖 |
| `02_import_fg_to_alpha_lab.py` | 把 `fg_30m` 30m 缓存导入 AlphaLab |
| `03_alpha_demo_lgb.py` | 单合约 FG609 上跑通 LGB 训练 demo |
| `04_export_signal.py` | 模型 → 导出 signal parquet |
| `05_alpha_backtest.py` | signal → Alpha 回测验证 |

```powershell
cd D:\LMS\vnpy_ai

# 1) 检查环境
python 01_check_env.py

# 2) 导入 FG609 30m → AlphaLab 数据目录
python 02_import_fg_to_alpha_lab.py --symbol FG609

# 3) LightGBM 训练 demo（输出到 artifacts/）
python 03_alpha_demo_lgb.py --symbol FG609

# 4) 导出预测信号
python 04_export_signal.py --symbol FG609

# 5) 用 signal 做 Alpha 回测
python 05_alpha_backtest.py --symbol FG609
```

数据目录默认：`D:\LMS\vnpy_ai\lab\`

### 2.4 核心 API 速查

```python
from vnpy.alpha import AlphaLab, AlphaDataset, Segment
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.lgb_model import LgbModel

lab = AlphaLab("D:/LMS/vnpy_ai/lab")
df = lab.load_bar_df(["FG609.CZCE"], interval="1m", start="2025-09-01", end="2026-07-01", extended_days=30)

dataset = Alpha158(df, train_period=("2025-09-01", "2026-03-01"),
                   valid_period=("2026-03-02", "2026-05-01"),
                   test_period=("2026-05-02", "2026-07-01"))
dataset.set_label("ts_delay(close, -5) / close - 1")   # 未来 5 根 K 线收益
dataset.prepare_data()

model = LgbModel()
model.fit(dataset)
pred = model.predict(dataset, Segment.TEST)

lab.save_dataset("fg609_alpha158", dataset)
lab.save_model("fg609_lgb", model)
```

内置因子集：

- `Alpha158` — 微软 Qlib 158 因子
- `Alpha101` — WorldQuant 101 因子（4.3+）

### 2.5 与 fd_vib 的关系

| 项目 | 类型 | 说明 |
|------|------|------|
| **fd_vib** | 规则策略（Don + EMA + 动能） | 已对接 VeighNa CTA 仿真/实盘，**当前生产路径** |
| **vnpy.alpha** | ML 预测信号 | 需单独训练、验证、再封装为 CTA/Alpha 策略 |

建议：**实盘继续用 fd_vib 的 `profile=smart` + `signal_mode=json`**；vnpy.alpha 用于 **研究新因子/新模型**，验证有效后再考虑接入。

### 2.6 训练好的模型怎么用？

`03` 只完成 **训练 + 存盘**。完整链路：

```
03 训练 → 04 导出 signal → 05 Alpha 回测 → （有效后再接 CTA）
```

**04 导出 signal**（列：`datetime`, `vt_symbol`, `signal`）：

```powershell
python 04_export_signal.py --symbol FG609
# → lab/signal/FG609_lgb_test.parquet
```

**05 回测验证**（预测 > 阈值做多，< -阈值做空）：

```powershell
python 05_alpha_backtest.py --symbol FG609
# → artifacts/alpha_backtest_FG609.json
```

**因子质量分析**（可选，会弹出 alphalens 图表）：

```python
lab = AlphaLab("D:/LMS/vnpy_ai/lab")
dataset = lab.load_dataset("FG609_demo")
signal = lab.load_signal("FG609_lgb_test")
dataset.show_signal_performance(signal)
```

**关于你这次训练结果**：Early stopping 在第 **1** 轮、测试集 `mean≈-0.0004`，说明 demo 模型 **几乎无预测力，不能实盘**。需要更多数据、更多因子（如 Alpha158）、调参后重复 03→05，直到回测指标合理。

将来若模型有效，可写 CTA 策略读取 `lab/signal/*.parquet` 调仓（类似 fd_vib 读 `live_signal.json`）。

---

## 三、怎么选？

```
只想在界面里问 VeighNa、自动生成策略代码？
  → 装 VeighNa Studio + 配置 Assistant API

想用机器学习做因子预测、Alpha 回测？
  → 用本目录 vnpy_ai 脚本跑 vnpy.alpha

继续玻璃主力规则交易（你现在的 fd_vib）？
  → 不必切到 alpha；Assistant 可当「技术顾问」
```

---

## 四、参考链接

- VeighNa GitHub：https://github.com/vnpy/vnpy
- VeighNa 社区：https://www.vnpy.com/portal/
- Assistant 发布公告：https://www.vnpy.com/forum/topic/34426-veighnafa-bu-v4-3-0-veighna-assistant
- fd_vib 使用说明：`D:\LMS\fd_vib\使用说明.md`
