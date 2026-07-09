"""
将 FgVibDonStrategy 安装到 VeighNa 界面版策略目录。

    cd D:\\LMS\\fd_vib
    python install_vnpy.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "vnpy" / "fg_vib_strategy.py"


def strategy_dirs() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".vntrader" / "strategies",
        home / "strategies",
        Path(r"C:\veighna_studio\strategies"),
        Path(r"D:\veighna_studio\strategies"),
    ]
    if sys.platform == "win32":
        appdata = Path.home() / "AppData" / "Roaming" / "VeighNa" / "strategies"
        candidates.append(appdata)
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"策略源文件不存在: {SRC}")

    installed: list[Path] = []
    for d in strategy_dirs():
        d.mkdir(parents=True, exist_ok=True)
        dst = d / "fg_vib_strategy.py"
        shutil.copy2(SRC, dst)
        installed.append(dst)

    primary = installed[0]
    print("已安装 FgVibDonStrategy ->")
    for p in installed:
        print(f"  {p}")

    print(
        f"""
VeighNa 界面版 —— 仿真 / 实盘步骤
================================

1. 数据准备（首次）
   cd D:\\LMS\\fg_30m
   python refresh_cache.py          # 1m → 30m 缓存

2. 盘后信号（可选，JSON 模式）
   cd D:\\LMS\\fd_vib
   python run.py --live --profile max
   -> artifacts/live_signal.json

3. 安装策略（本脚本，改代码后重跑）
   python install_vnpy.py

4. 重启 VeighNa Studio，打开「CTA 策略」模块

5. 仿真账户
   · 连接 CTP 仿真 / SimNow
   · 订阅当期主力，例如 FG609.CZCE
   · K 线周期：30 分钟

6. 添加策略 FgVibDonStrategy
   · vt_symbol = FG609.CZCE（与 dominant 一致）
   · fd_vib_root = {ROOT}
   · profile = max（或 balanced）
   · signal_mode = compute（策略内算信号）
     或 json（读 live_signal.json，适合先 semi-auto 验证）

7. 换月（投机客户，不平移）
   · 换月日：停止旧合约策略 → 平掉旧仓
   · 在新主力合约上新建策略实例
   · 更新 fd_vib/dominant.py 中的 DOMINANT_SEGMENTS 后 refresh 数据

8. 实盘
   · 与仿真相同，换真实 CTP 网关
   · 建议先用 signal_mode=json + 人工核对 1~2 周

策略文件: {primary}
"""
    )


if __name__ == "__main__":
    main()
