"""检查 VeighNa AI 相关环境（vnpy.alpha / Station / 依赖）。"""
from __future__ import annotations

import importlib.util
import sys


def check(name: str) -> bool:
    ok = importlib.util.find_spec(name) is not None
    print(f"  {'OK' if ok else 'MISSING':8} {name}")
    return ok


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    print("\n[核心]")
    check("vnpy")
    try:
        import vnpy

        print(f"         vnpy version = {vnpy.__version__}")
        check("vnpy.alpha")
    except Exception as e:
        print(f"         vnpy import error: {e}")

    print("\n[vnpy.alpha 依赖]")
    deps = ["polars", "lightgbm", "alphalens", "matplotlib", "tqdm"]
    missing = [d for d in deps if not check(d)]

    print("\n[VeighNa Station / Assistant]")
    has_station = check("veighna_station")
    check("vnag")
    if not has_station:
        print(
            "\n提示: VeighNa Assistant 需要安装 VeighNa Studio，"
            "不会随 pip install vnpy 自动安装。"
        )
        print("下载: https://download.vnpy.com/veighna_studio-4.4.0.exe")

    print("\n[本仓库数据]")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    fg = root / "fg_30m" / "artifacts" / "bars_30m_FG609.parquet"
    fd = root / "fd_vib" / "vnpy" / "fg_vib_strategy.py"
    print(f"  {'OK' if fg.exists() else 'MISSING':8} {fg}")
    print(f"  {'OK' if fd.exists() else 'MISSING':8} {fd}")

    if missing:
        print(f"\n可安装缺失包: pip install {' '.join(missing)}")
        sys.exit(1)
    print("\n环境满足 vnpy.alpha 代码示例运行条件。")


if __name__ == "__main__":
    main()
