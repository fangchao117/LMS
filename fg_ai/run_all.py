"""
一键流水线：准备数据 -> 训练 -> 回测

    python run_all.py
"""
from __future__ import annotations

import importlib


def main() -> None:
    print("=" * 60)
    print(" 玻璃期货 AI 择时 —— 全流程")
    print("=" * 60)

    prepare = importlib.import_module("01_prepare_lab")
    train = importlib.import_module("02_train")
    backtest = importlib.import_module("03_backtest")

    prepare.main()
    print()
    train.main()
    print()
    backtest.main()

    print("\n完成。")


if __name__ == "__main__":
    main()
