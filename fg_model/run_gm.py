"""东财数据一键：导入 → Walk-forward 训练 → 导出信号。"""
from __future__ import annotations

import argparse

from import_gm_to_train import import_gm


def main() -> None:
    parser = argparse.ArgumentParser(description="GM 30m 导入 + WF 训练 + 信号")
    parser.add_argument("--symbol", default="FG609")
    parser.add_argument("--no-fd-vib", action="store_true")
    parser.add_argument("--skip-wf", action="store_true", help="仅导入不训练")
    args = parser.parse_args()

    import_gm(args.symbol)
    if args.skip_wf:
        print("已导入。训练: python wf_train.py --source gm")
        return

    import wf_train
    import sys

    argv = ["wf_train.py", "--source", "gm"]
    if args.no_fd_vib:
        argv.append("--no-fd-vib")
    old = sys.argv
    try:
        sys.argv = argv
        wf_train.main()
    finally:
        sys.argv = old

    import predict
    argv2 = ["predict.py", "--wf", "--source", "gm"]
    old = sys.argv
    try:
        sys.argv = argv2
        predict.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
