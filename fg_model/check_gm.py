"""检查东财掘金 Token 与 FG 合约是否可拉数据。"""
from __future__ import annotations

import argparse
import sys

from download_gm import FREQ_30M, gm_symbol, resolve_token


def main() -> None:
    parser = argparse.ArgumentParser(description="检查掘金连接")
    parser.add_argument("--symbol", default="FG609")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    token = resolve_token(args.token)
    from gm.api import history, set_token

    set_token(token)
    code = gm_symbol(args.symbol, False)
    print(f"Token OK  合约={code}  频率={FREQ_30M}")

    df = history(
        symbol=code,
        frequency=FREQ_30M,
        start_time="2026-06-01 09:00:00",
        end_time="2026-06-05 15:00:00",
        fields="open,high,low,close,volume,position,bob",
        df=True,
    )
    if df is None or len(df) == 0:
        print("警告: 返回空数据。请检查 Token 权限、合约是否上市、掘金终端是否需登录。")
        sys.exit(1)

    print(f"样本 {len(df)} 根:")
    print(df.head(3).to_string())
    print("掘金连接正常，可运行: python download_gm.py --symbol FG609")


if __name__ == "__main__":
    main()
