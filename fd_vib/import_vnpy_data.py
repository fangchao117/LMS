"""
VeighNa 数据导入 —— 1 分钟下载 / 入库 / 合成 30 分钟缓存

VeighNa 数据库只有 1m / 1h / 日线，没有 30m。标准做法：
  1. 下载或导入 1 分钟 K 线到 VeighNa
  2. 回测 / 仿真界面选 K 线周期 = 1 分钟
  3. FgVibDonStrategy 内部 BarGenerator 合成 30 分钟

免费数据源（项目已集成）—— 信易 EDB：
  · 日线：任意历史，免费
  · 1 分钟：最近约 1 年，免费
  · 文档：https://doc.shinnytech.com/edb/latest/md_server.html

示例：
    # 从 EDB 下载 FG609 1m，写入 vnpy，更新 fg_30m 30m 缓存
    python import_vnpy_data.py --symbol FG609 --start 2026-04-15 --end 2026-07-08

    # 已有 data/bars_1m_FG609.parquet，仅写入 vnpy
    python import_vnpy_data.py --symbol FG609 --from-cache
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOWNLOAD = ROOT / "data" / "download_edb.py"


def _sym_to_edb(symbol: str) -> str:
    sym = symbol.upper().replace(".CZCE", "")
    return f"CZCE.{sym}"


def import_parquet_to_vnpy(symbol: str) -> int:
    sym = symbol.upper().replace(".CZCE", "")
    path = DATA_DIR / f"bars_1m_{sym}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"无 1m 缓存: {path}，请先 EDB 下载")

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    sys.path.insert(0, str(DATA_DIR))
    from download_edb import import_minute_to_vnpy, update_fg30m_cache

    n = import_minute_to_vnpy(df, _sym_to_edb(sym))
    p1, p30, n30 = update_fg30m_cache(df, _sym_to_edb(sym))
    print(f"[import] vnpy 1m {n} 根")
    print(f"[import] 30m 缓存 {n30} 根 -> {p30}")
    return n


def download_and_import(symbol: str, start: str, end: str, user: str, password: str) -> None:
    edb = _sym_to_edb(symbol)
    cmd = [
        sys.executable,
        str(DOWNLOAD),
        "--symbol", edb,
        "--period", "1m",
        "--start", start,
        "--end", end,
        "--import-vnpy",
        "--update-fg30m",
    ]
    if user and password:
        cmd.extend(["--user", user, "--password", password])
    print("[import]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="VeighNa 1m 数据导入 + 30m 缓存")
    ap.add_argument("--symbol", default="FG609", help="如 FG609")
    ap.add_argument("--start", default="", help="开始 YYYY-MM-DD")
    ap.add_argument("--end", default="", help="结束 YYYY-MM-DD")
    ap.add_argument("--from-cache", action="store_true", help="从 data/bars_1m_*.parquet 导入")
    ap.add_argument("--user", default="", help="信易专业版账号（可选）")
    ap.add_argument("--password", default="", help="信易专业版密码（可选）")
    args = ap.parse_args()

    if args.from_cache:
        import_parquet_to_vnpy(args.symbol)
        _print_vnpy_hint(args.symbol)
        return

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start_dt = (
        datetime.strptime(args.start, "%Y-%m-%d")
        if args.start
        else end_dt - timedelta(days=365)
    )
    download_and_import(
        args.symbol,
        start_dt.strftime("%Y-%m-%d 09:00:00"),
        end_dt.strftime("%Y-%m-%d 15:00:00"),
        args.user,
        args.password,
    )
    _print_vnpy_hint(args.symbol)


def _print_vnpy_hint(symbol: str) -> None:
    sym = symbol.upper().replace(".CZCE", "")
    print(
        f"""
VeighNa 回测 / 仿真设置
-----------------------
  本地代码：{sym}.CZCE
  K 线周期：1 分钟   ← 必须选 1m，策略内部合成 30m
  策略类：  FgVibDonStrategy

若刚导入数据，请重启 VeighNa 或在「数据管理」刷新后再回测。
"""
    )


if __name__ == "__main__":
    main()
