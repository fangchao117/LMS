"""将 fg_30m 的 30 分钟 parquet 导入 vnpy.alpha 的 AlphaLab 目录。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from vnpy.alpha import AlphaLab

ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = Path(__file__).resolve().parent / "lab"
FG_SIZE = 20  # 玻璃每点 20 元


def load_fg_bars(symbol: str) -> pd.DataFrame:
    path = ROOT / "fg_30m" / "artifacts" / f"bars_30m_{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"未找到 30m 缓存: {path}\n请先运行 fg_30m/refresh_cache.py")
    df = pd.read_parquet(path)
    if "vt_symbol" not in df.columns:
        df["vt_symbol"] = f"{symbol}.CZCE"
    return df


def to_bar_data(row: pd.Series, vt_symbol: str) -> BarData:
    sym = vt_symbol.split(".")[0]
    vol = float(row.get("volume", 0) or 0)
    close = float(row["close"])
    turnover = vol * close * FG_SIZE
    return BarData(
        symbol=sym,
        exchange=Exchange.CZCE,
        datetime=pd.Timestamp(row["datetime"]).to_pydatetime(),
        interval=Interval.MINUTE,
        open_price=float(row["open"]),
        high_price=float(row["high"]),
        low_price=float(row["low"]),
        close_price=close,
        volume=vol,
        turnover=turnover,
        open_interest=float(row.get("open_interest", 0) or 0),
        gateway_name="FG30M",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="FG609", help="合约代码，如 FG609")
    parser.add_argument("--lab", default=str(LAB_PATH), help="AlphaLab 根目录")
    args = parser.parse_args()

    df = load_fg_bars(args.symbol)
    vt_symbol = str(df["vt_symbol"].iloc[0])

    lab = AlphaLab(args.lab)
    bars = [to_bar_data(row, vt_symbol) for _, row in df.iterrows()]
    lab.save_bar_data(bars)

    lab.add_contract_setting(
        vt_symbol=vt_symbol,
        long_rate=0.0001,
        short_rate=0.0001,
        size=FG_SIZE,
        pricetick=1.0,
    )

    out = Path(args.lab) / "minute" / f"{vt_symbol}.parquet"
    n = len(pl.read_parquet(out))
    print(f"已导入 {vt_symbol}: {n} 根 K 线（30m 序列，存入 minute/）")
    print(f"AlphaLab 路径: {args.lab}")
    print("下一步: python 03_alpha_demo_lgb.py")


if __name__ == "__main__":
    main()
