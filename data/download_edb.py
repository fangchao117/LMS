"""
信易 EDB 历史 K 线下载 —— 玻璃 FG 等国内期货

文档: https://doc.shinnytech.com/edb/latest/md_server.html

免费版:
  · 日线 period=86400  任意历史区间
  · 1 分钟 period=60   仅最近约 1 年（超出返回 403）

专业版:
  · 环境变量 SHINNY_USER / SHINNY_PASS 或 --user / --password 获取 token

示例:
    # 玻璃 FG505 日线 -> CSV
    python data/download_edb.py --symbol CZCE.FG505 --period daily \\
        --start "2024-12-20 00:00:00" --end "2025-04-20 15:00:00"

    # 玻璃主连日线 -> parquet
    python data/download_edb.py --symbol "KQ.m@CZCE.FG" --period daily \\
        --start "2024-12-20" --end "2025-04-20" --out data/FG_main_daily.parquet

    # 1 分钟 -> vnpy + 30 分钟缓存（fg_30m 回测直接用）
    python data/download_edb.py --symbol CZCE.FG609 --period 1m \\
        --start "2026-06-01 09:00:00" --end "2026-07-07 15:00:00" \\
        --import-vnpy --update-fg30m
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent
EDB_BASE = "https://edb.shinnytech.com"
KLINE_URL = f"{EDB_BASE}/md/kline"
TOKEN_URL = f"{EDB_BASE}/token"

PERIOD_DAILY = 86400
PERIOD_MINUTE = 60


def fetch_token(username: str, password: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        json={"username": username, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["token"]


def _parse_time(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=15, minute=0, second=0)
            return dt
        except ValueError:
            continue
    raise ValueError(f"invalid time: {s}")


def _edb_to_vnpy_symbol(edb_symbol: str) -> tuple[str, str]:
    """CZCE.FG505 -> (FG505, CZCE); KQ.m@CZCE.FG -> (FG, CZCE)"""
    if "@" in edb_symbol:
        # KQ.m@CZCE.FG
        part = edb_symbol.split("@", 1)[1]
        exch, sym = part.split(".", 1)
        return sym, exch
    exch, sym = edb_symbol.split(".", 1)
    return sym, exch


def download_kline(
    symbol: str,
    period: int,
    start_time: str,
    end_time: str,
    token: str | None = None,
    cols: str | None = None,
) -> pd.DataFrame:
    params: dict[str, str | int] = {
        "period": period,
        "symbol": symbol,
        "start_time": start_time,
        "end_time": end_time,
    }
    if cols:
        params["col"] = cols

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(KLINE_URL, params=params, headers=headers, timeout=120)
    if resp.status_code == 403:
        raise PermissionError(
            "403 免费用户仅可下日线 + 近 1 年分钟线。"
            "更早分钟数据需专业版: https://www.shinnytech.com/ "
            f"响应: {resp.text.strip()}"
        )
    resp.raise_for_status()
    if not resp.text.strip():
        return pd.DataFrame()

    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime_nano"], unit="ns")
    if "close_oi" in df.columns:
        df["open_interest"] = df["close_oi"]
    elif "open_oi" in df.columns:
        df["open_interest"] = df["open_oi"]
    else:
        df["open_interest"] = 0

    keep = ["datetime", "open", "high", "low", "close", "volume", "open_interest"]
    return df[[c for c in keep if c in df.columns]].sort_values("datetime").reset_index(drop=True)


def download_kline_chunked(
    symbol: str,
    period: int,
    start: datetime,
    end: datetime,
    token: str | None = None,
    chunk_days: int = 20,
) -> pd.DataFrame:
    """分钟线按块下载，避免单次请求过大。"""
    if period == PERIOD_DAILY:
        chunk_days = 3650

    parts: list[pd.DataFrame] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        st = cur.strftime("%Y-%m-%d %H:%M:%S")
        en = nxt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [edb] {symbol} {st} ~ {en} …")
        try:
            part = download_kline(symbol, period, st, en, token=token)
        except PermissionError:
            raise
        except requests.HTTPError as e:
            print(f"  [edb] skip chunk: {e}")
            part = pd.DataFrame()
        if not part.empty:
            parts.append(part)
        cur = nxt

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def import_minute_to_vnpy(df: pd.DataFrame, edb_symbol: str) -> int:
    if df.empty:
        return 0

    from vnpy.trader.constant import Exchange, Interval
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData

    sym, exch_str = _edb_to_vnpy_symbol(edb_symbol)
    if sym == "FG" and "@" in edb_symbol:
        raise ValueError("导入 vnpy 请用具体合约如 CZCE.FG505，主连 KQ.m@ 仅适合日线研究")

    exchange = Exchange(exch_str)
    bars: list[BarData] = []
    for row in df.itertuples(index=False):
        bars.append(
            BarData(
                symbol=sym,
                exchange=exchange,
                datetime=row.datetime.to_pydatetime(),
                interval=Interval.MINUTE,
                open_price=float(row.open),
                high_price=float(row.high),
                low_price=float(row.low),
                close_price=float(row.close),
                volume=float(row.volume),
                turnover=0.0,
                open_interest=float(row.open_interest),
                gateway_name="EDB",
            )
        )

    db = get_database()
    batch = 500
    for i in range(0, len(bars), batch):
        db.save_bar_data(bars[i : i + batch])

    try:
        from vnpy_ctastrategy.backtesting import load_bar_data

        load_bar_data.cache_clear()
    except Exception:
        pass

    return len(bars)


def resample_bars(df_1m: pd.DataFrame, minutes: int, vt_symbol: str) -> pd.DataFrame:
    pdf = df_1m.set_index("datetime")
    bar = pdf.resample(f"{minutes}min", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
        }
    )
    bar = bar.dropna(subset=["close"]).reset_index()
    bar["vt_symbol"] = vt_symbol
    return bar


def _merge_parquet(path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        old["datetime"] = pd.to_datetime(old["datetime"])
        df = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def update_fg30m_cache(df_1m: pd.DataFrame, edb_symbol: str, bar_minutes: int = 30) -> tuple[Path, Path, int]:
    """
    合并 1 分钟 parquet，重采样 30 分钟，写入 fg_30m/artifacts 缓存。
    返回 (1m_path, 30m_path, 30m_bars_count)
    """
    sym, exch = _edb_to_vnpy_symbol(edb_symbol)
    if sym == "FG" and "@" in edb_symbol:
        raise ValueError("30 分钟缓存请用具体合约，如 CZCE.FG609")

    vt_symbol = f"{sym}.{exch}"
    fg30_art = ROOT / "fg_30m" / "artifacts"
    fg30_art.mkdir(parents=True, exist_ok=True)

    path_1m = DATA_DIR / f"bars_1m_{sym}.parquet"
    df_1m_full = _merge_parquet(path_1m, df_1m)
    df_1m_full.to_parquet(path_1m, index=False)

    df_30 = resample_bars(df_1m_full, bar_minutes, vt_symbol)
    path_30 = fg30_art / f"bars_30m_{sym}.parquet"
    df_30.to_parquet(path_30, index=False)

    return path_1m, path_30, len(df_30)


def main() -> None:
    parser = argparse.ArgumentParser(description="信易 EDB K 线下载")
    parser.add_argument("--symbol", default="CZCE.FG505", help="如 CZCE.FG505, KQ.m@CZCE.FG")
    parser.add_argument(
        "--period", choices=("daily", "1m"), default="daily",
        help="daily=日线(免费全历史), 1m=1分钟(免费近1年)",
    )
    parser.add_argument("--start", default="2024-12-20 00:00:00")
    parser.add_argument("--end", default="2025-04-20 15:00:00")
    parser.add_argument("--out", default="", help="输出路径 .csv / .parquet，默认自动命名")
    parser.add_argument("--user", default=os.environ.get("SHINNY_USER", ""))
    parser.add_argument("--password", default=os.environ.get("SHINNY_PASS", ""))
    parser.add_argument("--import-vnpy", action="store_true", help="1 分钟数据写入 vnpy SQLite")
    parser.add_argument(
        "--update-fg30m", action="store_true",
        help="1 分钟 -> 合并 parquet -> 合成 30 分钟 -> fg_30m/artifacts 缓存",
    )
    parser.add_argument("--no-update-fg30m", action="store_true", help="禁用 30 分钟缓存更新")
    parser.add_argument("--chunk-days", type=int, default=20, help="分钟线下载分块天数")
    args = parser.parse_args()

    period = PERIOD_DAILY if args.period == "daily" else PERIOD_MINUTE
    start_dt = _parse_time(args.start)
    end_dt = _parse_time(args.end)
    if start_dt >= end_dt:
        raise SystemExit("start 必须早于 end")

    token: str | None = None
    if args.user and args.password:
        print("[edb] 登录信易获取 token …")
        token = fetch_token(args.user, args.password)

    print(f"[edb] 下载 {args.symbol} period={args.period} {args.start} ~ {args.end}")
    df = download_kline_chunked(
        args.symbol, period, start_dt, end_dt,
        token=token, chunk_days=args.chunk_days,
    )

    if df.empty:
        print("[edb] 无数据")
        sys.exit(1)

    print(f"[edb] 共 {len(df)} 行  {df['datetime'].min()} ~ {df['datetime'].max()}")

    if args.out:
        out = Path(args.out)
    else:
        tag = args.symbol.replace("@", "_at_").replace(".", "_")
        ext = "parquet" if args.period == "daily" else "csv"
        out = DATA_DIR / f"edb_{tag}_{args.period}_{start_dt.date()}_{end_dt.date()}.{ext}"

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"[edb] -> {out}")

    if args.import_vnpy:
        if args.period != "1m":
            raise SystemExit("--import-vnpy 仅用于 1 分钟数据")
        n = import_minute_to_vnpy(df, args.symbol)
        print(f"[edb] 已导入 vnpy {n} 根 1 分钟 K 线")

    do_fg30 = args.update_fg30m or (args.period == "1m" and not args.no_update_fg30m)
    if do_fg30:
        if args.period != "1m":
            raise SystemExit("--update-fg30m 仅用于 1 分钟数据")
        p1, p30, n30 = update_fg30m_cache(df, args.symbol)
        print(f"[edb] 1m 合并 -> {p1}")
        print(f"[edb] 30m 缓存 {n30} 根 -> {p30}")
        print(f"[edb] fg_30m 回测: cd fg_30m && python backtest.py --symbol {_edb_to_vnpy_symbol(args.symbol)[0]}")


if __name__ == "__main__":
    main()
