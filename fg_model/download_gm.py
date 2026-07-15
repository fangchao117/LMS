"""
东财掘金 —— 下载期货 30 分钟 K 线

仅写入 fg_model/data/，不修改 fd_vib 策略。

准备：
  1. 注册 https://emquant.18.cn/  申请 Token（用户 → 密钥管理）
  2. pip install gm
  3. 复制 gm_token.example.txt → gm_token.txt 填入 Token
     或设置环境变量 GM_TOKEN

示例：
    cd D:\\LMS\\fg_model
    python download_gm.py --symbol FG609
    python download_gm.py --symbol FG609 --start 2025-06-20 --end 2026-07-08
    python download_gm.py --continuous          # 主力连续 CZCE.FG
    python download_gm.py --symbol FG609 --sync-fg30m   # 另存到 fg_30m 对比
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config

DATA_DIR = config.ROOT / "data"
TOKEN_FILE = config.ROOT / "gm_token.txt"
FREQ_30M = "1800s"  # 掘金：30 分钟 = 1800 秒
CHUNK_DAYS = 28
# 掘金免费档：期货 Bar 约最近 180 自然日（错误码 2002 会提示最早日期）
GM_MAX_HISTORY_DAYS = 180


def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    env = os.environ.get("GM_TOKEN", "").strip()
    if env:
        return env
    if TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if t and not t.startswith("在此填入"):
            return t
    raise SystemExit(
        "未找到掘金 Token。请：\n"
        "  1) 复制 gm_token.example.txt → gm_token.txt 填入 Token\n"
        "  2) 或 set GM_TOKEN=xxx\n"
        "  3) 或 --token xxx\n"
        "申请地址: https://emquant.18.cn/ → 用户 → 密钥管理"
    )


def gm_symbol(symbol: str, continuous: bool) -> str:
    if continuous:
        # 玻璃主力连续
        return "CZCE.FG"
    sym = symbol.upper().replace(".CZCE", "")
    return f"CZCE.{sym}"


def _fetch_chunk(gm_symbol_code: str, start: datetime, end: datetime) -> pd.DataFrame:
    from gm.api import history, set_token  # noqa: WPS433 — 延迟导入

    # set_token 在 main 已调用；history 需终端/SDK 网络
    df = history(
        symbol=gm_symbol_code,
        frequency=FREQ_30M,
        start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        fields="open,high,low,close,volume,amount,position,bob,eob",
        df=True,
    )
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def normalize_gm_df(df: pd.DataFrame, vt_symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df["bob"] if "bob" in df.columns else df.get("eob"))
    for src, dst in [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("position", "open_interest"),
    ]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    if "amount" in df.columns:
        out["turnover"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        out["turnover"] = out["close"] * out["volume"] * 20.0
    out["vt_symbol"] = vt_symbol
    out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return out.reset_index(drop=True)


def clamp_start_for_gm(start: str, end: str) -> str:
    """免费 Token 常限 180 日 Bar，将 start 钳到允许范围内。"""
    end_ts = pd.Timestamp(end)
    earliest = end_ts - timedelta(days=GM_MAX_HISTORY_DAYS - 1)
    start_ts = pd.Timestamp(start)
    if start_ts < earliest:
        clamped = earliest.strftime("%Y-%m-%d")
        print(f"[gm] 提示: start 由 {start} 调整为 {clamped}（账号 Bar 历史约 {GM_MAX_HISTORY_DAYS} 日）")
        return clamped
    return start


def download_30m(
    symbol: str,
    start: str,
    end: str,
    token: str,
    continuous: bool = False,
) -> pd.DataFrame:
    from gm.api import set_token

    set_token(token)
    gm_code = gm_symbol(symbol, continuous)
    vt = gm_code if continuous else f"{symbol.upper().replace('.CZCE', '')}.CZCE"

    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    parts: list[pd.DataFrame] = []
    cur = t0

    print(f"[gm] 下载 {gm_code}  30m({FREQ_30M})  {start} ~ {end}")
    while cur < t1:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), t1)
        try:
            raw = _fetch_chunk(gm_code, cur.to_pydatetime(), chunk_end.to_pydatetime())
        except Exception as e:
            msg = str(e)
            if "ERR_NO_DATA_PERMISSION" in msg or "2002" in msg:
                raise RuntimeError(
                    f"掘金数据权限不足 {cur.date()}~{chunk_end.date()}: {msg}\n"
                    f"免费档期货 Bar 约仅 {GM_MAX_HISTORY_DAYS} 日，请缩小 --start 或升级权限。"
                ) from e
            raise RuntimeError(
                f"掘金请求失败 {cur.date()}~{chunk_end.date()}: {e}\n"
                "请确认：Token 有效、掘金终端已登录（部分环境需要）、网络正常。"
            ) from e
        if not raw.empty:
            parts.append(normalize_gm_df(raw, vt))
            print(f"  {cur.date()} ~ {chunk_end.date()}  bars={len(raw)}")
        else:
            print(f"  {cur.date()} ~ {chunk_end.date()}  (空)")
        cur = chunk_end

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["datetime"]).sort_values("datetime")


def main() -> dict:
    parser = argparse.ArgumentParser(description="东财掘金下载 30 分钟期货 K 线")
    parser.add_argument("--symbol", default="FG609", help="合约，如 FG609")
    parser.add_argument("--continuous", action="store_true", help="主力连续 CZCE.FG")
    parser.add_argument("--start", default="2025-06-20")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--sync-fg30m",
        action="store_true",
        help="额外写入 fg_30m/artifacts 供对比（不影响 fd_vib）",
    )
    args = parser.parse_args()

    token = resolve_token(args.token)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start = clamp_start_for_gm(args.start, args.end)
    df = download_30m(args.symbol, start, args.end, token, args.continuous)
    if df.empty:
        raise SystemExit("未下载到任何数据，请检查合约代码、日期范围、Token 权限。")

    tag = "FG_continuous" if args.continuous else args.symbol.upper().replace(".CZCE", "")
    out = DATA_DIR / f"bars_30m_gm_{tag}.parquet"
    df.to_parquet(out, index=False)

    meta = {
        "source": "eastmoney_gm",
        "frequency": FREQ_30M,
        "symbol": tag,
        "gm_code": gm_symbol(args.symbol, args.continuous),
        "start": start,
        "end": args.end,
        "bars": len(df),
        "datetime_min": str(df["datetime"].min()),
        "datetime_max": str(df["datetime"].max()),
        "path": str(out),
    }
    meta_path = DATA_DIR / f"bars_30m_gm_{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.sync_fg30m and not args.continuous:
        sync_path = config.DATA_ROOT / "fg_30m" / "artifacts" / f"bars_30m_{tag}_gm.parquet"
        df.to_parquet(sync_path, index=False)
        meta["sync_fg30m"] = str(sync_path)
        print(f"  同步对比 -> {sync_path}")

    print(f"\n完成: {len(df)} 根 30m K 线")
    print(f"  时间 {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"-> {out}")
    print(f"-> {meta_path}")
    return meta


if __name__ == "__main__":
    main()
