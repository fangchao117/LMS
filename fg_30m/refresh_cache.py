"""
从 1 分钟数据重建 30 分钟缓存（vnpy DB 或 data/bars_1m_*.parquet）

    python refresh_cache.py
    python refresh_cache.py --symbol FG609
"""
from __future__ import annotations

import argparse

import config
import data


def main(symbol: str | None = None) -> None:
    sym = symbol or config.SYMBOL
    config.SYMBOL = sym
    config.VT_SYMBOL = f"{sym}.{config.EXCHANGE_STR}"
    config.ensure_dirs()

    print(f"[refresh_cache] 加载 {sym} 1 分钟 …")
    df_1m = data.load_1m(sym, prefer="vnpy")
    print(f"  1m {len(df_1m)} 根  {df_1m['datetime'].min()} ~ {df_1m['datetime'].max()}")

    df_30 = data.resample_30m(df_1m)
    cache_30 = config.ARTIFACT_PATH / f"bars_30m_{sym}.parquet"
    cache_1m = config.ROOT / "data" / f"bars_1m_{sym}.parquet"
    df_30.to_parquet(cache_30, index=False)
    df_1m.to_parquet(cache_1m, index=False)

    print(f"  30m {len(df_30)} 根 -> {cache_30}")
    print(f"  1m 备份 -> {cache_1m}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=config.SYMBOL)
    main(p.parse_args().symbol)
