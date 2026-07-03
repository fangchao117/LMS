"""
步骤 01：准备 AlphaLab 数据仓库

把玻璃日线行情写入 vnpy AlphaLab（落地为 parquet），并登记合约交易参数。
后续训练与回测都从这个 lab 读取，只需运行一次（数据更新后重跑即可覆盖合并）。
"""
from __future__ import annotations

from vnpy.alpha import AlphaLab

import config
import data_loader


def main() -> None:
    config.ensure_dirs()

    lab: AlphaLab = AlphaLab(str(config.LAB_PATH))

    # 写入日线行情
    bars = data_loader.build_bar_data()
    lab.save_bar_data(bars)
    print(f"[01] 已写入行情 {len(bars)} 根：{bars[0].datetime.date()} -> {bars[-1].datetime.date()}")

    # 登记合约参数（手续费率 / 合约乘数 / 最小变动价位）
    lab.add_contract_setting(
        vt_symbol=config.VT_SYMBOL,
        long_rate=config.LONG_RATE,
        short_rate=config.SHORT_RATE,
        size=config.CONTRACT_SIZE,
        pricetick=config.PRICE_TICK,
    )
    print(f"[01] 已登记合约参数 {config.VT_SYMBOL}："
          f"size={config.CONTRACT_SIZE}, tick={config.PRICE_TICK}, "
          f"rate={config.LONG_RATE}")
    print(f"[01] AlphaLab 目录：{config.LAB_PATH}")


if __name__ == "__main__":
    main()
