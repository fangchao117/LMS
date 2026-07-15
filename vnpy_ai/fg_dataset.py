"""玻璃期货 Alpha 因子数据集（供训练/加载 pickle 共用）。"""
from vnpy.alpha import AlphaDataset


class FgAlphaDemo(AlphaDataset):
    """精简因子集，加快单合约 demo。"""

    def __init__(self, df, train_period, valid_period, test_period):
        super().__init__(df, train_period, valid_period, test_period)
        self.add_feature("kmid", "(close - open) / open")
        self.add_feature("klen", "(high - low) / open")
        for w in (5, 10, 20, 30):
            self.add_feature(f"roc_{w}", f"ts_delay(close, {w}) / close")
            self.add_feature(f"ma_{w}", f"ts_mean(close, {w}) / close")
            self.add_feature(f"std_{w}", f"ts_std(close, {w}) / close")
        self.add_feature("oi_chg", "open_interest / ts_delay(open_interest, 1) - 1")
        self.add_feature("vol_ma10", "volume / ts_mean(volume, 10)")
