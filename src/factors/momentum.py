from __future__ import annotations

import pandas as pd

from src.factors.base import Factor


class MomentumFactor(Factor):
    """Momentum factor: total return over lookback period, skipping most recent month."""

    direction = "higher_is_better"

    def __init__(self, lookback_days, name=None):
        self.lookback_days = lookback_days
        self.skip_days = 21  # Skip most recent month to avoid short-term reversal
        self.name = name or f"momentum_{lookback_days}d"

    def compute(self, prices, fundamentals, date):
        # Use only data up to the given date (no lookahead)
        available = prices.loc[:date]
        if len(available) < self.lookback_days + self.skip_days:
            return pd.Series(dtype=float)

        # End price is skip_days ago, start price is lookback_days + skip_days ago
        end_prices = available.iloc[-(self.skip_days + 1)]
        start_idx = -(self.lookback_days + self.skip_days)
        if abs(start_idx) > len(available):
            return pd.Series(dtype=float)
        start_prices = available.iloc[start_idx]

        # Total return
        returns = (end_prices / start_prices) - 1
        return returns.dropna()


# Convenience instances
Momentum3M = lambda: MomentumFactor(63, name="momentum_3m")
Momentum6M = lambda: MomentumFactor(126, name="momentum_6m")
Momentum12M = lambda: MomentumFactor(252, name="momentum_12m")
