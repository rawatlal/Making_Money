from __future__ import annotations

import pandas as pd


class Rebalancer:
    """Determines portfolio rebalance dates."""

    def __init__(self, frequency="monthly"):
        self.frequency = frequency

    def get_rebalance_dates(self, trading_dates):
        """Return list of rebalance dates from the trading calendar.

        Args:
            trading_dates: DatetimeIndex of all trading dates.

        Returns:
            List of dates on which to rebalance.
        """
        if self.frequency == "monthly":
            return self._monthly(trading_dates)
        elif self.frequency == "quarterly":
            return self._quarterly(trading_dates)
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")

    def _monthly(self, dates):
        """First trading day of each month."""
        df = pd.Series(dates, index=dates)
        return df.groupby([df.index.year, df.index.month]).first().tolist()

    def _quarterly(self, dates):
        """First trading day of each quarter."""
        df = pd.Series(dates, index=dates)
        return df.groupby([df.index.year, df.index.quarter]).first().tolist()
