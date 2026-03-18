from __future__ import annotations

import abc

import pandas as pd


class Factor(abc.ABC):
    """Abstract base class for all factor computations."""

    name: str
    direction: str  # "higher_is_better" or "lower_is_better"

    @abc.abstractmethod
    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Compute raw factor values for all tickers as of the given date.

        Args:
            prices: DataFrame of close prices (date index, ticker columns).
                    Only data up to `date` should be used.
            fundamentals: DataFrame of fundamental data (ticker index).
            date: The as-of date for factor computation.

        Returns:
            Series indexed by ticker with raw factor values.
        """
        ...
