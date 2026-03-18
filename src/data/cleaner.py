import logging

import pandas as pd

logger = logging.getLogger("making_money")


class DataCleaner:
    """Clean and align price/fundamental data for factor computation."""

    def __init__(self, max_fill_days=5, max_missing_pct=0.20):
        self.max_fill_days = max_fill_days
        self.max_missing_pct = max_missing_pct

    def clean_prices(self, prices_df):
        """Clean price DataFrame: forward-fill, drop sparse tickers, align dates.

        Args:
            prices_df: DataFrame with MultiIndex columns (Price field, Ticker)
                       or single-level columns of close prices per ticker.

        Returns:
            DataFrame of clean close prices (date index, ticker columns).
        """
        # Extract close prices if MultiIndex
        if isinstance(prices_df.columns, pd.MultiIndex):
            if "Close" in prices_df.columns.get_level_values(0):
                close = prices_df["Close"]
            else:
                close = prices_df.iloc[
                    :, prices_df.columns.get_level_values(0) == "Close"
                ]
        else:
            close = prices_df

        # Forward-fill gaps (max N days)
        close = close.ffill(limit=self.max_fill_days)

        # Drop tickers with too much missing data
        missing_pct = close.isnull().mean()
        valid_tickers = missing_pct[missing_pct <= self.max_missing_pct].index
        dropped = len(close.columns) - len(valid_tickers)
        if dropped > 0:
            logger.info(f"Dropped {dropped} tickers with >{self.max_missing_pct:.0%} missing data")
        close = close[valid_tickers]

        # Drop any remaining rows that are all NaN
        close = close.dropna(how="all")

        logger.info(f"Clean prices: {close.shape[0]} dates x {close.shape[1]} tickers")
        return close

    def clean_fundamentals(self, fundamentals_df):
        """Clean fundamentals: drop rows with all NaN numeric fields."""
        numeric_cols = fundamentals_df.select_dtypes(include="number").columns
        valid = fundamentals_df.dropna(subset=numeric_cols, how="all")
        dropped = len(fundamentals_df) - len(valid)
        if dropped > 0:
            logger.info(f"Dropped {dropped} tickers with no fundamental data")
        return valid
