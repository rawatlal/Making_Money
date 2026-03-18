import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger("making_money")


class YFinanceDownloader:
    """Batch downloader for price and fundamental data via yfinance."""

    def __init__(self, chunk_size=50, max_retries=3):
        self.chunk_size = chunk_size
        self.max_retries = max_retries

    def download_prices(self, tickers, years=5):
        """Download daily OHLCV price data for all tickers."""
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        all_data = []

        chunks = [
            tickers[i : i + self.chunk_size]
            for i in range(0, len(tickers), self.chunk_size)
        ]

        for chunk in tqdm(chunks, desc="Downloading prices"):
            df = self._download_with_retry(chunk, start, end)
            if df is not None and not df.empty:
                all_data.append(df)

        if not all_data:
            logger.error("No price data downloaded")
            return pd.DataFrame()

        combined = pd.concat(all_data, axis=1)
        # Ensure we have a multi-level column (ticker, field)
        if isinstance(combined.columns, pd.MultiIndex):
            return combined
        return combined

    def _download_with_retry(self, tickers, start, end):
        """Download with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                df = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                return df
            except Exception as e:
                wait = 2**attempt
                logger.warning(
                    f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait}s"
                )
                time.sleep(wait)
        return None

    def download_fundamentals(self, tickers):
        """Download fundamental data (P/E, P/B, EV, EBITDA, sector, market cap)."""
        records = []
        for ticker in tqdm(tickers, desc="Downloading fundamentals"):
            try:
                info = yf.Ticker(ticker).info
                records.append(
                    {
                        "ticker": ticker,
                        "pe_ratio": info.get("trailingPE"),
                        "pb_ratio": info.get("priceToBook"),
                        "enterprise_value": info.get("enterpriseValue"),
                        "ebitda": info.get("ebitda"),
                        "market_cap": info.get("marketCap"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {ticker}: {e}")
                continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("ticker")
